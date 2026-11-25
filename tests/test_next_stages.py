import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

AUDIO_BPR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_BPR_DIR))

from fm_encoders import FMInputAdapter, channel_layout, load_montages
from mv_dataloader import DynamicMaskingDataloader, _question_group_id
from mv_models import create_model
from mv_training import _limit_wandb_tags, test_model, train_model
from tests.test_experiment_pipeline import TinyTokenizer, write_synthetic_pairs

try:
    import braindecode  # noqa: F401
    HAS_BRAINDECODE = True
except ImportError:
    HAS_BRAINDECODE = False


def split_pairs(pairs, strategy, split):
    dataset = DynamicMaskingDataloader(
        combined_dataset={"ict_pairs": pairs, "metadata": {}}, tokenizer=TinyTokenizer(), max_text_len=32,
        split=split, global_eeg_dims=(2, 8, 2), split_strategy=strategy)
    return [(pair["dataset_source"], pair["original_doc_text"], pair["query_text"]) for pair in dataset.processed_pairs]


class NextStageTests(unittest.TestCase):
    def test_wandb_tags_are_bounded_and_long_tags_remain_distinct(self):
        tags = ["short", "condition-gen_query_10fold_transformer_typing_continuous_fold01_seed42",
                "condition-gen_subject_10fold_transformer_typing_continuous_fold01_seed42"]
        bounded = _limit_wandb_tags(tags)
        self.assertEqual(bounded[0], "short")
        self.assertTrue(all(1 <= len(tag) <= 64 for tag in bounded))
        self.assertNotEqual(bounded[1], bounded[2])

    def test_transformer_sequence_limit_uses_configured_eeg_length(self):
        model = create_model(hidden_dim=8, eeg_arch="transformer", device="cpu", use_lora=False,
                             pooling_strategy="mean", encoder_type="dual", global_eeg_dims=(100, 8, 2),
                             query_type="eeg", use_pretrained_text=False)
        eeg = torch.randn(2, 100, 8, 2)
        with torch.no_grad():
            encoded = model.encode_eeg(eeg, torch.ones(2, 32))
        self.assertEqual(model.eeg_encoder.max_seq_len, 100)
        self.assertEqual(tuple(encoded.shape), (2, 1, 8))

    def test_test_model_computes_noise_at_every_masking_level(self):
        class DatasetStub:
            unique_subjects = {"participant"}

            def __init__(self):
                self.probability = 0.9

            def get_current_masking_probability(self):
                return self.probability

            def set_masking_probability(self, probability):
                self.probability = probability

            def __len__(self):
                return 1

        dataloader = SimpleNamespace(dataset=DatasetStub())
        levels = [0, 25, 50, 75, 90, 100]
        noise_calls = []

        def fake_noise(_model, _loader, _device, masking_level, metric_prefix):
            noise_calls.append(masking_level)
            return {f"{metric_prefix}/rr": masking_level / 100}

        with patch("mv_training.perform_dual_encoder_test_ranking_at_masking_level",
                   return_value={"test_ranking/main/rr": 0.2}), \
             patch("mv_training.detect_dataset_sources", return_value={"has_multiple": False}), \
             patch("mv_training.perform_noise_ranking_evaluation", side_effect=fake_noise), \
             patch("mv_training.calculate_masking_performance_trends", return_value={}), \
             patch("mv_training.print_multi_masking_test_summary"), \
             patch("mv_training.wandb.log"):
            metrics = test_model(torch.nn.Identity(), dataloader, "cpu", test_masking_levels=levels)

        self.assertEqual(noise_calls, levels)
        self.assertEqual(metrics["test_ranking/noise/rr"], 0.9)
        self.assertEqual(dataloader.dataset.probability, 0.9)

    def test_original_fixed_split_keeps_task_test_set_when_sources_are_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = np.load(write_synthetic_pairs(root / "qr.npy", "authentic_query_response", 80, False),
                           allow_pickle=True).item()["ict_pairs"]
            external = np.load(write_synthetic_pairs(root / "alice.npy", "alice", 120, True),
                               allow_pickle=True).item()["ict_pairs"]
            for split in ("train", "val", "test"):
                # One source: identical to the original Audio BPR split.
                self.assertEqual(split_pairs(task, "original_fixed", split), split_pairs(task, "original", split))
                alone = split_pairs(task, "original_fixed", split)
                mixed = [row for row in split_pairs(task + external, "original_fixed", split)
                         if row[0] == "authentic_query_response"]
                self.assertEqual(sorted(alone), sorted(mixed), split)
            pooled = [row for row in split_pairs(task + external, "original", "test")
                      if row[0] == "authentic_query_response"]
            self.assertNotEqual(sorted(pooled), sorted(split_pairs(task, "original", "test")))

    def test_subject_10fold_keeps_people_disjoint_and_tests_each_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_synthetic_pairs(Path(temp_dir) / "task.npy", "authentic_query_response", 400, False)
            pairs = np.load(path, allow_pickle=True).item()["ict_pairs"]
            all_participants = {pair["participant_id"] for pair in pairs}
            tested = set()
            for fold in range(1, 11):
                groups = {}
                for split in ("train", "val", "test"):
                    dataset = DynamicMaskingDataloader(
                        combined_dataset={"ict_pairs": pairs, "metadata": {}}, tokenizer=TinyTokenizer(),
                        max_text_len=32, split=split, global_eeg_dims=(2, 8, 2),
                        split_strategy="subject_10fold", fold=fold, pair_split_seed=42)
                    groups[split] = {pairs[index]["participant_id"] for index in dataset.indices}
                self.assertFalse(groups["train"] & groups["val"])
                self.assertFalse(groups["train"] & groups["test"])
                self.assertFalse(groups["val"] & groups["test"])
                tested.update(groups["test"])
            self.assertEqual(tested, all_participants)

    def test_query_10fold_groups_same_question_across_people(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_synthetic_pairs(Path(temp_dir) / "task.npy", "authentic_query_response", 400, False)
            pairs = np.load(path, allow_pickle=True).item()["ict_pairs"]
            all_questions = {pair["question_document_id"] for pair in pairs}
            tested = set()
            for fold in range(1, 11):
                groups = {}
                for split in ("train", "val", "test"):
                    dataset = DynamicMaskingDataloader(
                        combined_dataset={"ict_pairs": pairs, "metadata": {}}, tokenizer=TinyTokenizer(),
                        max_text_len=32, split=split, global_eeg_dims=(2, 8, 2),
                        split_strategy="query_10fold", fold=fold, pair_split_seed=42)
                    groups[split] = {pairs[index]["question_document_id"] for index in dataset.indices}
                self.assertFalse(groups["train"] & groups["val"])
                self.assertFalse(groups["train"] & groups["test"])
                self.assertFalse(groups["val"] & groups["test"])
                tested.update(groups["test"])
            self.assertEqual(tested, all_questions)

    def test_mixed_source_rotations_include_each_source_and_keep_shared_queries_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_path = write_synthetic_pairs(root / "task.npy", "authentic_query_response", 400, False)
            alice_path = write_synthetic_pairs(root / "alice.npy", "alice", 120, True)
            nieuwland_path = write_synthetic_pairs(root / "nieuwland.npy", "nieuwland", 120, True)
            task = np.load(task_path, allow_pickle=True).item()["ict_pairs"]
            alice = np.load(alice_path, allow_pickle=True).item()["ict_pairs"]
            nieuwland = np.load(nieuwland_path, allow_pickle=True).item()["ict_pairs"]
            shared_text = task[0]["query_text"]
            alice[0]["doc_text"] = shared_text
            alice[0]["full_sentence_text"] = shared_text
            alice[0]["doc_words"] = shared_text.split()
            alice[0]["question_document_id"] = task[0]["question_document_id"]
            pairs = task + alice + nieuwland

            for strategy in ("subject_10fold", "query_10fold"):
                split_sources, split_groups = {}, {}
                shared_group_splits = {}
                for split in ("train", "val", "test"):
                    dataset = DynamicMaskingDataloader(
                        combined_dataset={"ict_pairs": pairs, "metadata": {}}, tokenizer=TinyTokenizer(),
                        max_text_len=32, split=split, global_eeg_dims=(2, 8, 2),
                        split_strategy=strategy, fold=1, pair_split_seed=42)
                    selected = [pairs[index] for index in dataset.indices]
                    split_sources[split] = {pair["dataset_source"] for pair in selected}
                    if strategy == "subject_10fold":
                        split_groups[split] = {(pair["training_source"], pair["participant_id"])
                                               for pair in selected}
                    else:
                        split_groups[split] = {_question_group_id(pair) for pair in selected}
                        for pair in selected:
                            if _question_group_id(pair) == _question_group_id(task[0]):
                                shared_group_splits.setdefault(split, set()).add(pair["dataset_source"])

                self.assertEqual(split_sources, {split: {"authentic_query_response", "alice", "nieuwland"}
                                                 for split in ("train", "val", "test")})
                self.assertFalse(split_groups["train"] & split_groups["val"])
                self.assertFalse(split_groups["train"] & split_groups["test"])
                self.assertFalse(split_groups["val"] & split_groups["test"])
                if strategy == "query_10fold":
                    self.assertEqual(len(shared_group_splits), 1)
                    self.assertEqual(next(iter(shared_group_splits.values())),
                                     {"authentic_query_response", "alice"})

    def test_best_checkpoint_copy_restores_the_best_epoch(self):
        def fake_epoch(model, **_):
            with torch.no_grad():
                model.weight.add_(1.0)
            return 0.0, 0.0, 0.0

        for copy, expected in ((True, 2.0), (False, 4.0)):
            losses = iter([1.0, 0.5, 0.9, 0.8])
            model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(model.weight)
            with patch("mv_training.initialize_wandb"), patch("mv_training.wandb.log"), \
                    patch("mv_training.train_epoch", side_effect=fake_epoch), \
                    patch("mv_training.validate_epoch", side_effect=lambda **_: (next(losses), 0.0)):
                train_model(model, None, None, torch.optim.SGD(model.parameters(), lr=0.0), num_epochs=4,
                            patience=10, device="cpu",
                            config={"best_checkpoint_copy": copy, "training_masking_level": 90})
            self.assertEqual(model.weight.item(), expected, f"best_checkpoint_copy={copy}")

    def test_attention_pooling_weights_cover_active_units_only(self):
        model = create_model(hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                             pooling_strategy="mean", encoder_type="dual", global_eeg_dims=(5, 8, 2),
                             query_type="eeg", use_pretrained_text=False, eeg_pooling="attention")
        model.eval()
        eeg = torch.randn(2, 5, 8, 2)
        eeg[1, 3:] = 0
        with torch.no_grad():
            pooled = model.encode_eeg(eeg, torch.ones(2, 32))
        self.assertEqual(tuple(pooled.shape), (2, 1, 8))
        weights = model.last_eeg_attention
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2)))
        self.assertTrue(torch.all(weights[1, 3:] == 0))
        self.assertEqual(model.pooling_strategy, "mean")

    def test_eeg_only_pooling_options_keep_mean_text_pooling(self):
        for pooling in ("max", "cls", "attention"):
            model = create_model(hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                                 pooling_strategy="mean", encoder_type="dual", global_eeg_dims=(5, 8, 2),
                                 query_type="eeg", use_pretrained_text=False, eeg_pooling=pooling)
            model.set_tokenizer_vocab_size(256)
            ids = torch.randint(4, 200, (3, 6))
            text = model.encode_text(ids, torch.ones_like(ids))
            eeg = model.encode_eeg(torch.randn(3, 5, 8, 2), torch.ones(3, 32))
            self.assertEqual(tuple(text.shape), (3, 1, 8), pooling)
            self.assertEqual(tuple(eeg.shape), (3, 1, 8), pooling)

    def test_fm_adapter_maps_task_and_alice_channels_by_name(self):
        montages = load_montages()
        task_names = montages["authentic"]["channel_names"]
        alice_names = montages["alice"]["mapped_names"]
        adapter = FMInputAdapter("cbramod")
        layout = [name for name, _ in channel_layout("cbramod", montages)]
        rng = np.random.default_rng(0)

        task_units = np.zeros((3, 513, 128), dtype=np.float32)
        task_units[:2, :, :64] = rng.normal(size=(2, 513, 64))
        out = adapter(task_units[:2], "authentic_query_response")
        self.assertEqual(out.shape, (2, 200, len(layout)))
        self.assertTrue(np.all(np.abs(out).max(axis=1) > 0))
        self.assertTrue(np.allclose(out.mean(axis=1), 0, atol=1e-5))
        # The ear channels are references, not scalp inputs.
        self.assertNotIn("A1", layout)
        self.assertEqual(len(layout), len(montages["authentic"]["scalp_channels"]))

        alice_units = np.zeros((1, 513, 128), dtype=np.float32)
        alice_units[0, :85, :62] = rng.normal(size=(85, 62))
        out = adapter(alice_units, "alice")
        mapped = {name.upper() for name in alice_names if name}
        for c, name in enumerate(layout):
            active = np.abs(out[0, :, c]).max() > 0
            self.assertEqual(active, name in mapped, name)
        # 85 samples at 128 Hz -> 133 samples at 200 Hz; the rest stays zero padding.
        self.assertTrue(np.all(out[0, 134:] == 0))
        nieuwland = np.zeros((1, 513, 128), dtype=np.float32)
        nieuwland[0, :250, :22] = rng.normal(size=(250, 22))
        out = adapter(nieuwland, "nieuwland")
        self.assertEqual(int((np.abs(out[0]).max(axis=0) > 0).sum()), 22)
        self.assertTrue(np.all(out[0, 100:] == 0))

    def test_biot_layout_uses_ears_for_mastoid_derivations(self):
        adapter = FMInputAdapter("biot")
        self.assertEqual(adapter.n_channels, 18)
        units = np.random.default_rng(1).normal(size=(1, 513, 64)).astype(np.float32)
        out = adapter(units, "authentic_typing_continuous_query_response")
        self.assertTrue(np.all(np.abs(out[0]).max(axis=0) > 0))

    @unittest.skipUnless(HAS_BRAINDECODE, "braindecode is installed only in the foundation-model environment")
    def test_foundation_model_encoder_frozen_and_top(self):
        for finetune in ("frozen", "top"):
            model = create_model(hidden_dim=8, eeg_arch="labram", device="cpu", use_lora=False,
                                 pooling_strategy="mean", encoder_type="dual", global_eeg_dims=(4, 200, 62),
                                 query_type="eeg", use_pretrained_text=False,
                                 fm_options={"finetune": finetune, "unfreeze_blocks": 2})
            encoder = model.eeg_encoder
            trainable = {name for name, p in encoder.backbone.named_parameters() if p.requires_grad}
            if finetune == "frozen":
                self.assertFalse(trainable)
            else:
                self.assertTrue(trainable)
                self.assertTrue(all(name.startswith(("blocks.10.", "blocks.11.", "fc_norm")) for name in trainable))
            model.train()
            self.assertEqual(encoder.backbone.training, finetune != "frozen")
            eeg = torch.randn(2, 4, 200, 62)
            eeg[1, 2:] = 0
            pooled = model.encode_eeg(eeg, torch.ones(2, 32))
            self.assertEqual(tuple(pooled.shape), (2, 1, 8))
            pooled.sum().backward()
            self.assertIsNotNone(encoder.projection.weight.grad)

    def test_stage_scheduler_prepares_without_submitting(self):
        import stage_scheduler

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(pilot=True, models=["labram", "reve"], best=None, only_fm=False,
                                   typing_format="continuous", python=Path("/usr/bin/python3"),
                                   exclude_nodes=["gpu04"], wandb_project="test", date="20990101",
                                   output_dir=temp_dir, overwrite=False)
            with patch.object(Path, "is_file", return_value=True):
                output = stage_scheduler.prepare("stage1a", args)
            matrix = json.loads((output / "experiment_matrix.json").read_text())
            self.assertFalse(matrix["submitted"])
            self.assertEqual(len(matrix["matrix"]), 2 * 3)
            script = (output / "jobs" / "s1a_reve_top_reading_seed42.sbatch").read_text()
            for expected in ("--eeg_arch reve", "--fm-finetune top", "--split-strategy original_fixed",
                             "--best-checkpoint-copy", "--primary-eval-source authentic_query",
                             "export HF_HUB_OFFLINE=1", "--time=2-00:00:00"):
                self.assertIn(expected, script)


if __name__ == "__main__":
    unittest.main()
