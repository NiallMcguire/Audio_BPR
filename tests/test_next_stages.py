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
from mv_dataloader import DynamicMaskingDataloader
from mv_models import create_model
from mv_training import train_model
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
                             "export HF_HUB_OFFLINE=1", "--time=3-00:00:00"):
                self.assertIn(expected, script)


if __name__ == "__main__":
    unittest.main()
