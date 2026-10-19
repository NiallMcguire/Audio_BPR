import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

AUDIO_BPR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_BPR_DIR))

from Dataset.authentic_query_reader import build_task_pairs
from controller import create_dataloaders, create_source_test_loaders
from mv_dataloader import compute_global_eeg_dimensions, load_combined_datasets
from mv_models import create_model
from mv_training import (build_document_database, compute_contrastive_loss,
                         perform_ranking_validation, test_model)
from experiment_scheduler import EXPERIMENTS, prepare_jobs


class TinyTokenizer:
    mask_token_id = 3
    sep_token_id = 2

    def __len__(self):
        return 256

    def encode_plus(self, text, add_special_tokens=True, max_length=32, truncation=True,
                    padding=None, return_tensors="pt"):
        import hashlib

        words = str(text).split()
        ids = [1] if add_special_tokens else []
        ids.extend(4 + int(hashlib.sha1(word.encode()).hexdigest()[:6], 16) % 240 for word in words)
        if add_special_tokens:
            ids.append(self.sep_token_id)
        if truncation:
            ids = ids[:max_length]
            if add_special_tokens and ids[-1] != self.sep_token_id:
                ids[-1] = self.sep_token_id
        mask = [1] * len(ids)
        if padding == "max_length":
            pad = max(0, max_length - len(ids))
            ids.extend([0] * pad)
            mask.extend([0] * pad)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def write_synthetic_pairs(path: Path, source: str, count: int, supports_masking: bool) -> Path:
    pairs = []
    for index in range(count):
        task_ict = source == "authentic_query"
        target_index = index // 4 if task_ict else index
        if supports_masking or task_ict:
            question = f"queryword{target_index} context{target_index}"
            document = f"before{target_index} queryword{target_index} context{target_index} after{target_index}"
            words = document.split()
            query_start, query_end = 1, 3
        else:
            question = f"What fact {index}?"
            document = f"Reference answer number {index} includes enough words."
            words = document.split()
            query_start, query_end = 0, 0
        query_eeg = np.random.default_rng(index).normal(size=(2, 8, 2)).astype(np.float32)
        pairs.append({
            "query_text": question,
            "query_eeg": query_eeg,
            "query_words": question.split(),
            "doc_text": document,
            "doc_words": words,
            "full_sentence_text": document,
            "full_sentence_words": words,
            "query_start_idx": query_start,
            "query_end_idx": query_end,
            "participant_id": f"{source}_p{index % 5}",
            "sentence_id": index,
            "question_id": f"{source}_q{target_index}",
            "split_group_id": f"{source}_q{target_index}",
            "supports_runtime_masking": supports_masking or task_ict,
            "dataset_source": (("authentic_question_ict" if index % 2 == 0 else "authentic_response_ict")
                               if task_ict else source),
            "training_source": "authentic_query" if task_ict else source,
            "query_kind": ("question" if index % 2 == 0 else "response") if task_ict else "",
        })
    np.save(path, {"ict_pairs": pairs, "metadata": {
        "dataset_name": source,
        "supports_runtime_masking": supports_masking,
    }})
    return path


def make_trial_row(participant, session, question_id):
    return {
        "participant_id": participant,
        "session_id": session,
        "part": "1",
        "trial": "1",
        "question_id": question_id,
        "question_text": "What is capital?",
        "typed_text": "capital",
        "condition": "test",
        "chatbot_condition": "baseline",
        "response_seen": "edited response",
        "ground_truth_answer": "Paris",
        "ground_truth_response": "Paris is the capital of France.",
        "answer_status": "source_verified",
        "answer_provenance": "synthetic fixture",
        "answer_notes": "",
    }


class ExperimentPipelineSmokeTests(unittest.TestCase):
    def test_single_source_loader_deserializes_source_once_for_three_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = write_synthetic_pairs(root / "alice_ict_pairs.npy", "alice", 12, True)
            dimensions = compute_global_eeg_dimensions(audio_path, max_eeg_len=4)
            with patch("controller.load_combined_datasets", wraps=load_combined_datasets) as loader:
                create_dataloaders(
                    data_path=audio_path, tokenizer=TinyTokenizer(), batch_size=4, max_text_len=24,
                    max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                    training_masking_level=90, global_eeg_dims=dimensions,
                    pair_split_seed=42, balance_sources=False,
                )
            self.assertEqual(loader.call_count, 1)

    def test_task_reader_keeps_aligned_and_requested_unaligned_people(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            root.mkdir()
            question_id = "q_smoke"
            trial_rows = [
                make_trial_row("P003", "ses-smoke", question_id),
                make_trial_row("P001", "unaligned_task_export", question_id),
                make_trial_row("P002", "unaligned_task_export", question_id),
            ]
            with (root / "trial_response_ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
                writer.writeheader()
                writer.writerows(trial_rows)

            aligned_dir = root / "P003" / "ses-smoke"
            aligned_dir.mkdir(parents=True)
            info = mne.create_info(["C1", "C2"], sfreq=10.0, ch_types="eeg")
            events = np.array([[20, 0, 1], [40, 0, 1]])
            metadata = pd.DataFrame([
                {"word": "what", "onset": 2.0, "part": 1, "trial": 1, "question_id": question_id,
                 "word_index": 0, "word_index_ambiguous": False},
                {"word": "capital", "onset": 4.0, "part": 1, "trial": 1, "question_id": question_id,
                 "word_index": 2, "word_index_ambiguous": False},
            ])
            epochs = mne.EpochsArray(np.ones((2, 2, 11), dtype=np.float64), info, events,
                                     tmin=-0.2, event_id={"fix": 1}, metadata=metadata, verbose="ERROR")
            epochs.save(aligned_dir / "P003_reading-epo.fif", overwrite=True, verbose="ERROR")

            for participant in ("P001", "P002"):
                unaligned_dir = root / participant / "unaligned_task_export"
                unaligned_dir.mkdir(parents=True)
                raw = mne.io.RawArray(np.ones((2, 200), dtype=np.float64), info, verbose="ERROR")
                raw.save(unaligned_dir / f"{participant}_unaligned_raw.fif", overwrite=True, verbose="ERROR")
                fixation_path = unaligned_dir / f"{participant}_reading_fixations.csv"
                with fixation_path.open("w", newline="", encoding="utf-8") as stream:
                    fields = ["word", "onset", "part", "trial", "question_id", "word_index", "word_index_ambiguous"]
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows([
                        {"word": "what", "onset": 2.0, "part": 1, "trial": 1, "question_id": question_id,
                         "word_index": 0, "word_index_ambiguous": False},
                        {"word": "capital", "onset": 4.0, "part": 1, "trial": 1, "question_id": question_id,
                         "word_index": 2, "word_index_ambiguous": False},
                    ])

            output = root / "task_pairs.npy"
            manifest = build_task_pairs(root, output, unaligned_clock_offset_seconds=0.0)
            pairs = np.load(output, allow_pickle=True).item()["ict_pairs"]
            self.assertEqual(len(pairs), 3)
            self.assertEqual({pair["participant_id"] for pair in pairs}, {"P001", "P002", "P003"})
            self.assertTrue(all(not pair["supports_runtime_masking"] for pair in pairs))
            self.assertTrue(all(pair["query_eeg"].shape == (2, 11, 2) for pair in pairs))
            self.assertEqual({pair["alignment_status"] for pair in pairs},
                             {"aligned_word_index", "unverified_psychopy_to_eeg_offset"})
            self.assertTrue(all(pair["source_identity_group"] == "P001_P002_byte_identical"
                                for pair in pairs if pair["participant_id"] in {"P001", "P002"}))
            self.assertEqual(manifest["pair_count"], 3)
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text())["pair_count"], 3)

    def test_all_seven_mixtures_split_balance_train_and_test_by_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "audio": write_synthetic_pairs(root / "alice_ict_pairs.npy", "alice", 30, True),
                "visual": write_synthetic_pairs(root / "nieuwland_ict_pairs.npy", "nieuwland", 30, True),
                "task": write_synthetic_pairs(root / "authentic_query_task_pairs.npy", "authentic_query", 30, False),
            }
            source_labels = {"audio": ["alice"], "visual": ["nieuwland"],
                             "task": ["authentic_question_ict", "authentic_response_ict"]}
            training_groups = {"audio": "alice", "visual": "nieuwland", "task": "authentic_query"}
            tokenizer = TinyTokenizer()
            test_ids_by_source = {source: [] for sources in source_labels.values() for source in sources}

            for name, source_names in EXPERIMENTS:
                selected_paths = [paths[source] for source in source_names]
                combined = None
                data_path = None
                if len(selected_paths) == 1:
                    data_path = selected_paths[0]
                    dimensions = compute_global_eeg_dimensions(data_path, max_eeg_len=4)
                else:
                    pairs, metadata = load_combined_datasets(selected_paths)
                    combined = {"ict_pairs": pairs, "metadata": metadata}
                    from mv_dataloader import compute_combined_eeg_dimensions
                    dimensions = compute_combined_eeg_dimensions(pairs, max_eeg_len=4)

                train_loader, val_loader, test_loader, _ = create_dataloaders(
                    data_path=data_path, tokenizer=tokenizer, batch_size=6, max_text_len=24,
                    max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                    training_masking_level=90, global_eeg_dims=dimensions,
                    combined_dataset=combined, pair_split_seed=42,
                    balance_sources=len(source_names) > 1 or "task" in source_names,
                )
                self.assertGreater(len(train_loader.dataset), 0, name)
                self.assertGreater(len(val_loader.dataset), 0, name)
                self.assertGreater(len(test_loader.dataset), 0, name)

                source_test_loaders = create_source_test_loaders(test_loader)
                for source, source_loader in source_test_loaders.items():
                    candidate_ids_by_mask = []
                    for masking_level in (0, 100):
                        source_loader.dataset.dataset.set_masking_probability(masking_level / 100.0)
                        documents, _ = build_document_database(source_loader)
                        candidate_ids_by_mask.append([document["document_id"] for document in documents])
                    self.assertEqual(candidate_ids_by_mask[0], candidate_ids_by_mask[1], (name, source))

                for source, source_loader in source_test_loaders.items():
                    test_ids = {
                        (row["participant_id"], row["sentence_id"], row["question_id"])
                        for row in test_loader.dataset.processed_pairs
                        if row["dataset_source"] == source
                    }
                    test_ids_by_source[source].append(test_ids)
                    self.assertEqual({
                        source_loader.dataset.dataset.processed_pairs[index]["dataset_source"]
                        for index in source_loader.dataset.indices
                    }, {source})

                if len(source_names) > 1 or "task" in source_names:
                    self.assertIsInstance(train_loader.sampler, WeightedRandomSampler, name)
                    expected_groups = {training_groups[source] for source in source_names}
                    totals = {source: 0.0 for source in expected_groups}
                    for weight, pair in zip(train_loader.sampler.weights.tolist(),
                                            train_loader.dataset.processed_pairs):
                        source = pair.get("training_source", pair["dataset_source"])
                        totals[source] += weight
                    self.assertTrue(totals, name)
                    for total in totals.values():
                        self.assertAlmostEqual(total, next(iter(totals.values())), places=6, msg=name)

                source_test_loaders = create_source_test_loaders(test_loader)
                for task_source in ("authentic_question_ict", "authentic_response_ict"):
                    task_loader = source_test_loaders.get(task_source)
                    if task_loader is not None:
                        task_loader.dataset.set_masking_probability(1.0)
                        sample = task_loader.dataset[0]
                        self.assertTrue(sample["metadata"]["was_masked"], name)
                        self.assertNotIn("queryword", sample["metadata"]["document_text"], name)

                model = create_model(
                    hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                    pooling_strategy="mean", encoder_type="dual", global_eeg_dims=dimensions,
                    query_type="eeg", use_pretrained_text=False,
                )
                model.multi_positive_loss = True
                model.set_tokenizer_vocab_size(len(tokenizer))
                batch = next(iter(train_loader))
                output = model(batch["eeg_queries"], batch["text_queries"], batch["docs"], batch["eeg_mv_masks"])
                loss, _ = compute_contrastive_loss(output["query_vectors"], output["doc_vectors"], "mean")
                self.assertTrue(torch.isfinite(loss).item(), name)
                loss.backward()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()

                for source, source_loader in source_test_loaders.items():
                    with patch("mv_training.wandb.log"):
                        metrics = test_model(
                            model, source_loader, torch.device("cpu"),
                            test_masking_levels=[0, 100], primary_masking_level=100,
                            wandb_namespace=f"smoke/{source}",
                        )
                    self.assertEqual(metrics["test_ranking/summary/num_test_samples"],
                                     len(source_loader.dataset), name)
                    self.assertEqual(metrics["test_ranking/summary/has_multiple_datasets"], False, name)
                    self.assertEqual(
                        metrics["test_ranking/masking_0/num_unique_documents"],
                        metrics["test_ranking/masking_100/num_unique_documents"],
                        (name, source),
                    )

            for source, split_sets in test_ids_by_source.items():
                self.assertTrue(split_sets, source)
                self.assertTrue(all(test_set == split_sets[0] for test_set in split_sets), source)

    def test_scheduler_prepares_without_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = prepare_jobs(
                root / "alice_ict_pairs.npy", root / "nieuwland_ict_pairs.npy",
                root / "authentic_query_task_pairs.npy", root / "prepared",
                python_executable="python", epochs=1, batch_size=2,
            )
            prepared = root / "prepared"
            self.assertFalse(report["submitted"])
            self.assertEqual(len(report["matrix"]), 7)
            self.assertEqual(report["pooling_strategy"], "mean")
            self.assertEqual(len(list((prepared / "jobs").glob("*.sbatch"))), 7)
            for script in (prepared / "jobs").glob("*.sbatch"):
                self.assertIn("#SBATCH --partition=gpu", script.read_text())
                self.assertIn("controller.py", script.read_text())
                self.assertIn("--pooling_strategy mean", script.read_text())
                self.assertIn("--no_lora", script.read_text())
                self.assertIn("--multi-positive-loss", script.read_text())
            self.assertEqual(report["text_encoder_policy"], "pretrained ColBERT frozen (LoRA disabled)")
            self.assertIn("No jobs were submitted", (prepared / "PREPARED_NOT_SUBMITTED.txt").read_text())

            selected = ["task_only", "audio_task", "visual_task", "audio_visual_task"]
            selected_report = prepare_jobs(
                root / "alice_ict_pairs.npy", root / "nieuwland_ict_pairs.npy",
                root / "authentic_query_task_pairs.npy", root / "selected",
                python_executable="python", epochs=200, patience=5,
                experiment_names=selected, pooling_strategy="mean", exclude_nodes=["gpu04"],
                include_task_target_holdout=True,
            )
            expected = selected + ["task_only_target_holdout"]
            self.assertEqual([item["name"] for item in selected_report["matrix"]], expected)
            self.assertEqual(selected_report["patience"], 5)
            self.assertTrue(all(item["balance_sources"] for item in selected_report["matrix"]))
            self.assertEqual(selected_report["excluded_nodes"], ["gpu04"])
            self.assertEqual(len(list((root / "selected" / "jobs").glob("*.sbatch"))), 5)
            for script in (root / "selected" / "jobs").glob("*.sbatch"):
                command = script.read_text()
                self.assertIn("--pooling_strategy mean", command)
                self.assertIn("--patience 5", command)
                self.assertIn("#SBATCH --exclude=gpu04", command)
                self.assertIn("--balance_sources", command)
                self.assertIn("--multi-positive-loss", command)
                self.assertIn("--split-strategy", command)
                if script.stem == "task_only_target_holdout":
                    self.assertIn("target", command)

    def test_target_grouped_split_keeps_task_question_spans_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_synthetic_pairs(root / "authentic_query_task_pairs.npy", "authentic_query", 120, False)
            task_data = np.load(path, allow_pickle=True).item()
            for index, pair in enumerate(task_data["ict_pairs"]):
                pair.pop("split_group_id", None)
                if index % 2:
                    pair["doc_text"] += " displayed response"
            np.save(path, task_data)
            dimensions = compute_global_eeg_dimensions(path, max_eeg_len=4)
            loaders = create_dataloaders(
                data_path=path, tokenizer=TinyTokenizer(), batch_size=8, max_text_len=24,
                max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                training_masking_level=90, global_eeg_dims=dimensions,
                pair_split_seed=42, balance_sources=True, split_strategy="target",
            )
            groups = [
                {sample["question_id"] for sample in loader.dataset.processed_pairs}
                for loader in loaders[:3]
            ]
            self.assertTrue(all(groups))
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])

    def test_multi_positive_loss_does_not_penalize_duplicate_targets(self):
        query_vectors = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
        document_vectors = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
        diagonal_loss, _ = compute_contrastive_loss(query_vectors, document_vectors, "mean")
        multi_positive_loss, _ = compute_contrastive_loss(
            query_vectors, document_vectors, "mean", positive_ids=["shared", "shared", "other"]
        )
        self.assertLess(multi_positive_loss.item(), diagonal_loss.item())

    def test_mixed_validation_uses_source_pools_without_combined_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = write_synthetic_pairs(root / "alice.npy", "alice", 30, True)
            visual = write_synthetic_pairs(root / "nieuwland.npy", "nieuwland", 30, True)
            pairs, metadata = load_combined_datasets([audio, visual])
            from mv_dataloader import compute_combined_eeg_dimensions
            dimensions = compute_combined_eeg_dimensions(pairs, max_eeg_len=4)
            _, val_loader, _, _ = create_dataloaders(
                data_path=None, tokenizer=TinyTokenizer(), batch_size=8, max_text_len=24,
                max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                training_masking_level=90, global_eeg_dims=dimensions,
                combined_dataset={"ict_pairs": pairs, "metadata": metadata},
                pair_split_seed=42, balance_sources=True,
            )
            expected = {"val_ranking/dataset/source_metric": 1.0}
            with patch("mv_training.build_document_database", side_effect=AssertionError("combined pool built")), \
                    patch("mv_training.perform_dataset_specific_ranking", return_value=expected) as source_eval, \
                    patch("mv_training.wandb.log"):
                result = perform_ranking_validation(object(), val_loader, torch.device("cpu"), epoch_num=3)
            self.assertEqual(result, expected)
            source_eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
