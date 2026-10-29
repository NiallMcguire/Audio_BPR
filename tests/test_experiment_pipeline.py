import csv
import hashlib
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
from Dataset.authentic_query_ict_reader import (
    _pair as make_ict_pair,
    build_task_ict_pairs,
)
from controller import (create_dataloaders, create_source_test_loaders,
                        select_primary_validation_loader)
from mv_dataloader import compute_global_eeg_dimensions, load_combined_datasets
from mv_models import compute_similarity, create_model
from mv_training import (build_document_database, build_expanded_candidate_database,
                         compute_contrastive_loss, compute_multi_positive_ranking_metrics,
                         evaluate_expanded_candidate_index, initialize_wandb,
                         validate_epoch,
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
    task_sources = {"authentic_query_response", "authentic_question_ict", "authentic_response_ict"}
    for index in range(count):
        task_pair = source in task_sources
        target_index = index // 4 if task_pair else index
        if source in {"authentic_query_response", "authentic_response_ict"}:
            question = f"What fact about query{target_index}?"
            document = f"Displayed response for query{target_index} has enough words."
            query_start, query_end = 0, 0
        elif task_pair:
            question = f"What fact about query{target_index}?"
            document = question
            query_start, query_end = 2, 3
        elif supports_masking:
            question = f"queryword{target_index} context{target_index}"
            document = f"before{target_index} queryword{target_index} context{target_index} after{target_index}"
            query_start, query_end = 1, 3
        else:
            question = f"What fact {index}?"
            document = f"Reference answer number {index} includes enough words."
            query_start, query_end = 0, 0
        words = document.split()
        query_eeg = np.random.default_rng(index).normal(size=(2, 8, 2)).astype(np.float32)
        pair = {
            "query_text": question,
            "query_eeg": query_eeg,
            "query_words": question.split(),
            "doc_text": document,
            "doc_words": words,
            "full_sentence_text": document,
            "full_sentence_words": words,
            "query_start_idx": query_start,
            "query_end_idx": query_end,
            "participant_id": f"{'task' if task_pair else source}_p{index % 10}",
            "sentence_id": target_index if task_pair else index,
            "question_id": f"task_q{target_index}" if task_pair else f"{source}_q{target_index}",
            "question_document_id": "authentic_task_text:question{}".format(target_index) if task_pair else "",
            "split_group_id": f"task_trial_{target_index}" if task_pair else f"{source}_target_{target_index}",
            "query_group_id": f"task_q{target_index}_span{index % 4}" if source in {
                "authentic_query_response", "authentic_question_ict"} else "",
            "supports_runtime_masking": (source == "authentic_question_ict") if task_pair else supports_masking,
            "dataset_source": source,
            "training_source": "authentic_query" if task_pair else source,
            "query_kind": source,
        }
        if source in {"authentic_query_response", "authentic_response_ict"}:
            normalized = " ".join(document.split()).casefold()
            pair["document_id"] = "authentic_task_response:" + hashlib.sha256(normalized.encode()).hexdigest()
        elif source == "authentic_question_ict":
            pair["document_id"] = pair["question_document_id"]
        pairs.append(pair)
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
    def test_question_eeg_pair_targets_displayed_response(self):
        record = make_trial_row("P003", "ses-smoke", "q_smoke")
        eeg = np.ones((2, 11), dtype=np.float32)
        span = (1, 2, [eeg])
        question_pair = make_ict_pair(
            record, "question", record["question_text"], record["question_text"].split(),
            span, Path("question-epochs.fif"), "aligned_word_index", None,
        )
        response_ict_pair = make_ict_pair(
            record, "response", record["response_seen"], record["response_seen"].split(),
            span, Path("response-epochs.fif"), "aligned_word_index", None,
        )
        query_response_pair = make_ict_pair(
            record, "question", record["question_text"], record["question_text"].split(),
            span, Path("question-epochs.fif"), "aligned_word_index", None, target_type="response",
        )
        repeated_prompt_record = make_trial_row("P004", "ses-other", "q_repeated")
        repeated_prompt_pair = make_ict_pair(
            repeated_prompt_record, "question", repeated_prompt_record["question_text"],
            repeated_prompt_record["question_text"].split(), span, Path("other-question-epochs.fif"),
            "aligned_word_index", None,
        )

        self.assertEqual(query_response_pair["dataset_source"], "authentic_query_response")
        self.assertEqual(query_response_pair["query_text"], question_pair["query_text"])
        np.testing.assert_array_equal(query_response_pair["query_eeg"], question_pair["query_eeg"])
        self.assertEqual(query_response_pair["doc_text"], "edited response")
        self.assertEqual(query_response_pair["document_id"], response_ict_pair["document_id"])
        self.assertEqual(query_response_pair["query_group_id"], question_pair["query_group_id"])
        self.assertNotEqual(query_response_pair["split_group_id"], repeated_prompt_pair["split_group_id"])
        self.assertEqual(question_pair["split_group_id"], query_response_pair["split_group_id"])
        self.assertEqual(set(question_pair["positive_document_ids"]),
                         {question_pair["document_id"], response_ict_pair["document_id"]})
        self.assertEqual(response_ict_pair["positive_document_ids"], [response_ict_pair["document_id"]])
        self.assertNotEqual(query_response_pair["query_group_id"], repeated_prompt_pair["query_group_id"])
        self.assertFalse(query_response_pair["supports_runtime_masking"])

    def test_task_ict_builder_writes_three_separate_pair_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            participant_dir = root / "P003" / "ses-smoke"
            participant_dir.mkdir(parents=True)
            trial = make_trial_row("P003", "ses-smoke", "q_smoke")
            trial["response_seen"] = "edited assistant response"
            unaligned_trial = make_trial_row("P001", "unaligned_task_export", "q_smoke")
            unaligned_trial["response_seen"] = trial["response_seen"]
            with (root / "trial_response_ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trial))
                writer.writeheader()
                writer.writerows([trial, unaligned_trial])

            info = mne.create_info(["C1", "C2"], sfreq=10.0, ch_types="eeg")
            question_metadata = pd.DataFrame([
                {"word": word, "onset": 2.0 + index, "part": 1, "trial": 1,
                 "question_id": "q_smoke", "word_index": index, "word_index_ambiguous": False}
                for index, word in enumerate(trial["question_text"].split())
            ])
            response_metadata = pd.DataFrame([
                {"word": word, "fixation_onset": 5.0 + index, "part": 1, "trial": 1,
                 "question_id": "q_smoke", "word_index": index, "word_index_ambiguous": False}
                for index, word in enumerate(trial["response_seen"].split())
            ])
            for filename, metadata in (("P003_reading-epo.fif", question_metadata),
                                       ("P003_response_fixations-epo.fif", response_metadata)):
                events = np.array([[20 + 10 * index, 0, 1] for index in range(len(metadata))])
                epochs = mne.EpochsArray(
                    np.ones((len(metadata), 2, 11), dtype=np.float64), info, events,
                    tmin=-0.2, event_id={"fix": 1}, metadata=metadata, verbose="ERROR",
                )
                epochs.save(participant_dir / filename, overwrite=True, verbose="ERROR")

            unaligned_dir = root / "P001" / "unaligned_task_export"
            unaligned_dir.mkdir(parents=True)
            raw = mne.io.RawArray(np.ones((2, 200), dtype=np.float64), info, verbose="ERROR")
            raw.save(unaligned_dir / "P001_unaligned_raw.fif", overwrite=True, verbose="ERROR")
            fixation_fields = ["word", "onset", "fixation_onset", "part", "trial", "question_id",
                               "word_index", "word_index_ambiguous"]
            for filename, time_field, text in (
                ("P001_reading_fixations.csv", "onset", unaligned_trial["question_text"]),
                ("P001_response_fixations.csv", "fixation_onset", unaligned_trial["response_seen"]),
            ):
                with (unaligned_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fixation_fields)
                    writer.writeheader()
                    for index, word in enumerate(text.split()):
                        writer.writerow({
                            "word": word, time_field: 2.0 + index, "part": 1, "trial": 1,
                            "question_id": "q_smoke", "word_index": index,
                            "word_index_ambiguous": False,
                        })

            output_dir = root / "task_ict_out"
            manifest = build_task_ict_pairs(
                root, output_dir, random_seed=42, query_ratio=0.3,
                min_query_length=1, max_query_length=2, pairs_per_trial=1,
            )
            query_response = np.load(output_dir / "authentic_query_response_pairs.npy",
                                     allow_pickle=True).item()["ict_pairs"]
            question_ict = np.load(output_dir / "authentic_question_ict_pairs.npy",
                                   allow_pickle=True).item()["ict_pairs"]
            response_ict = np.load(output_dir / "authentic_response_ict_pairs.npy",
                                   allow_pickle=True).item()["ict_pairs"]
            aligned_query_response = np.load(
                output_dir / "authentic_query_response_aligned_pairs.npy", allow_pickle=True
            ).item()["ict_pairs"]
            expanded_index = json.loads((output_dir / "authentic_task_expanded_eval_index.json").read_text())
            self.assertEqual(manifest["counts"]["query_response"], 2)
            self.assertEqual(manifest["counts"]["question_ict"], 2)
            self.assertEqual(manifest["counts"]["response_ict"], 2)
            self.assertEqual(query_response[0]["doc_text"], trial["response_seen"])
            self.assertEqual(query_response[0]["query_text"], trial["question_text"])
            self.assertEqual(query_response[0]["query_eeg"].shape[0], len(trial["question_text"].split()))
            self.assertEqual(query_response[0]["split_group_id"], question_ict[0]["split_group_id"])
            self.assertEqual(query_response[0]["document_id"], response_ict[0]["document_id"])
            self.assertEqual(len(aligned_query_response), 1)
            self.assertTrue(all(pair["alignment_status"] == "aligned_word_index"
                                for pair in aligned_query_response))
            self.assertTrue(any(pair["alignment_status"] == "unverified_psychopy_to_eeg_offset"
                                for pair in query_response))
            self.assertEqual(expanded_index["candidate_count"], 2)
            self.assertEqual({doc_type for doc in expanded_index["candidate_documents"]
                              for doc_type in doc["document_types"]}, {"question", "response"})

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

    def test_seven_ablation_arms_share_query_response_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "audio": write_synthetic_pairs(root / "alice_ict_pairs.npy", "alice", 120, True),
                "visual": write_synthetic_pairs(root / "nieuwland_ict_pairs.npy", "nieuwland", 120, True),
                "query_response": write_synthetic_pairs(
                    root / "authentic_query_response_pairs.npy", "authentic_query_response", 120, False),
                "question_ict": write_synthetic_pairs(
                    root / "authentic_question_ict_pairs.npy", "authentic_question_ict", 120, True),
                "response_ict": write_synthetic_pairs(
                    root / "authentic_response_ict_pairs.npy", "authentic_response_ict", 160, False),
            }
            tokenizer = TinyTokenizer()
            test_ids_by_source = {source: [] for source in (
                "alice", "nieuwland", "authentic_query_response",
                "authentic_question_ict", "authentic_response_ict",
            )}

            for name, source_names, *_ in EXPERIMENTS:
                if name.startswith(("typing_", "imagining_")):
                    continue
                selected_paths = [paths[source] for source in source_names]
                pairs, metadata = load_combined_datasets(selected_paths)
                combined = {"ict_pairs": pairs, "metadata": metadata}
                from mv_dataloader import compute_combined_eeg_dimensions
                dimensions = compute_combined_eeg_dimensions(pairs, max_eeg_len=4)

                train_loader, val_loader, test_loader, _ = create_dataloaders(
                    data_path=None, tokenizer=tokenizer, batch_size=6, max_text_len=24,
                    max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                    training_masking_level=90, global_eeg_dims=dimensions,
                    combined_dataset=combined, pair_split_seed=42,
                    balance_sources=len(source_names) > 1, split_strategy="original",
                )
                split_indices = [set(loader.dataset.indices)
                                 for loader in (train_loader, val_loader, test_loader)]
                self.assertFalse(split_indices[0] & split_indices[1], name)
                self.assertEqual(len(split_indices[0]) + len(split_indices[1]), len(pairs), name)
                self.assertTrue(split_indices[2] <= split_indices[1], name)
                self.assertGreater(len(train_loader.dataset), 0, name)
                self.assertGreater(len(val_loader.dataset), 0, name)
                self.assertGreater(len(test_loader.dataset), 0, name)

                primary_val_loader = select_primary_validation_loader(
                    val_loader, "authentic_query_response")
                self.assertTrue(primary_val_loader.dataset.indices, name)
                self.assertEqual({
                    primary_val_loader.dataset.dataset.processed_pairs[index]["dataset_source"]
                    for index in primary_val_loader.dataset.indices
                }, {"authentic_query_response"}, name)

                source_test_loaders = create_source_test_loaders(test_loader)
                for source, source_loader in source_test_loaders.items():
                    pool_sizes = []
                    for masking_level in (0, 100):
                        source_loader.dataset.dataset.set_masking_probability(masking_level / 100.0)
                        documents, _ = build_document_database(source_loader)
                        pool_sizes.append(len(documents))
                    # Original Audio BPR: candidates are deduplicated by (masked) text.
                    self.assertLessEqual(pool_sizes[0], pool_sizes[1], (name, source))

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

                if len(source_names) > 1:
                    self.assertIsInstance(train_loader.sampler, WeightedRandomSampler, name)
                    expected_groups = {
                        "authentic_query" if source in {"query_response", "question_ict", "response_ict"}
                        else {"audio": "alice", "visual": "nieuwland"}.get(source, source)
                        for source in source_names
                    }
                    totals = {source: 0.0 for source in expected_groups}
                    for weight, pair in zip(train_loader.sampler.weights.tolist(),
                                            train_loader.dataset.processed_pairs):
                        source = pair.get("training_source", pair["dataset_source"])
                        totals[source] += weight
                    self.assertTrue(totals, name)
                    for total in totals.values():
                        self.assertAlmostEqual(total, next(iter(totals.values())), places=6, msg=name)

                source_test_loaders = create_source_test_loaders(test_loader)
                for task_source in ("authentic_question_ict", "authentic_response_ict",
                                    "authentic_query_response"):
                    task_loader = source_test_loaders.get(task_source)
                    if task_loader is not None:
                        task_loader.dataset.set_masking_probability(1.0)
                        sample = task_loader.dataset[0]
                        if task_source == "authentic_question_ict":
                            self.assertTrue(sample["metadata"]["was_masked"], name)
                            self.assertNotIn("queryword", sample["metadata"]["document_text"], name)
                        else:
                            self.assertFalse(sample["metadata"]["was_masked"], name)
                            self.assertIn("displayed response", sample["metadata"]["document_text"].casefold(), name)

                model = create_model(
                    hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                    pooling_strategy="mean", encoder_type="dual", global_eeg_dims=dimensions,
                    query_type="eeg", use_pretrained_text=False,
                )
                model.multi_positive_loss = True
                model.set_tokenizer_vocab_size(len(tokenizer))
                batch = next(iter(train_loader))
                output = model(batch["eeg_queries"], batch["text_queries"], batch["docs"], batch["eeg_mv_masks"])
                loss, _ = compute_contrastive_loss(
                    output["query_vectors"], output["doc_vectors"], "mean",
                    positive_ids=[item["document_id"] for item in batch["metadata"]],
                    query_group_ids=[item.get("query_group_id", "") for item in batch["metadata"]],
                )
                self.assertTrue(torch.isfinite(loss).item(), name)
                loss.backward()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()

                if name == "query_response_only":
                    expanded_documents = {}
                    for task_source in ("question_ict", "response_ict"):
                        task_pairs = np.load(paths[task_source], allow_pickle=True).item()["ict_pairs"]
                        for task_pair in task_pairs:
                            expanded_documents[task_pair["document_id"]] = {
                                "document_id": task_pair["document_id"],
                                "text": task_pair["doc_text"],
                                "document_types": [task_source],
                            }
                    eval_index = {"candidate_documents": list(expanded_documents.values())}
                    expanded_metrics = evaluate_expanded_candidate_index(
                        model, primary_val_loader, torch.device("cpu"), eval_index,
                        prefix="test_ranking/expanded_index",
                    )
                    expanded_noise_metrics = evaluate_expanded_candidate_index(
                        model, primary_val_loader, torch.device("cpu"), eval_index,
                        prefix="test_ranking/expanded_index_noise", use_noise=True,
                    )
                    self.assertEqual(expanded_metrics["test_ranking/expanded_index/num_unique_documents"],
                                     len(expanded_documents))
                    self.assertGreater(expanded_metrics["test_ranking/expanded_index/num_queries_evaluated"], 0)
                    self.assertIn("test_ranking/expanded_index_noise/rr", expanded_noise_metrics)

                for source, source_loader in source_test_loaders.items():
                    with patch("mv_training.wandb.log"):
                        metrics = test_model(
                            model, source_loader, torch.device("cpu"),
                            test_masking_levels=[0, 100], primary_masking_level=100,
                            wandb_namespace=f"smoke/{source}",
                            expanded_eval_index=(eval_index if name == "query_response_only"
                                                 and source == "authentic_query_response" else None),
                        )
                    self.assertEqual(metrics["test_ranking/summary/num_test_samples"],
                                     len(source_loader.dataset), name)
                    self.assertEqual(metrics["test_ranking/summary/has_multiple_datasets"], False, name)
                    self.assertLessEqual(
                        metrics["test_ranking/masking_0/num_unique_documents"],
                        metrics["test_ranking/masking_100/num_unique_documents"],
                        (name, source),
                    )
                    if name == "query_response_only" and source == "authentic_query_response":
                        self.assertEqual(
                            metrics["test_ranking/expanded_index/num_unique_documents"],
                            len(expanded_documents),
                        )
                        self.assertIn("test_ranking/expanded_index_noise/rr", metrics)

    def test_scheduler_prepares_without_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "task_data"
            task_dir.mkdir()
            task_data = task_dir / "authentic_task_ict_pairs.npy"
            task_data.touch()
            for filename in (
                "authentic_query_response_pairs.npy", "authentic_question_ict_pairs.npy",
                "authentic_response_ict_pairs.npy", "authentic_query_response_aligned_pairs.npy",
                "authentic_question_ict_aligned_pairs.npy", "authentic_response_ict_aligned_pairs.npy",
                "authentic_task_ict_aligned_pairs.npy", "authentic_task_expanded_eval_index.json",
            ):
                (task_dir / filename).touch()
            typing_dir = root / "authentic_typing_pairs"
            typing_dir.mkdir()
            for fmt in ("continuous", "keypress", "word"):
                for kind in ("query_response", "ict", "question_ict"):
                    (typing_dir / f"authentic_typing_{fmt}_{kind}_pairs.npy").touch()
            imagining_dir = root / "authentic_imagining_pairs"
            imagining_dir.mkdir()
            for fmt in ("window", "pace", "uniform", "shuffled"):
                (imagining_dir / f"authentic_imagining_{fmt}_response_pairs.npy").touch()
            for variant in ("position", "pace", "random"):
                (imagining_dir / f"authentic_imagining_ict_{variant}_question_pairs.npy").touch()
            (task_dir / "authentic_task_expanded_eval_index.json").write_text(json.dumps({
                "candidate_count": 2,
                "candidate_documents": [],
            }))
            audio_data = root / "alice_ict_pairs.npy"
            visual_data = root / "nieuwland_ict_pairs.npy"
            audio_data.touch()
            visual_data.touch()
            report = prepare_jobs(
                audio_data, visual_data, task_data, root / "prepared",
                python_executable="python", epochs=1, batch_size=2, validate_inputs=True,
            )
            prepared = root / "prepared"
            self.assertFalse(report["submitted"])
            self.assertEqual(len(report["matrix"]), len(EXPERIMENTS))
            self.assertEqual([item["name"] for item in report["matrix"]],
                             [name for name, *_ in EXPERIMENTS])
            self.assertEqual(report["pooling_strategy"], "mean")
            self.assertEqual(report["wandb_project"], "Authentic Queries BPR")
            self.assertIn("aligned_word_index", report["primary_alignment_policy"])
            self.assertEqual(report["expanded_eval_candidate_count"], 2)
            self.assertEqual(len(list((prepared / "jobs").glob("*.sbatch"))), len(EXPERIMENTS))
            for script in (prepared / "jobs").glob("*.sbatch"):
                self.assertIn("#SBATCH --partition=gpu", script.read_text())
                self.assertIn("controller.py", script.read_text())
                self.assertIn("--pooling_strategy mean", script.read_text())
                self.assertIn("--no_lora", script.read_text())
                self.assertIn("--multi-positive-loss", script.read_text())
                self.assertIn("--wandb-project 'Authentic Queries BPR'", script.read_text())
                trial_arm = script.stem.startswith("imagining_v3_") or script.stem == "reading_arm_a_trialsplit"
                self.assertIn(f"--split-strategy {'trial' if trial_arm else 'original'}", script.read_text())
                self.assertEqual("--split-seed 42" in script.read_text(), trial_arm)
                self.assertNotIn("--holdout_subjects", script.read_text())
                self.assertNotIn("--train-sources", script.read_text())
                text = script.read_text()
                imagining_flags = ("--eeg-train-crop-units", "--eeg-test-crop-units", "--eeg-window-units",
                                   "--init-checkpoint")
                if script.stem.startswith("imagining_"):
                    for flag in ("--lr 3e-05", "--patience 20", "--min-epochs", "--grad-clip 5.0",
                                 "--primary-eval-source authentic_imagining"):
                        self.assertIn(flag, text)
                    self.assertNotIn("--expanded-eval-index", text)
                    if script.stem.startswith("imagining_v3_"):
                        self.assertIn("authentic_imagining_ict_", text)
                        self.assertEqual("--eeg-window-units 2" in text, script.stem == "imagining_v3_c_mil")
                        self.assertEqual("--question-as-positive" in text,
                                         script.stem == "imagining_v3_c_combined_targets")
                        self.assertEqual("--eeg-crop-sources authentic_imagining_window" in text,
                                         script.stem in ("imagining_v3_c_crop", "imagining_v3_c_cropavg"))
                        if script.stem == "imagining_v3_arm_a_nieuwland_transfer":
                            lines = [line for line in text.splitlines() if "controller.py" in line]
                            self.assertEqual(len(lines), 2)
                            self.assertIn("nieuwland_ict_pairs.npy", lines[0])
                            self.assertIn("--eeg-unit-shape 513 64", lines[0])
                            self.assertIn("pretrain/model_mean_dual.pt", lines[1])
                        if script.stem == "imagining_v3_arm_a_long":
                            self.assertIn("--epochs 150", text)
                            self.assertIn("--min-epochs 150", text)
                        continue
                    self.assertIn("--min-epochs 10", text)
                    if script.stem.startswith("imagining_arm_"):
                        self.assertIn("authentic_imagining_window_response_pairs.npy", text)
                        variant = script.stem.rsplit("_", 1)[1] if "_a_ict_" in script.stem else "position"
                        self.assertIn(f"authentic_imagining_ict_{variant}_question_pairs.npy", text)
                        self.assertEqual("--balance_sources" in text, "_a_ict_" not in script.stem)
                        for flag in imagining_flags:
                            self.assertNotIn(flag, text)
                        continue
                    method, target = script.stem.split("_")[1:3]
                    self.assertEqual(target, "response")
                    fmt = method if method in ("pace", "uniform", "shuffled") else "window"
                    self.assertIn(f"authentic_imagining_{fmt}_{target}_pairs.npy", text)
                    self.assertEqual("--eeg-train-crop-units 2 4" in text, method in ("crop", "cropavg"))
                    self.assertEqual("--eeg-test-crop-units 3" in text, method == "cropavg")
                    self.assertEqual("--eeg-window-units 2" in text, method == "mil")
                    self.assertEqual("--init-checkpoint" in text, method == "transfer")
                    continue
                for flag in imagining_flags:
                    self.assertNotIn(flag, text)
                if script.stem.startswith("typing_v2_"):
                    fmt = script.stem.split("_")[2]
                    for flag in ("--lr 3e-05", "--patience 20", "--min-epochs 10", "--grad-clip 5.0",
                                 "--max_eeg_len 100", "--primary-eval-source authentic_typing"):
                        self.assertIn(flag, text)
                    self.assertIn(f"authentic_typing_{fmt}_question_ict_pairs.npy", text)
                    self.assertEqual("--balance_sources" in text,
                                     "audio" in script.stem or "visual" in script.stem)
                    if "_seed" in script.stem:
                        self.assertIn(f"--seed {script.stem.rsplit('seed', 1)[1]}", text)
                else:
                    for flag in ("--primary-eval-source", "--balance_sources", "--grad-clip",
                                 "--min-epochs", "--max_eeg_len"):
                        self.assertNotIn(flag, text)
                    self.assertIn("--lr 1e-05", text)
                if script.stem.startswith("typing_"):
                    fmt = script.stem.split("_")[2 if script.stem.startswith("typing_v2_") else 1]
                    self.assertNotIn("--expanded-eval-index", text)
                    self.assertIn(f"authentic_typing_{fmt}_query_response_pairs.npy", text)
                    self.assertNotIn("authentic_query_response_aligned_pairs.npy", text)
                    if not script.stem.startswith("typing_v2_"):
                        self.assertIn(f"authentic_typing_{fmt}_ict_pairs.npy", text)
                else:
                    self.assertIn("--expanded-eval-index", text)
                    self.assertIn("authentic_query_response_aligned_pairs.npy", text)
                self.assertRegex(text, r"_(original|trial)split")
                self.assertIn("--wandb-run-name aqbpr_", script.read_text())
            self.assertEqual(report["text_encoder_policy"], "pretrained ColBERT frozen (LoRA disabled)")
            self.assertIn("No jobs were submitted", (prepared / "PREPARED_NOT_SUBMITTED.txt").read_text())

            selected = ["query_response_only", "all_task_sources_audio_visual"]
            selected_report = prepare_jobs(
                audio_data, visual_data, task_data, root / "selected",
                python_executable="python", epochs=200, patience=5,
                experiment_names=selected, pooling_strategy="mean", exclude_nodes=["gpu04"],
                validate_inputs=True,
            )
            self.assertEqual([item["name"] for item in selected_report["matrix"]], selected)
            self.assertEqual(selected_report["patience"], 5)
            self.assertEqual(selected_report["matrix"][0]["sources"], ["query_response"])
            self.assertEqual(selected_report["matrix"][0]["split_strategy"], "original")
            self.assertEqual(len(selected_report["matrix"][0]["data_paths"]), 1)
            self.assertIn("_aligned_pairs.npy", selected_report["matrix"][0]["data_paths"][0])
            self.assertIn("audio", selected_report["matrix"][1]["sources"])
            self.assertIn("visual", selected_report["matrix"][1]["sources"])
            self.assertEqual(selected_report["excluded_nodes"], ["gpu04"])
            self.assertEqual(len(list((root / "selected" / "jobs").glob("*.sbatch"))), 2)
            for script in (root / "selected" / "jobs").glob("*.sbatch"):
                command = script.read_text()
                self.assertIn("--pooling_strategy mean", command)
                self.assertIn("--patience 5", command)
                self.assertIn("#SBATCH --exclude=gpu04", command)
                self.assertIn("--multi-positive-loss", command)
                self.assertIn("--split-strategy original", command)
                self.assertIn("--wandb-project 'Authentic Queries BPR'", command)

    def test_wandb_uses_authentic_queries_project_and_ablation_name(self):
        config = {
            "encoder_type": "dual", "query_type": "eeg", "pooling_strategy": "mean",
            "eeg_arch": "simple", "dataset_name": "combined", "use_lora": False,
            "experiment_name": "all_task_sources", "wandb_project": "Authentic Queries BPR",
            "wandb_run_name": "aqbpr_all_task_sources_mean_seed42_expandedindex",
            "enable_multi_masking_validation": True, "multi_positive_loss": True,
            "query_group_multi_positive": True, "primary_task_source": "authentic_query_response",
            "primary_eval_source": "authentic_query_response",
            "expanded_eval_candidate_count": 48,
            "expanded_eval_policy": "Questions and displayed responses are both relevant.",
        }
        with patch("mv_training.wandb.init") as wandb_init:
            initialize_wandb(config)
        logged = wandb_init.call_args.kwargs
        self.assertEqual(logged["project"], "Authentic Queries BPR")
        self.assertEqual(logged["name"],
                 "aqbpr_all_task_sources_mean_seed42_expandedindex_dynamic_mask")
        self.assertIn("condition-all_task_sources", logged["tags"])
        self.assertEqual(logged["config"]["primary_task_source"], "authentic_query_response")
        self.assertEqual(logged["config"]["primary_eval_source"], "authentic_query_response")
        self.assertTrue(logged["config"]["query_group_multi_positive"])
        self.assertEqual(logged["config"]["expanded_eval_candidate_count"], 48)
        self.assertIn("expanded-multi-positive-index", logged["tags"])

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

    def test_imagining_window_and_crop_pooling(self):
        model = create_model(hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                             pooling_strategy="mean", encoder_type="dual", global_eeg_dims=(10, 8, 2),
                             query_type="eeg", use_pretrained_text=False)
        model.eval()
        eeg = torch.randn(3, 10, 8, 2)
        eeg[2, 6:] = 0
        mask = torch.ones(3, 32)
        with torch.no_grad():
            default = model.encode_eeg(eeg, mask)
            self.assertEqual(tuple(default.shape), (3, 1, 8))
            model.eeg_window_units, model.eeg_window_stride = 2, 1
            windows = model.encode_eeg(eeg, mask)
            self.assertEqual(tuple(windows.shape), (3, 9, 8))
            docs = torch.randn(3, 1, 8)
            similarity = compute_similarity([windows[0:1]], [docs[0:1]], "mean")[0]
            expected = (torch.nn.functional.normalize(windows[0], dim=-1)
                        @ torch.nn.functional.normalize(docs[0, 0], dim=0)).max()
            self.assertAlmostEqual(similarity.item(), expected.item(), places=5)
            loss, _ = compute_contrastive_loss(windows, docs, "mean")
            self.assertTrue(torch.isfinite(loss).item())
            model.eeg_window_units, model.eeg_test_crop_units, model.eeg_test_crops = 0, 3, 5
            cropped = model.encode_eeg(eeg, mask)
            self.assertEqual(tuple(cropped.shape), (3, 1, 8))
            self.assertFalse(torch.allclose(cropped, default))
            model.eeg_test_crop_units = 0
            self.assertTrue(torch.allclose(model.encode_eeg(eeg, mask), default))

    def test_multi_positive_loss_does_not_penalize_duplicate_targets(self):
        query_vectors = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
        document_vectors = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
        diagonal_loss, _ = compute_contrastive_loss(query_vectors, document_vectors, "mean")
        multi_positive_loss, _ = compute_contrastive_loss(
            query_vectors, document_vectors, "mean", positive_ids=["shared", "shared", "other"]
        )
        self.assertLess(multi_positive_loss.item(), diagonal_loss.item())

    def test_multi_positive_ranking_rewards_either_full_text_target(self):
        metrics = compute_multi_positive_ranking_metrics([3, 2, 1, 0], {2, 0})
        self.assertEqual(metrics["rr"], 1.0 / 2.0)
        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)

    def test_multi_positive_loss_accepts_distinct_targets_for_same_query_eeg(self):
        query_vectors = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
        document_vectors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
        document_only_loss, _ = compute_contrastive_loss(
            query_vectors, document_vectors, "mean",
            positive_ids=["question", "response", "response"],
        )
        grouped_loss, _ = compute_contrastive_loss(
            query_vectors, document_vectors, "mean",
            positive_ids=["question", "response", "response"],
            query_group_ids=["same_question_span", "same_question_span", "other_query"],
        )
        self.assertLess(grouped_loss.item(), document_only_loss.item())

    def test_positive_sets_make_question_and_response_both_positive(self):
        query_vectors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        document_vectors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        ids = ["question", "response"]
        plain_loss, _ = compute_contrastive_loss(query_vectors, document_vectors, "mean", positive_ids=ids)
        set_loss, _ = compute_contrastive_loss(
            query_vectors, document_vectors, "mean", positive_ids=ids,
            positive_sets=[{"question", "response"}, {"response"}],
        )
        self.assertLess(set_loss.item(), plain_loss.item())

    def test_trial_split_keeps_trials_together_and_train_filter_keeps_shared_eval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [write_synthetic_pairs(root / f"{source}.npy", source, 120, source == "authentic_question_ict")
                     for source in ("authentic_query_response", "authentic_question_ict", "authentic_response_ict")]
            pairs, metadata = load_combined_datasets(paths)
            from mv_dataloader import compute_combined_eeg_dimensions
            dimensions = compute_combined_eeg_dimensions(pairs, max_eeg_len=4)
            common = dict(data_path=None, tokenizer=TinyTokenizer(), batch_size=8, max_text_len=24,
                          max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                          training_masking_level=90, global_eeg_dims=dimensions,
                          combined_dataset={"ict_pairs": pairs, "metadata": metadata},
                          pair_split_seed=42, balance_sources=True, split_strategy="trial")
            full = create_dataloaders(**common)
            filtered = create_dataloaders(**common, train_sources=["authentic_query_response"])
            groups = [{row["split_group_id"] for row in loader.dataset.processed_pairs} for loader in full[:3]]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual({row["dataset_source"] for row in filtered[0].dataset.processed_pairs},
                             {"authentic_query_response"})
            for full_loader, filtered_loader in zip(full[1:3], filtered[1:3]):
                self.assertEqual([row["original_idx"] for row in full_loader.dataset.processed_pairs],
                                 [row["original_idx"] for row in filtered_loader.dataset.processed_pairs])

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

    def test_validation_logs_expanded_index_ranking(self):
        validation_metrics = {"loss": 0.5, "eeg_query_similarity": 0.1}
        expanded_metrics = {"val_ranking/expanded_index/rr": 0.4}
        with patch("mv_training.validate_single_masking_level",
                   return_value=(validation_metrics, 0.1)), \
                patch("mv_training.detect_dataset_sources", return_value={"has_multiple": False}), \
                patch("mv_training.perform_ranking_validation", return_value={}), \
                patch("mv_training.evaluate_expanded_candidate_index",
                      return_value=expanded_metrics) as expanded_eval, \
                patch("mv_training.wandb.log") as wandb_log:
            loss, metric = validate_epoch(
                object(), object(), torch.device("cpu"), epoch_num=3,
                expanded_eval_index={"candidate_documents": [{"document_id": "doc"}]},
            )
        self.assertEqual((loss, metric), (0.5, 0.1))
        expanded_eval.assert_called_once()
        self.assertIn(expanded_metrics, [call.args[0] for call in wandb_log.call_args_list])


if __name__ == "__main__":
    unittest.main()
