"""Build separate question-ICT and displayed-response-ICT task datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

try:
    from .authentic_query_reader import _is_true, _key, _matched_word_index, _read_rows
except ImportError:
    from authentic_query_reader import _is_true, _key, _matched_word_index, _read_rows


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "authentic_task_ict_query_response"


def _sample_spans(tokens: list[str], events: list[tuple[int, np.ndarray, float]], *,
                  rng: random.Random, query_ratio: float, min_query_length: int,
                  max_query_length: int, pairs_per_trial: int, min_fixations: int,
                  counters: defaultdict) -> list[tuple[int, int, list[np.ndarray]]]:
    if len(tokens) < 3:
        counters["texts_too_short_for_ict"] += 1
        return []
    query_length = max(min_query_length, int(len(tokens) * query_ratio))
    query_length = min(query_length, max_query_length, len(tokens) - 1)
    if query_length < 1:
        counters["texts_too_short_for_ict"] += 1
        return []

    output = []
    used_spans = set()
    max_start = len(tokens) - query_length
    attempts = 0
    while len(output) < pairs_per_trial and attempts < pairs_per_trial * 20:
        attempts += 1
        start = rng.randint(0, max_start)
        end = start + query_length
        if (start, end) in used_spans:
            continue
        matched = [eeg for word_index, eeg, _ in events if start <= word_index < end]
        if len(matched) < min_fixations:
            counters["sampled_spans_without_enough_fixations"] += 1
            continue
        used_spans.add((start, end))
        output.append((start, end, matched))
    if len(output) < pairs_per_trial:
        counters["trials_with_fewer_than_requested_spans"] += 1
    return output


def _pair(record: dict[str, str], kind: str, full_text: str, tokens: list[str],
          span: tuple[int, int, list[np.ndarray]], source_path: Path,
          alignment_status: str, offset_seconds: float | None,
          target_type: str = "ict") -> dict:
    start, end, eeg_rows = span
    query_text = " ".join(tokens[start:end])
    query_words = tokens[start:end]
    doc_text = full_text
    doc_words = tokens
    pair_source = "authentic_question_ict" if kind == "question" else "authentic_response_ict"
    supports_runtime_masking = True
    question_document_id = _task_document_id(record["question_text"])
    document_id = None
    if kind == "question":
        query_group_parts = (record["participant_id"], record["session_id"], record["part"],
                             record["trial"], record["question_id"], start, end)
        query_group_id = hashlib.sha256("\0".join(map(str, query_group_parts)).encode("utf-8")).hexdigest()
    else:
        query_group_id = ""
    split_group_id = "task_trial:" + "|".join(
        str(record[field]) for field in ("participant_id", "session_id", "part", "trial", "question_id"))

    if kind == "question" and target_type == "response":
        doc_text = record["response_seen"].strip()
        if not doc_text:
            return None
        doc_words = doc_text.split()
        pair_source = "authentic_query_response"
        supports_runtime_masking = False
    document_id = _task_document_id(doc_text)
    response_document_id = _task_document_id(record["response_seen"]) if record["response_seen"].strip() else ""
    # Question-reading EEG is relevant to both the question it read and the response it led to.
    if kind == "question":
        positive_document_ids = [question_document_id] + ([response_document_id] if response_document_id else [])
    else:
        positive_document_ids = [document_id]

    eeg_sequence = np.asarray([np.asarray(eeg, dtype=np.float32).T for eeg in eeg_rows], dtype=np.float32)
    pair = {
        "query_text": query_text,
        "query_eeg": eeg_sequence,
        "query_words": query_words,
        "doc_text": doc_text,
        "doc_eeg": np.zeros((0, 0, 0), dtype=np.float32),
        "doc_words": doc_words,
        "full_sentence_text": doc_text,
        "full_sentence_words": doc_words,
        "query_start_idx": start,
        "query_end_idx": end,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "part": int(record["part"]),
        "trial": int(record["trial"]),
        "sentence_id": int(record["trial"]),
        "question_id": record["question_id"],
        "question_document_id": question_document_id,
        "question_text": record["question_text"],
        "positive_document_ids": positive_document_ids,
        "split_group_id": split_group_id,
        "query_group_id": query_group_id,
        "query_kind": "question_to_response" if pair_source == "authentic_query_response" else kind,
        "condition": record["condition"],
        "chatbot_condition": record["chatbot_condition"],
        "response_seen": record["response_seen"],
        "ground_truth_answer": record["ground_truth_answer"],
        "answer_status": record["answer_status"],
        "answer_provenance": record["answer_provenance"],
        "dataset_source": pair_source,
        "training_source": "authentic_query",
        "supports_runtime_masking": supports_runtime_masking,
        "alignment_status": alignment_status,
        "unaligned_clock_offset_seconds": offset_seconds,
        "source_identity_group": "P001_P002_byte_identical" if record["participant_id"] in {"P001", "P002"} else "",
        "source_eeg_file": str(source_path),
    }
    pair["document_id"] = document_id
    return pair


def _full_query_response_pair(record: dict[str, str], tokens: list[str],
                              events: list[tuple[int, np.ndarray, float]], source_path: Path,
                              alignment_status: str, offset_seconds: float | None,
                              min_fixations: int, counters: defaultdict) -> dict | None:
    if len(events) < min_fixations:
        counters["question_trials_without_full_query_fixations"] += 1
        return None
    eeg_rows = [eeg for _, eeg, _ in sorted(events, key=lambda event: event[2])]
    pair = _pair(record, "question", record["question_text"], tokens, (0, len(tokens), eeg_rows),
                 source_path, alignment_status, offset_seconds, target_type="response")
    if pair is None:
        counters["question_trials_missing_displayed_response"] += 1
    return pair


def _task_document_id(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"authentic_task_text:{digest}"


def _build_from_aligned_epochs(epochs_path: Path, participant: str, session: str,
                               records: dict, kind: str, random_seed: int,
                               query_ratio: float, min_query_length: int,
                               max_query_length: int, pairs_per_trial: int,
                               min_fixations: int, counters: defaultdict) -> tuple[list[dict], list[dict]]:
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
    if epochs.metadata is None:
        counters[f"{kind}_aligned_sessions_missing_metadata"] += 1
        return [], []
    data = epochs.get_data(copy=False)
    grouped = defaultdict(list)
    for event_index, row in enumerate(epochs.metadata.to_dict("records")):
        key = _key(participant, session, row.get("part", 0), row.get("trial", 0), str(row.get("question_id", "")))
        grouped[key].append((row, data[event_index]))

    output = []
    query_response_pairs = []
    for key, rows in grouped.items():
        record = records.get(key)
        if record is None:
            counters[f"{kind}_aligned_trial_missing_record"] += 1
            continue
        full_text = record["question_text"] if kind == "question" else record["response_seen"]
        tokens = full_text.split()
        events = []
        for row, eeg in rows:
            index = _matched_word_index(row, tokens)
            if index is not None:
                onset = float(row.get("fixation_onset", row.get("onset", 0.0)))
                events.append((index, eeg, onset))
        counters[f"{kind}_aligned_trials_seen"] += 1
        counters[f"{kind}_aligned_fixations_matched"] += len(events)
        rng = random.Random(f"{random_seed}:{kind}:{key}")
        spans = _sample_spans(tokens, events, rng=rng, query_ratio=query_ratio,
                              min_query_length=min_query_length, max_query_length=max_query_length,
                              pairs_per_trial=pairs_per_trial, min_fixations=min_fixations,
                              counters=counters)
        for span in spans:
            output.append(_pair(record, kind, full_text, tokens, span, epochs_path, "aligned_word_index", None))
        if kind == "question":
            pair = _full_query_response_pair(record, tokens, events, epochs_path, "aligned_word_index",
                                             None, min_fixations, counters)
            if pair is not None:
                query_response_pairs.append(pair)
    return output, query_response_pairs


def _build_from_unaligned_raw(participant_dir: Path, records: dict, kind: str,
                              unaligned_clock_offset_seconds: float, random_seed: int,
                              query_ratio: float, min_query_length: int,
                              max_query_length: int, pairs_per_trial: int,
                              min_fixations: int, counters: defaultdict) -> tuple[list[dict], list[dict]]:
    participant = participant_dir.name
    task_dir = participant_dir / "unaligned_task_export"
    raw_path = task_dir / f"{participant}_unaligned_raw.fif"
    fixation_path = task_dir / f"{participant}_{'reading' if kind == 'question' else 'response'}_fixations.csv"
    if not raw_path.is_file() or not fixation_path.is_file():
        counters[f"{kind}_unaligned_participant_missing_inputs"] += 1
        return [], []
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    rows_by_trial = defaultdict(list)
    for row in _read_rows(fixation_path):
        key = _key(participant, "unaligned_task_export", row.get("part", 0), row.get("trial", 0),
                   str(row.get("question_id", "")))
        rows_by_trial[key].append(row)

    sfreq = float(raw.info["sfreq"])
    epoch_tmin = -0.2
    epoch_samples = int(round(1.0 * sfreq)) + 1
    output = []
    query_response_pairs = []
    for key, rows in rows_by_trial.items():
        record = records.get(key)
        if record is None:
            counters[f"{kind}_unaligned_trial_missing_record"] += 1
            continue
        full_text = record["question_text"] if kind == "question" else record["response_seen"]
        tokens = full_text.split()
        events = []
        time_field = "onset" if kind == "question" else "fixation_onset"
        for row in sorted(rows, key=lambda item: float(item.get(time_field, 0.0))):
            index = _matched_word_index(row, tokens)
            if index is None:
                continue
            onset = float(row[time_field])
            start = int(round((onset + unaligned_clock_offset_seconds + epoch_tmin) * sfreq))
            stop = start + epoch_samples
            if start < 0 or stop > raw.n_times:
                counters[f"{kind}_unaligned_fixations_outside_eeg"] += 1
                continue
            events.append((index, raw.get_data(start=start, stop=stop), onset))
        counters[f"{kind}_unaligned_trials_seen"] += 1
        counters[f"{kind}_unaligned_fixations_matched"] += len(events)
        rng = random.Random(f"{random_seed}:{kind}:{key}")
        spans = _sample_spans(tokens, events, rng=rng, query_ratio=query_ratio,
                              min_query_length=min_query_length, max_query_length=max_query_length,
                              pairs_per_trial=pairs_per_trial, min_fixations=min_fixations,
                              counters=counters)
        for span in spans:
            output.append(_pair(record, kind, full_text, tokens, span, raw_path,
                                "unverified_psychopy_to_eeg_offset", unaligned_clock_offset_seconds))
        if kind == "question":
            pair = _full_query_response_pair(record, tokens, events, raw_path,
                                             "unverified_psychopy_to_eeg_offset",
                                             unaligned_clock_offset_seconds, min_fixations, counters)
            if pair is not None:
                query_response_pairs.append(pair)
    return output, query_response_pairs


def build_task_ict_pairs(dataset_dir: Path, output_dir: Path,
                         unaligned_clock_offset_seconds: float = 0.0,
                         random_seed: int = 42, query_ratio: float = 0.3,
                         min_query_length: int = 2, max_query_length: int = 50,
                         pairs_per_trial: int = 2, min_fixations: int = 1) -> dict:
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    records = {}
    for row in _read_rows(dataset_dir / "trial_response_ground_truth.csv"):
        key = (row["participant_id"], row["session_id"], row["part"], row["trial"], row["question_id"])
        records[key] = row

    counters = defaultdict(int)
    question_pairs = []
    response_pairs = []
    query_response_pairs = []
    for participant_dir in sorted(path for path in dataset_dir.glob("P*") if path.is_dir()):
        participant = participant_dir.name
        for session_dir in sorted(participant_dir.glob("ses-*")):
            question_path = session_dir / f"{participant}_reading-epo.fif"
            response_path = session_dir / f"{participant}_response_fixations-epo.fif"
            if question_path.is_file():
                question_results, query_response_results = _build_from_aligned_epochs(
                    question_path, participant, session_dir.name, records, "question", random_seed,
                    query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters)
                question_pairs.extend(question_results)
                query_response_pairs.extend(query_response_results)
            if response_path.is_file():
                response_results, _ = _build_from_aligned_epochs(
                    response_path, participant, session_dir.name, records, "response", random_seed,
                    query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters)
                response_pairs.extend(response_results)
        question_results, query_response_results = _build_from_unaligned_raw(
            participant_dir, records, "question", unaligned_clock_offset_seconds, random_seed,
            query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters)
        question_pairs.extend(question_results)
        query_response_pairs.extend(query_response_results)
        response_results, _ = _build_from_unaligned_raw(
            participant_dir, records, "response", unaligned_clock_offset_seconds, random_seed,
            query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters)
        response_pairs.extend(response_results)

    if not question_pairs or not response_pairs or not query_response_pairs:
        raise ValueError("All three Task pair types need usable pairs; "
                         f"question_ict={len(question_pairs)}, response_ict={len(response_pairs)}, "
                         f"query_response={len(query_response_pairs)}")

    combined_pairs = query_response_pairs + question_pairs + response_pairs
    base_metadata = {
        "dataset_name": "authentic_query_ict",
        "version": "authentic_query_full_query_response_and_ict_v4",
        "supports_runtime_masking": True,
        "training_source": "authentic_query",
        "query_pair_definitions": {
            "authentic_question_ict": "question-span EEG to full question text",
            "authentic_response_ict": "displayed-response-span EEG to full response_seen text",
            "authentic_query_response": "full-question-reading EEG to the same trial's full displayed response_seen text",
        },
        "positive_policy": "Question-reading EEG has its full question and response_seen as positives; response-reading EEG has response_seen.",
        "split_group": "participant/session/part/trial/question; all EEG from one trial stays in one split",
        "response_target_note": "Response ICT and question-to-response pairs use the exact response displayed to the participant, not the gold answer.",
        "sampling": {"random_seed": random_seed, "query_ratio": query_ratio,
                     "min_query_length": min_query_length, "max_query_length": max_query_length,
                     "pairs_per_trial_per_kind": pairs_per_trial, "min_fixations_in_span": min_fixations},
        "unaligned_policy": "P001/P002 use explicit unverified PsychoPy-to-EEG zero-offset assumption and remain labeled unverified.",
        "unaligned_clock_offset_seconds": unaligned_clock_offset_seconds,
        "p001_p002_identity_warning": "Their source files are byte-identical and independent identity is unverified.",
        "counts": {"query_response": len(query_response_pairs),
               "question_ict": len(question_pairs), "response_ict": len(response_pairs),
                   "combined": len(combined_pairs)},
        "counters": dict(counters),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_query_response = [pair for pair in query_response_pairs
                              if pair["alignment_status"] == "aligned_word_index"]
    aligned_question_ict = [pair for pair in question_pairs
                            if pair["alignment_status"] == "aligned_word_index"]
    aligned_response_ict = [pair for pair in response_pairs
                            if pair["alignment_status"] == "aligned_word_index"]
    aligned_combined = aligned_query_response + aligned_question_ict + aligned_response_ict
    aligned_metadata = {
        **base_metadata,
        "dataset_name": "authentic_task_ict_aligned",
        "alignment_filter": "Only aligned_word_index EEG; unverified clock-offset pairs excluded.",
        "counts": {
            "query_response": len(aligned_query_response),
            "question_ict": len(aligned_question_ict),
            "response_ict": len(aligned_response_ict),
            "combined": len(aligned_combined),
        },
    }
    outputs = {
        "authentic_query_response_pairs.npy": (query_response_pairs, {**base_metadata, "dataset_name": "authentic_query_response"}),
        "authentic_question_ict_pairs.npy": (question_pairs, {**base_metadata, "dataset_name": "authentic_question_ict"}),
        "authentic_response_ict_pairs.npy": (response_pairs, {**base_metadata, "dataset_name": "authentic_response_ict"}),
        "authentic_task_ict_pairs.npy": (combined_pairs, base_metadata),
        "authentic_query_response_aligned_pairs.npy": (
            aligned_query_response, {**aligned_metadata, "dataset_name": "authentic_query_response_aligned"}),
        "authentic_question_ict_aligned_pairs.npy": (
            aligned_question_ict, {**aligned_metadata, "dataset_name": "authentic_question_ict_aligned"}),
        "authentic_response_ict_aligned_pairs.npy": (
            aligned_response_ict, {**aligned_metadata, "dataset_name": "authentic_response_ict_aligned"}),
        "authentic_task_ict_aligned_pairs.npy": (aligned_combined, aligned_metadata),
    }
    candidate_documents = {}
    for pair in aligned_combined:
        for document_type, text in (("question", pair["question_text"]), ("response", pair["response_seen"])):
            if not text.strip():
                continue
            document_id = _task_document_id(text)
            candidate = candidate_documents.setdefault(document_id, {
                "document_id": document_id,
                "text": text.strip(),
                "document_types": [],
            })
            if document_type not in candidate["document_types"]:
                candidate["document_types"].append(document_type)

    for name, (pairs, metadata) in outputs.items():
        np.save(output_dir / name, {"ict_pairs": pairs, "metadata": metadata}, allow_pickle=True)
        (output_dir / name.replace(".npy", ".manifest.json")).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    eval_index = {
        "index_name": "aligned_full_questions_and_displayed_responses",
        "alignment_filter": "Only aligned_word_index EEG-derived Task text; one row per unique normalized full-text document.",
        "candidate_count": len(candidate_documents),
        "candidate_documents": list(candidate_documents.values()),
    }
    (output_dir / "authentic_task_expanded_eval_index.json").write_text(
        json.dumps(eval_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return base_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR / "authentic_task_ict")
    parser.add_argument("--unaligned-clock-offset-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-ratio", type=float, default=0.3)
    parser.add_argument("--min-query-length", type=int, default=2)
    parser.add_argument("--max-query-length", type=int, default=50)
    parser.add_argument("--pairs-per-trial", type=int, default=2)
    parser.add_argument("--min-fixations", type=int, default=1)
    args = parser.parse_args()
    result = build_task_ict_pairs(args.dataset_dir, args.output_dir, args.unaligned_clock_offset_seconds,
                                  args.seed, args.query_ratio, args.min_query_length,
                                  args.max_query_length, args.pairs_per_trial, args.min_fixations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
