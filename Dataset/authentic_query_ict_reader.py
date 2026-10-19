"""Build separate question-ICT and displayed-response-ICT task datasets."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "authentic_task_ict"


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
          alignment_status: str, offset_seconds: float | None) -> dict:
    start, end, eeg_rows = span
    pair_source = "authentic_question_ict" if kind == "question" else "authentic_response_ict"
    eeg_sequence = np.asarray([np.asarray(eeg, dtype=np.float32).T for eeg in eeg_rows], dtype=np.float32)
    query_text = " ".join(tokens[start:end])
    return {
        "query_text": query_text,
        "query_eeg": eeg_sequence,
        "query_words": tokens[start:end],
        "doc_text": full_text,
        "doc_eeg": np.zeros((0, 0, 0), dtype=np.float32),
        "doc_words": tokens,
        "full_sentence_text": full_text,
        "full_sentence_words": tokens,
        "query_start_idx": start,
        "query_end_idx": end,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "part": int(record["part"]),
        "trial": int(record["trial"]),
        "sentence_id": int(record["trial"]),
        "question_id": record["question_id"],
        "split_group_id": record["question_id"],
        "query_kind": kind,
        "condition": record["condition"],
        "chatbot_condition": record["chatbot_condition"],
        "response_seen": record["response_seen"],
        "ground_truth_answer": record["ground_truth_answer"],
        "answer_status": record["answer_status"],
        "answer_provenance": record["answer_provenance"],
        "dataset_source": pair_source,
        "training_source": "authentic_query",
        "supports_runtime_masking": True,
        "alignment_status": alignment_status,
        "unaligned_clock_offset_seconds": offset_seconds,
        "source_identity_group": "P001_P002_byte_identical" if record["participant_id"] in {"P001", "P002"} else "",
        "source_eeg_file": str(source_path),
    }


def _build_from_aligned_epochs(epochs_path: Path, participant: str, session: str,
                               records: dict, kind: str, random_seed: int,
                               query_ratio: float, min_query_length: int,
                               max_query_length: int, pairs_per_trial: int,
                               min_fixations: int, counters: defaultdict) -> list[dict]:
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
    if epochs.metadata is None:
        counters[f"{kind}_aligned_sessions_missing_metadata"] += 1
        return []
    data = epochs.get_data(copy=False)
    grouped = defaultdict(list)
    for event_index, row in enumerate(epochs.metadata.to_dict("records")):
        key = _key(participant, session, row.get("part", 0), row.get("trial", 0), str(row.get("question_id", "")))
        grouped[key].append((row, data[event_index]))

    output = []
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
    return output


def _build_from_unaligned_raw(participant_dir: Path, records: dict, kind: str,
                              unaligned_clock_offset_seconds: float, random_seed: int,
                              query_ratio: float, min_query_length: int,
                              max_query_length: int, pairs_per_trial: int,
                              min_fixations: int, counters: defaultdict) -> list[dict]:
    participant = participant_dir.name
    task_dir = participant_dir / "unaligned_task_export"
    raw_path = task_dir / f"{participant}_unaligned_raw.fif"
    fixation_path = task_dir / f"{participant}_{'reading' if kind == 'question' else 'response'}_fixations.csv"
    if not raw_path.is_file() or not fixation_path.is_file():
        counters[f"{kind}_unaligned_participant_missing_inputs"] += 1
        return []
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
    return output


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
    for participant_dir in sorted(path for path in dataset_dir.glob("P*") if path.is_dir()):
        participant = participant_dir.name
        for session_dir in sorted(participant_dir.glob("ses-*")):
            question_path = session_dir / f"{participant}_reading-epo.fif"
            response_path = session_dir / f"{participant}_response_fixations-epo.fif"
            if question_path.is_file():
                question_pairs.extend(_build_from_aligned_epochs(
                    question_path, participant, session_dir.name, records, "question", random_seed,
                    query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters))
            if response_path.is_file():
                response_pairs.extend(_build_from_aligned_epochs(
                    response_path, participant, session_dir.name, records, "response", random_seed,
                    query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters))
        question_pairs.extend(_build_from_unaligned_raw(
            participant_dir, records, "question", unaligned_clock_offset_seconds, random_seed,
            query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters))
        response_pairs.extend(_build_from_unaligned_raw(
            participant_dir, records, "response", unaligned_clock_offset_seconds, random_seed,
            query_ratio, min_query_length, max_query_length, pairs_per_trial, min_fixations, counters))

    if not question_pairs or not response_pairs:
        raise ValueError(f"Both ICT types need usable pairs; question={len(question_pairs)}, response={len(response_pairs)}")

    combined_pairs = question_pairs + response_pairs
    base_metadata = {
        "dataset_name": "authentic_query_ict",
        "version": "authentic_query_question_and_response_ict_v1",
        "supports_runtime_masking": True,
        "training_source": "authentic_query",
        "query_pair_definitions": {
            "authentic_question_ict": "question-span EEG to full question text",
            "authentic_response_ict": "displayed-response-span EEG to full response_seen text",
        },
        "response_target_note": "Response ICT uses the edited response actually displayed to the participant, not the gold answer.",
        "sampling": {"random_seed": random_seed, "query_ratio": query_ratio,
                     "min_query_length": min_query_length, "max_query_length": max_query_length,
                     "pairs_per_trial_per_kind": pairs_per_trial, "min_fixations_in_span": min_fixations},
        "unaligned_policy": "P001/P002 use explicit unverified PsychoPy-to-EEG zero-offset assumption and remain labeled unverified.",
        "unaligned_clock_offset_seconds": unaligned_clock_offset_seconds,
        "p001_p002_identity_warning": "Their source files are byte-identical and independent identity is unverified.",
        "counts": {"question_ict": len(question_pairs), "response_ict": len(response_pairs),
                   "combined": len(combined_pairs)},
        "counters": dict(counters),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "authentic_question_ict_pairs.npy": (question_pairs, {**base_metadata, "dataset_name": "authentic_question_ict"}),
        "authentic_response_ict_pairs.npy": (response_pairs, {**base_metadata, "dataset_name": "authentic_response_ict"}),
        "authentic_task_ict_pairs.npy": (combined_pairs, base_metadata),
    }
    for name, (pairs, metadata) in outputs.items():
        np.save(output_dir / name, {"ict_pairs": pairs, "metadata": metadata}, allow_pickle=True)
        (output_dir / name.replace(".npy", ".manifest.json")).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
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
