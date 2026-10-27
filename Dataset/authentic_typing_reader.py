"""Build typing-EEG pairs: full typing EEG -> response_seen, and typing-span ICT pairs.

Three EEG formats are written for the same trials and spans:
- continuous: consecutive 513-sample (1 s) windows of the raw typing EEG.
- keypress: one -200..800 ms epoch per surviving keypress of the submitted query.
- word: one -200..800 ms epoch per typed word, anchored at its first surviving key.

Two ICT targets: the submitted typed query (`ict`) and the displayed question with the
matching span masked (`question_ict`, only for spans found verbatim in the question).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

try:
    from .authentic_query_reader import _key, _read_rows
    from .authentic_query_ict_reader import _sample_spans, _task_document_id
except ImportError:
    from authentic_query_reader import _key, _read_rows
    from authentic_query_ict_reader import _sample_spans, _task_document_id

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from typing_annotations import read_windows, replay, text_matches  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "authentic_typing_pairs"
FORMATS = ("continuous", "keypress", "word")
KINDS = ("query_response", "ict", "question_ict")
WINDOW_SAMPLES = 513
EPOCH_PRE_SAMPLES = 102  # matches reading epochs: tmin = -102/512 s
CLOCK_TOLERANCE_S = 0.05


def _textbox_starts(psychopy_csv: Path) -> dict[int, list[float]]:
    starts = {1: [], 2: [], 3: []}
    with psychopy_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            for part in starts:
                value = row.get("textbox.started" if part == 1 else f"textbox_{part}.started")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    starts[part].append(number)
    return starts


def _continuous_windows(data: np.ndarray, start: int, stop: int) -> list[np.ndarray]:
    start, stop = max(0, start), min(data.shape[1], stop)
    windows = []
    for left in range(start, stop, WINDOW_SAMPLES):
        chunk = data[:, left:min(left + WINDOW_SAMPLES, stop)]
        if chunk.shape[1] < WINDOW_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, WINDOW_SAMPLES - chunk.shape[1])))
        windows.append(chunk)
    return windows


def _normalize_token(token: str) -> str:
    return re.sub(r"[^0-9a-z]", "", token.casefold())


def _question_span(question: str, typed_tokens: list[str], start: int, end: int) -> tuple[int, int] | None:
    """Locate the typed span verbatim (case/punctuation-insensitive) in the displayed question."""
    wanted = [_normalize_token(token) for token in typed_tokens[start:end]]
    if not all(wanted):
        return None
    question_tokens = [_normalize_token(token) for token in question.split()]
    for left in range(len(question_tokens) - len(wanted) + 1):
        if question_tokens[left:left + len(wanted)] == wanted:
            return left, left + len(wanted)
    return None


def _pair(record: dict, fmt: str, kind: str, typed_text: str, tokens: list[str], start: int, end: int,
          eeg_units: list[np.ndarray], source_path: Path, doc_span: tuple[int, int] | None = None) -> dict:
    typed_document_id = _task_document_id(typed_text)
    doc_start, doc_end = start, end
    if kind == "query_response":
        doc_text = record["response_seen"].strip()
        response_document_id = _task_document_id(doc_text)
        positive_document_ids = [typed_document_id, response_document_id]
        query_group_id = ""
    else:
        if kind == "question_ict":
            doc_text = record["question_text"].strip()
            doc_start, doc_end = doc_span
        else:
            doc_text = typed_text
        positive_document_ids = [_task_document_id(doc_text)]
        query_group_id = hashlib.sha256("\0".join(map(str, (
            record["participant_id"], record["session_id"], record["part"], record["trial"],
            record["question_id"], "typing", start, end))).encode("utf-8")).hexdigest()
    doc_words = doc_text.split()
    return {
        "query_text": " ".join(tokens[start:end]),
        "query_eeg": np.asarray([unit.T for unit in eeg_units], dtype=np.float32),
        "query_words": tokens[start:end],
        "doc_text": doc_text,
        "doc_eeg": np.zeros((0, 0, 0), dtype=np.float32),
        "doc_words": doc_words,
        "full_sentence_text": doc_text,
        "full_sentence_words": doc_words,
        "query_start_idx": doc_start,
        "query_end_idx": doc_end,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "part": int(record["part"]),
        "trial": int(record["trial"]),
        "sentence_id": int(record["trial"]),
        "question_id": record["question_id"],
        "question_document_id": _task_document_id(record["question_text"]),
        "question_text": record["question_text"],
        "typed_text": typed_text,
        "typed_document_id": typed_document_id,
        "positive_document_ids": positive_document_ids,
        "split_group_id": "task_trial:" + "|".join(
            str(record[field]) for field in ("participant_id", "session_id", "part", "trial", "question_id")),
        "query_group_id": query_group_id,
        "query_kind": f"typing_{fmt}_{'to_response' if kind == 'query_response' else kind}",
        "typing_format": fmt,
        "num_eeg_units": len(eeg_units),
        "condition": record["condition"],
        "chatbot_condition": record["chatbot_condition"],
        "response_seen": record["response_seen"],
        "ground_truth_answer": record["ground_truth_answer"],
        "answer_status": record["answer_status"],
        "answer_provenance": record["answer_provenance"],
        "dataset_source": f"authentic_typing_{fmt}_{kind}",
        "training_source": "authentic_typing",
        "supports_runtime_masking": kind != "query_response",
        "alignment_status": "aligned_keystroke_log",
        "source_eeg_file": str(source_path),
        "document_id": _task_document_id(doc_text),
    }


def _session_inputs(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    manifest = json.loads((dataset_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    sessions = []
    for participant in manifest["participants"]:
        for session in participant.get("sessions", []):
            output_dir = Path(session.get("output_dir", ""))
            if output_dir.name.startswith("ses-") and session.get("selected_psychopy_csv"):
                sessions.append((participant["participant_id"], output_dir,
                                 Path(session["selected_psychopy_csv"])))
    return sessions


def _build_session(participant: str, session_dir: Path, psychopy_csv: Path, records: dict,
                   random_seed: int, query_ratio: float, min_query_length: int, max_query_length: int,
                   pairs_per_trial: int, counters: defaultdict, pairs: dict) -> None:
    session = session_dir.name
    log_path = psychopy_csv.with_suffix(".log")
    segments_path = session_dir / f"{participant}_typing_segments.csv"
    report_path = session_dir / f"{participant}_alignment_report.json"
    if not log_path.is_file() or not segments_path.is_file() or not report_path.is_file():
        counters["sessions_missing_log_segments_or_report"] += 1
        counters[f"session_skipped:{participant}/{session}"] += 1
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not (report.get("success") and report.get("passes_frame_threshold") and report.get("method") == "markers"):
        counters["sessions_failing_alignment"] += 1
        return
    clock_rate = float(report["a"])
    windows = read_windows(log_path)
    starts = _textbox_starts(psychopy_csv)
    segment_files = {}
    for path in sorted((session_dir / "typing_raw_segments").glob("*.fif")):
        segment_files[mne.io.read_raw_fif(path, preload=False, verbose="ERROR").first_samp] = path
    used = set()
    counters["sessions_used"] += 1

    for row in _read_rows(segments_path):
        counters["typing_trials_seen"] += 1
        onset, offset, part = float(row["onset"]), float(row["offset"]), int(float(row["part"]))
        record = records.get(_key(participant, session, row["part"], row["trial"], row["question_id"]))
        if record is None or not record["response_seen"].strip():
            counters["trials_missing_response_record"] += 1
            continue
        candidates = [index for index, window in enumerate(windows)
                      if index not in used and window["part"] == part
                      and onset - CLOCK_TOLERANCE_S <= window["log_start_s"] <= offset]
        if len(candidates) != 1:
            counters["trials_unmatched_or_ambiguous_log_window"] += 1
            continue
        used.add(candidates[0])
        window = windows[candidates[0]]
        characters, unsupported, _ = replay(window["keys"])
        typed_text = (row.get("typed_text") or "").rstrip("\r\n")
        clock_ok = (window["log_end_s"] is not None
                    and abs(window["log_end_s"] - offset) <= CLOCK_TOLERANCE_S
                    and sum(abs(value - window["log_start_s"]) <= CLOCK_TOLERANCE_S
                            for value in starts[part]) == 1)
        if unsupported or not typed_text.strip() or not text_matches(characters, typed_text) or not clock_ok:
            counters["trials_failing_keystroke_validation"] += 1
            continue
        segment_path = segment_files.get(int(row["eeg_start_sample"]))
        if segment_path is None:
            counters["trials_missing_typing_eeg_file"] += 1
            continue
        raw = mne.io.read_raw_fif(segment_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data().astype(np.float32)

        def sample(log_time: float) -> int:
            return int(round(clock_rate * (log_time - onset) * sfreq))

        tokens = typed_text.split()
        word_spans = [match.span() for match in re.finditer(r"\S+", typed_text)]
        char_word = []
        for position in range(len(characters)):
            index = next((i for i, (left, right) in enumerate(word_spans) if position < right), len(word_spans) - 1)
            char_word.append(index)
        word_times = defaultdict(list)
        for position, (character, timestamp) in enumerate(characters):
            if not character.isspace():
                word_times[char_word[position]].append(timestamp)

        keypress = []
        for position, (character, timestamp) in enumerate(characters):
            left = sample(timestamp) - EPOCH_PRE_SAMPLES
            if left < 0 or left + WINDOW_SAMPLES > data.shape[1]:
                counters["keypress_epochs_outside_typing_eeg"] += 1
                continue
            keypress.append((char_word[position], data[:, left:left + WINDOW_SAMPLES], timestamp))
        logged = [event["log_time_s"] for event in window["keys"]]
        continuous_full = _continuous_windows(data, sample(min(logged)) - EPOCH_PRE_SAMPLES,
                                              sample(max(logged)) + WINDOW_SAMPLES - EPOCH_PRE_SAMPLES)
        words = []
        for index, times in sorted(word_times.items()):
            left = sample(min(times)) - EPOCH_PRE_SAMPLES
            if left < 0 or left + WINDOW_SAMPLES > data.shape[1]:
                counters["word_epochs_outside_typing_eeg"] += 1
                continue
            words.append((index, data[:, left:left + WINDOW_SAMPLES]))
        if not keypress or not continuous_full or not words:
            counters["trials_without_usable_typing_eeg"] += 1
            continue
        counters["typing_trials_used"] += 1
        full_units = {"continuous": continuous_full, "keypress": [eeg for _, eeg, _ in keypress],
                      "word": [eeg for _, eeg in words]}
        for fmt in FORMATS:
            pairs[fmt]["query_response"].append(
                _pair(record, fmt, "query_response", typed_text, tokens, 0, len(tokens), full_units[fmt], segment_path))

        word_events = [(index, index, min(times)) for index, times in sorted(word_times.items())]
        rng = random.Random(f"{random_seed}:typing:{participant}:{session}:{row['part']}:{row['trial']}")
        spans = _sample_spans(tokens, word_events, rng=rng, query_ratio=query_ratio,
                              min_query_length=min_query_length, max_query_length=max_query_length,
                              pairs_per_trial=pairs_per_trial, min_fixations=1, counters=counters)
        for start, end, _ in spans:
            span_times = [t for index in range(start, end) for t in word_times.get(index, [])]
            span_keys = [eeg for index, eeg, _ in keypress if start <= index < end]
            span_words = [eeg for index, eeg in words if start <= index < end]
            span_windows = _continuous_windows(data, sample(min(span_times)) - EPOCH_PRE_SAMPLES,
                                               sample(max(span_times)) + WINDOW_SAMPLES - EPOCH_PRE_SAMPLES)
            if not span_keys or not span_windows or not span_words:
                counters["ict_spans_without_usable_eeg"] += 1
                continue
            span_units = {"continuous": span_windows, "keypress": span_keys, "word": span_words}
            question_span = _question_span(record["question_text"], tokens, start, end)
            counters["question_ict_spans_matched" if question_span else "question_ict_spans_not_in_question"] += 1
            for fmt in FORMATS:
                pairs[fmt]["ict"].append(
                    _pair(record, fmt, "ict", typed_text, tokens, start, end, span_units[fmt], segment_path))
                if question_span:
                    pairs[fmt]["question_ict"].append(
                        _pair(record, fmt, "question_ict", typed_text, tokens, start, end, span_units[fmt],
                              segment_path, doc_span=question_span))


def build_typing_pairs(dataset_dir: Path = DEFAULT_DATASET_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
                       random_seed: int = 42, query_ratio: float = 0.3, min_query_length: int = 2,
                       max_query_length: int = 50, pairs_per_trial: int = 2) -> dict:
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    records = {}
    for row in _read_rows(dataset_dir / "trial_response_ground_truth.csv"):
        records[(row["participant_id"], row["session_id"], row["part"], row["trial"], row["question_id"])] = row
    counters = defaultdict(int)
    pairs = {fmt: {kind: [] for kind in KINDS} for fmt in FORMATS}
    for participant, session_dir, psychopy_csv in _session_inputs(dataset_dir):
        _build_session(participant, session_dir, psychopy_csv, records, random_seed, query_ratio,
                       min_query_length, max_query_length, pairs_per_trial, counters, pairs)
    if not all(pairs[fmt][kind] for fmt in FORMATS for kind in KINDS):
        raise ValueError(f"Missing typing pairs: { {fmt: {k: len(v) for k, v in p.items()} for fmt, p in pairs.items()} }")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    definitions = {
        "query_response": "full typing EEG -> same-trial response_seen",
        "ict": "typing-span EEG -> full submitted typed query (span masked at runtime)",
        "question_ict": "typing-span EEG -> displayed question (verbatim-matched span masked at runtime)",
    }
    units = {
        "continuous": "consecutive 513-sample (1 s) windows from first logged key -200 ms to last logged key +800 ms",
        "keypress": "one -200..800 ms epoch per keypress surviving into the submitted query, in typing order",
        "word": "one -200..800 ms epoch per typed word, anchored at the word's first surviving keypress",
    }
    for fmt in FORMATS:
        for kind in KINDS:
            items = pairs[fmt][kind]
            lengths = [pair["num_eeg_units"] for pair in items]
            metadata = {
                "dataset_name": f"authentic_typing_{fmt}_{kind}",
                "version": "authentic_typing_v2",
                "supports_runtime_masking": kind != "query_response",
                "training_source": "authentic_typing",
                "typing_format": fmt,
                "pair_definition": definitions[kind],
                "eeg_units": units[fmt],
                "keystroke_policy": "Only trials whose PsychoPy log replays to the submitted text and passes the 50 ms clock check.",
                "sampling": {"random_seed": random_seed, "query_ratio": query_ratio,
                             "min_query_length": min_query_length, "max_query_length": max_query_length,
                             "pairs_per_trial": pairs_per_trial, "spans_shared_across_formats": True},
                "count": len(items),
                "eeg_units_per_pair": {"min": min(lengths), "median": float(np.median(lengths)),
                                       "max": max(lengths), "over_50": sum(n > 50 for n in lengths),
                                       "over_100": sum(n > 100 for n in lengths)},
                "participants": len({pair["participant_id"] for pair in items}),
                "counters": dict(counters),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            name = f"authentic_typing_{fmt}_{kind}_pairs.npy"
            np.save(output_dir / name, {"ict_pairs": items, "metadata": metadata}, allow_pickle=True)
            (output_dir / name.replace(".npy", ".manifest.json")).write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            summary[name] = {key: metadata[key] for key in ("count", "eeg_units_per_pair", "participants")}
    summary["counters"] = dict(counters)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_typing_pairs(args.dataset_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
