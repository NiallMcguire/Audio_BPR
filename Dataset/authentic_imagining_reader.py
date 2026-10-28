"""Build imagined-query EEG pairs (main task) and imagining ICT pairs.

The imagining interval is the 10 s fixation period between reading the question and typing.
Main pairs: imagining EEG -> the response the participant was shown (other targets available
via --targets). They are never masked.

Formats for main pairs (all units are 513 samples, matching reading/typing):
- window: consecutive 1 s windows covering the whole interval (baseline, crops, multiple instance, transfer).
- pace: one segment per typed word, each lasting the participant's mean reading fixation duration,
  laid end to end from the start of the interval (zero-padded to 513 samples).
- uniform: the same segments spread evenly across the interval (control).
- shuffled: the same segments at random positions in the interval (control).

Imagining ICT: two 2-4 s crops of the window per trial -> the displayed question, with a span masked:
- position: the words at the same relative position in the question as the crop is in the window.
- pace: the words the participant's reading pace places inside the crop (unmasked if none).
- random: a random span of the same length as the position span (control).
"""

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
    from .authentic_query_reader import _key, _read_rows
    from .authentic_query_ict_reader import _task_document_id
    from .authentic_typing_reader import WINDOW_SAMPLES, _continuous_windows
except ImportError:
    from authentic_query_reader import _key, _read_rows
    from authentic_query_ict_reader import _task_document_id
    from authentic_typing_reader import WINDOW_SAMPLES, _continuous_windows

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "authentic_imagining_pairs"
FORMATS = ("window", "pace", "uniform", "shuffled")
TARGETS = {"question": "question_text", "typed": "typed_text", "response": "response_seen"}
ICT_VARIANTS = ("position", "pace", "random")
ICT_CROPS_PER_TRIAL = 2
ICT_CROP_UNITS = (2, 4)


def _cap(start: int, end: int, n_words: int) -> tuple[int, int]:
    """Keep at least one unmasked word so the masked question is never empty."""
    end = min(end, n_words)
    if end - start >= n_words:
        end = start + n_words - 1
    return start, end


def _ict_spans(n_words: int, crop_start: int, crop_units: int, n_units: int, unit_seconds: float,
               pace_seconds: float | None, rng: random.Random) -> dict[str, tuple[int, int] | None]:
    start = int(math.floor(crop_start / n_units * n_words))
    end = max(start + 1, int(math.ceil((crop_start + crop_units) / n_units * n_words)))
    position = _cap(start, end, n_words)
    pace = None
    if pace_seconds:
        t0, t1 = crop_start * unit_seconds, (crop_start + crop_units) * unit_seconds
        start, end = int(math.floor(t0 / pace_seconds)), min(n_words, int(math.ceil(t1 / pace_seconds)))
        if start < end:
            pace = _cap(start, end, n_words)
    length = position[1] - position[0]
    random_start = rng.randint(0, n_words - length)
    return {"position": position, "pace": pace, "random": (random_start, random_start + length)}


def _participant_fixation_seconds(dataset_dir: Path) -> dict[str, float]:
    durations = defaultdict(list)
    for path in dataset_dir.glob("P*/ses-*/P*_reading_fixations.csv"):
        participant = path.parents[1].name
        for row in _read_rows(path):
            try:
                durations[participant].append(float(row["duration_s"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {participant: float(np.mean(values)) for participant, values in durations.items() if values}


def _segments(data: np.ndarray, onsets: list[int], length: int) -> list[np.ndarray]:
    units = []
    for onset in onsets:
        chunk = data[:, onset:onset + length]
        units.append(np.pad(chunk, ((0, 0), (0, WINDOW_SAMPLES - chunk.shape[1]))))
    return units


def _pair(record: dict, fmt: str, target: str, units: list[np.ndarray], source_path: Path,
          extra: dict) -> dict:
    doc_text = (record[TARGETS[target]] or "").strip()
    doc_words = doc_text.split()
    typed = (record.get("typed_text") or "").strip()
    return {
        "query_text": typed or record["question_text"],
        "query_eeg": np.asarray([unit.T for unit in units], dtype=np.float32),
        "query_words": (typed or record["question_text"]).split(),
        "doc_text": doc_text,
        "doc_eeg": np.zeros((0, 0, 0), dtype=np.float32),
        "doc_words": doc_words,
        "full_sentence_text": doc_text,
        "full_sentence_words": doc_words,
        "query_start_idx": 0,
        "query_end_idx": 0,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "part": int(record["part"]),
        "trial": int(record["trial"]),
        "sentence_id": int(record["trial"]),
        "question_id": record["question_id"],
        "question_document_id": _task_document_id(record["question_text"]),
        "question_text": record["question_text"],
        "typed_text": typed,
        "positive_document_ids": [_task_document_id(doc_text)],
        "split_group_id": "task_trial:" + "|".join(
            str(record[field]) for field in ("participant_id", "session_id", "part", "trial", "question_id")),
        "query_group_id": "",
        "query_kind": f"imagining_{fmt}_to_{target}",
        "imagining_format": fmt,
        "imagining_target": target,
        "num_eeg_units": len(units),
        "condition": record["condition"],
        "chatbot_condition": record["chatbot_condition"],
        "response_seen": record["response_seen"],
        "ground_truth_answer": record["ground_truth_answer"],
        "answer_status": record["answer_status"],
        "answer_provenance": record["answer_provenance"],
        "dataset_source": f"authentic_imagining_{fmt}_{target}",
        "training_source": "authentic_imagining",
        "supports_runtime_masking": False,
        "alignment_status": "aligned_imagining_segment",
        "source_eeg_file": str(source_path),
        "document_id": _task_document_id(doc_text),
        **extra,
    }


def build_imagining_pairs(dataset_dir: Path = DEFAULT_DATASET_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
                          random_seed: int = 42, targets: tuple[str, ...] = ("response",)) -> dict:
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    records = {}
    for row in _read_rows(dataset_dir / "trial_response_ground_truth.csv"):
        records[(row["participant_id"], row["session_id"], row["part"], row["trial"], row["question_id"])] = row
    fixation_seconds = _participant_fixation_seconds(dataset_dir)
    counters = defaultdict(int)
    pairs = {fmt: {target: [] for target in targets} for fmt in FORMATS}
    ict_pairs = {variant: [] for variant in ICT_VARIANTS}

    for segments_path in sorted(dataset_dir.glob("P*/ses-*/P*_imagining_segments.csv")):
        session_dir = segments_path.parent
        participant, session = session_dir.parent.name, session_dir.name
        report = json.loads((session_dir / f"{participant}_alignment_report.json").read_text(encoding="utf-8"))
        if not (report.get("success") and report.get("passes_frame_threshold")):
            counters["sessions_failing_alignment"] += 1
            continue
        files = {mne.io.read_raw_fif(path, preload=False, verbose="ERROR").first_samp: path
                 for path in (session_dir / "imagining_raw_segments").glob("*.fif")}
        pace_seconds = fixation_seconds.get(participant)
        for row in _read_rows(segments_path):
            counters["imagining_segments_seen"] += 1
            record = records.get(_key(participant, session, row["part"], row["trial"], row["question_id"]))
            if record is None:
                counters["segments_missing_trial_record"] += 1
                continue
            path = files.get(int(row["eeg_start_sample"]))
            if path is None:
                counters["segments_missing_eeg_file"] += 1
                continue
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            data = raw.get_data().astype(np.float32)
            counters["imagining_segments_used"] += 1

            units = {"window": _continuous_windows(data, 0, data.shape[1])}
            typed_words = (record.get("typed_text") or "").split()
            extra = {"pace_fixation_s": pace_seconds, "pace_words": len(typed_words)}
            if typed_words and pace_seconds:
                length = min(WINDOW_SAMPLES, max(1, int(round(pace_seconds * sfreq))))
                count = min(len(typed_words), data.shape[1] // length)
                rng = random.Random(f"{random_seed}:imagining:{participant}:{session}:{row['part']}:{row['trial']}")
                units["pace"] = _segments(data, [i * length for i in range(count)], length)
                units["uniform"] = _segments(
                    data, np.linspace(0, data.shape[1] - length, count).round().astype(int).tolist(), length)
                units["shuffled"] = _segments(
                    data, sorted(rng.randint(0, data.shape[1] - length) for _ in range(count)), length)
                counters["pace_words_truncated_to_fit"] += len(typed_words) - count
            else:
                counters["segments_without_typed_words_or_pace"] += 1

            for fmt, fmt_units in units.items():
                for target in targets:
                    if not (record[TARGETS[target]] or "").strip():
                        counters[f"missing_target_text:{target}"] += 1
                        continue
                    pairs[fmt][target].append(_pair(record, fmt, target, fmt_units, path, extra))

            question_words = record["question_text"].split()
            if len(question_words) < 3:
                counters["questions_too_short_for_imagining_ict"] += 1
                continue
            window = units["window"]
            crop_rng = random.Random(f"{random_seed}:imagining_ict:{participant}:{session}:{row['part']}:{row['trial']}")
            crops = set()
            while len(crops) < ICT_CROPS_PER_TRIAL:
                width = crop_rng.randint(*ICT_CROP_UNITS)
                crops.add((crop_rng.randint(0, len(window) - width), width))
            for crop_start, width in sorted(crops):
                spans = _ict_spans(len(question_words), crop_start, width, len(window), WINDOW_SAMPLES / sfreq,
                                   pace_seconds, crop_rng)
                for variant, span in spans.items():
                    pair = _pair(record, "window", "question", window[crop_start:crop_start + width], path, extra)
                    pair.update({
                        "query_start_idx": span[0] if span else 0,
                        "query_end_idx": span[1] if span else 0,
                        "supports_runtime_masking": span is not None,
                        "query_group_id": hashlib.sha256("\0".join(map(str, (
                            participant, session, row["part"], row["trial"], crop_start, width))).encode()).hexdigest(),
                        "query_kind": f"imagining_ict_{variant}",
                        "imagining_format": "ict_crop",
                        "imagining_ict_variant": variant,
                        "crop_start_unit": crop_start,
                        "crop_units": width,
                        "dataset_source": f"authentic_imagining_ict_{variant}",
                    })
                    if span is None:
                        counters[f"ict_{variant}_crops_without_words"] += 1
                    ict_pairs[variant].append(pair)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    outputs = [(f"authentic_imagining_{fmt}_{target}", pairs[fmt][target], fmt, target, False,
                f"10 s imagining-interval EEG -> {TARGETS[target]}")
               for fmt in FORMATS for target in targets]
    outputs += [(f"authentic_imagining_ict_{variant}_question", ict_pairs[variant], "ict_crop", "question", True,
                 f"2-4 s imagining crop -> displayed question, {variant} span masked at runtime")
                for variant in ICT_VARIANTS]
    for name, items, fmt, target, masking, definition in outputs:
        if not items:
            raise ValueError(f"No imagining pairs for {name}")
        lengths = [pair["num_eeg_units"] for pair in items]
        metadata = {
            "dataset_name": name,
            "version": "authentic_imagining_v2",
            "supports_runtime_masking": masking,
            "training_source": "authentic_imagining",
            "imagining_format": fmt,
            "imagining_target": target,
            "pair_definition": definition,
            "count": len(items),
            "participants": len({pair["participant_id"] for pair in items}),
            "eeg_units_per_pair": {"min": min(lengths), "median": float(np.median(lengths)),
                                   "max": max(lengths)},
            "random_seed": random_seed,
            "counters": dict(counters),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        np.save(output_dir / f"{name}_pairs.npy", {"ict_pairs": items, "metadata": metadata}, allow_pickle=True)
        (output_dir / f"{name}_pairs.manifest.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                                 encoding="utf-8")
        summary[f"{name}_pairs.npy"] = {key: metadata[key] for key in ("count", "participants", "eeg_units_per_pair")}
    summary["counters"] = dict(counters)
    summary["mean_fixation_seconds"] = {
        "min": min(fixation_seconds.values()), "median": float(np.median(list(fixation_seconds.values()))),
        "max": max(fixation_seconds.values())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=["response"])
    args = parser.parse_args()
    print(json.dumps(build_imagining_pairs(args.dataset_dir, args.output_dir, args.seed, tuple(args.targets)),
                     indent=2))


if __name__ == "__main__":
    main()
