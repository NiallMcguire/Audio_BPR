"""Build Nieuwland ICT pairs from the continuous recordings (OSF eyzaq "Raw data").

Every word of every sentence has its own onset marker in the continuous data
(255 = any word, 201-204 = article, 205/206 = noun, 207/208 = final word), so each word gets
its own EEG window. Sentence text comes from REPLICATION_ITEMS.xlsx and every sentence's
marker count and article/noun positions are checked against it; sentences that do not match
are dropped.

Per word: the paper's common 22 channels, 0.5-100 Hz band-pass, 50 Hz notch, average
reference over the 22 channels, resampled to 500 Hz, window 0-500 ms after onset (250 x 22).

Steps:
  python nieuwland_continuous_reader.py words <recording> <out.npz>   # one participant
  python nieuwland_continuous_reader.py pairs <words_dir> <out.npy>   # all participants -> ICT pairs
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SENTENCE_FILE = Path.home() / "Sentence Materials" / "REPLICATION_ITEMS.xlsx"
CHANNELS = ["F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "Oz"]
SFREQ = 500.0
WINDOW_SAMPLES = 250
WORD_CODES = {255, *range(201, 209)}
SENTENCE_GAP_S = 1.0
MAX_BIOSEMI_DEGREES = 12.0


def sentences_from_sheet() -> dict:
    sheet = pd.read_excel(SENTENCE_FILE, sheet_name="Delong_Replication").set_index("Item Number")
    items = {}
    for number, row in sheet.iterrows():
        for expected, article, noun in ((True, "Expected", "Expected.1"), (False, "Unexpected", "Unexpected.1")):
            words = f"{row['Sentence context']} {row[article]} {row[noun]} {row['Sentence Ending']}".split()
            items[(100 + int(number), expected)] = words
    return items


def _repair_cnt_header(path: Path) -> None:
    """Some Neuroscan files store a corrupt sample count; recompute it from the event-table position."""
    with path.open("r+b") as stream:
        header = stream.read(900)
        n_channels = int(np.frombuffer(header[370:372], "<u2")[0])
        stored = int(np.frombuffer(header[864:868], "<i4")[0])
        event_table = int(np.frombuffer(header[886:890], "<i4")[0])
        samples = (event_table - 900 - 75 * n_channels) // (4 * n_channels)
        if stored != samples:
            stream.seek(864)
            stream.write(np.int32(samples).tobytes())


def load_recording(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".cnt":
        _repair_cnt_header(path)
        raw = mne.io.read_raw_cnt(path, data_format="int32", preload=True, verbose="ERROR")
        if len(raw.annotations) and raw.annotations.onset.max() > raw.times[-1]:
            raise ValueError("markers fall after the end of the data; sample format not int32")
        return raw
    readers = {".vhdr": mne.io.read_raw_brainvision, ".bdf": mne.io.read_raw_bdf}
    return readers[suffix](path, preload=True, verbose="ERROR")


def marker_events(raw: mne.io.BaseRaw) -> list[tuple[float, int]]:
    """(onset seconds from recording start, code) for every stimulus marker."""
    out = []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        match = re.search(r"(\d+)\s*$", description)
        if match and ("Stimulus" in description or description.strip().isdigit()):
            out.append((float(onset), int(match.group(1))))
    if not out and "Status" in raw.ch_names:
        events = mne.find_events(raw, stim_channel="Status", shortest_event=1, mask=255, mask_type="and",
                                 verbose="ERROR")
        out = [((sample - raw.first_samp) / raw.info["sfreq"], int(code)) for sample, _, code in events]
    return out


def sentence_runs(events: list[tuple[float, int]]) -> list[dict]:
    """Group word markers into sentences (gaps > 1 s) and attach each sentence's item marker."""
    runs, current, item = [], [], None
    last_word = None
    for onset, code in sorted(events):
        if code in WORD_CODES:
            if last_word is not None and onset - last_word > SENTENCE_GAP_S and current:
                runs.append({"item": item, "words": current})
                current, item = [], None
            current.append((onset, code))
            last_word = onset
        elif 101 <= code <= 180 and item is None and current:
            item = code
    if current:
        runs.append({"item": item, "words": current})
    return runs


def match_sentences(runs: list[dict], items: dict) -> tuple[list[dict], dict]:
    kept, counts = [], {"runs": len(runs), "no_article": 0, "no_item": 0, "word_count_mismatch": 0}
    for run in runs:
        codes = [code for _, code in run["words"]]
        articles = [i for i, code in enumerate(codes) if 201 <= code <= 204]
        if len(articles) != 1:
            counts["no_article"] += 1  # control-experiment sentences have no 201-204 article marker
            continue
        if run["item"] is None:
            counts["no_item"] += 1
            continue
        a = articles[0]
        words = items[(run["item"], codes[a] in (201, 202))]
        if (len(words) != len(codes) or words[a].lower() not in ("a", "an")
                or a + 1 >= len(codes) or codes[a + 1] not in (205, 206)):
            counts["word_count_mismatch"] += 1
            continue
        kept.append({"item": run["item"], "expected": codes[a] in (201, 202), "words": words,
                     "onsets": [onset for onset, _ in run["words"]]})
    counts["kept"] = len(kept)
    return kept, counts


def channel_picks(raw: mne.io.BaseRaw) -> dict:
    """Recording channel for each of the 22 target channels (by name, or nearest BioSemi direction)."""
    by_name = {name.upper(): name for name in raw.ch_names}
    picks = {target: by_name[target.upper()] for target in CHANNELS if target.upper() in by_name}
    if len(picks) < len(CHANNELS) and {"A1", "B1", "C1", "D1"} <= set(raw.ch_names):
        biosemi = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
        standard = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        names = [name for name in biosemi if name in raw.ch_names]
        directions = np.array([biosemi[name] / np.linalg.norm(biosemi[name]) for name in names])
        for target in CHANNELS:
            if target not in picks:
                angle = np.degrees(np.arccos(np.clip(directions @ (standard[target] / np.linalg.norm(standard[target])),
                                                     -1, 1)))
                if angle.min() <= MAX_BIOSEMI_DEGREES:
                    picks[target] = names[int(angle.argmin())]
    return picks


def build_words(recording: Path, output: Path) -> dict:
    items = sentences_from_sheet()
    raw = load_recording(recording)
    events = marker_events(raw)
    sentences, counts = match_sentences(sentence_runs(events), items)
    picks = channel_picks(raw)
    report = {"recording": str(recording), "sfreq_recorded": raw.info["sfreq"], "duration_s": raw.times[-1],
              "channels_found": len(picks), "channel_map": picks, **counts}
    if not sentences or len(picks) < len(CHANNELS) - 2:
        report["status"] = "skipped"
        output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        return report

    raw.pick(list(picks.values()))
    raw.rename_channels({source: target for target, source in picks.items()})
    missing = [name for name in CHANNELS if name not in picks]
    raw.set_channel_types({name: "eeg" for name in raw.ch_names})
    nyquist_cap = min(100.0, raw.info["sfreq"] * 0.45)
    raw.filter(0.5, nyquist_cap, verbose="ERROR")
    if raw.info["sfreq"] > 110:
        raw.notch_filter(50.0, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.resample(SFREQ, verbose="ERROR")
    data = raw.get_data().astype(np.float32) * 1e6  # microvolts
    order = [raw.ch_names.index(name) if name in raw.ch_names else None for name in CHANNELS]

    word_eeg, words, sentence_index = [], [], []
    for s, sentence in enumerate(sentences):
        for word, onset in zip(sentence["words"], sentence["onsets"]):
            start = int(round(onset * SFREQ))
            window = np.zeros((WINDOW_SAMPLES, len(CHANNELS)), dtype=np.float32)
            segment = data[:, start:start + WINDOW_SAMPLES]
            for c, index in enumerate(order):
                if index is not None:
                    window[:segment.shape[1], c] = segment[index]
            word_eeg.append(window)
            words.append(word)
            sentence_index.append(s)
    np.savez_compressed(output, word_eeg=np.stack(word_eeg), words=np.array(words), sentence_index=np.array(sentence_index),
                        items=np.array([s["item"] for s in sentences]),
                        expected=np.array([s["expected"] for s in sentences]))
    report.update({"status": "ok", "sentences": len(sentences), "words": len(words), "missing_channels": missing})
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def build_pairs(words_dir: Path, output: Path, seed: int = 42, query_ratio: float = 0.3, min_query: int = 2,
                min_sentence: int = 6, pairs_per_sentence: int = 2) -> dict:
    """ICT pairs as in nieuwland_ict_reader_v2: span of ~30% of the words -> full sentence (masked at runtime)."""
    rng = random.Random(seed)
    pairs, participants = [], []
    for path in sorted(words_dir.glob("*.npz")):
        participant = path.stem
        store = np.load(path)
        word_eeg, words, index = store["word_eeg"], store["words"].tolist(), store["sentence_index"]
        participants.append(participant)
        for s, item in enumerate(store["items"].tolist()):
            positions = np.flatnonzero(index == s)
            sentence = [words[i] for i in positions]
            if len(sentence) < min_sentence:
                continue
            length = min(max(min_query, int(len(sentence) * query_ratio)), len(sentence) - 1)
            made, attempts = 0, 0
            while made < pairs_per_sentence and attempts < pairs_per_sentence * 5:
                attempts += 1
                start = rng.randint(0, len(sentence) - length)
                end = start + length
                pairs.append({
                    "query_text": " ".join(sentence[start:end]), "query_words": sentence[start:end],
                    "query_eeg": word_eeg[positions[start:end]],
                    "doc_text": " ".join(sentence), "doc_words": sentence,
                    "full_sentence_text": " ".join(sentence), "full_sentence_words": sentence,
                    "query_start_idx": start, "query_end_idx": end,
                    "participant_id": participant, "sentence_id": int(item),
                    "expected_condition": bool(store["expected"][s]), "fs": SFREQ,
                    "supports_runtime_masking": True,
                })
                made += 1
    metadata = {"dataset_name": "nieuwland_continuous", "version": "nieuwland_continuous_v3",
                "created": datetime.now().isoformat(), "supports_runtime_masking": True,
                "source": "OSF eyzaq Raw data (continuous recordings), word onset markers",
                "channels": CHANNELS, "sfreq": SFREQ, "window": "0-500 ms after word onset",
                "preprocessing": "0.5-100 Hz band-pass, 50 Hz notch, average reference over 22 channels",
                "participants": len(participants), "generated_ict_pairs": len(pairs), "random_seed": seed,
                "ict_generation_params": {"query_length_ratio": query_ratio, "min_query_length": min_query,
                                          "min_sentence_length": min_sentence,
                                          "max_pairs_per_sentence": pairs_per_sentence}}
    np.save(output, {"ict_pairs": pairs, "metadata": metadata}, allow_pickle=True)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    mode, source, target = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    result = build_words(source, target) if mode == "words" else build_pairs(source, target)
    print(json.dumps({k: v for k, v in result.items() if k != "channel_map"}, indent=2))
