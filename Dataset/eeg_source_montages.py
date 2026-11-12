"""Record each EEG source's sampling rate and channel layout, and map them to 10-10 names.

Writes Dataset/eeg_source_montages.json (read by the foundation-model input adapter) and
findings/eeg_montage_report.txt.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = Path(__file__).resolve().parent / "eeg_source_montages.json"
OUT_REPORT = REPO / "findings" / "eeg_montage_report.txt"
ALICE_DIR = Path.home() / "Alice"
NIEUWLAND_DIR = Path.home() / "Processed segmented data"
ALICE_MAX_MM = 20.0
EAR_ALIASES = {"LEFT EAR": "A1", "RIGHT EAR": "A2"}


def task_montage() -> dict:
    segment = next(REPO.glob("dataset/P0*/ses-*/typing_raw_segments/*.fif"))
    raw = mne.io.read_raw_fif(segment, preload=False, verbose="ERROR")
    epochs_file = next(REPO.glob("dataset/P0*/ses-*/*_reading-epo.fif"))
    epochs = mne.read_epochs(epochs_file, preload=False, verbose="ERROR")
    if epochs.ch_names != raw.ch_names or epochs.info["sfreq"] != raw.info["sfreq"]:
        raise ValueError("Reading epochs and typing/imagining segments use different channel layouts")
    names = list(raw.ch_names)
    return {
        "sfreq": float(raw.info["sfreq"]),
        "unit_samples": 513,
        "channel_names": names,
        "mapped_names": [EAR_ALIASES.get(name, name) for name in names],
        "scalp_channels": [name for name in names if name not in EAR_ALIASES],
        "filters": {"highpass": raw.info["highpass"], "lowpass": raw.info["lowpass"]},
        "reference": "recording reference (no re-referencing applied in the pair files)",
        "checked_files": [str(segment), str(epochs_file)],
    }


def alice_montage(task_scalp: list[str]) -> dict:
    labels_by_file = {}
    fsample = set()
    for path in sorted(ALICE_DIR.glob("S*.mat"))[::16]:
        raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["raw"]
        labels_by_file[path.name] = [str(label) for label in raw.label]
        fsample.add(float(raw.fsample))
    # The last channel is an auxiliary input named Aux5 or AUD; only the electrodes must agree.
    label_sets = {tuple(labels[:-1]) for labels in labels_by_file.values()}
    if len(label_sets) != 1 or len(fsample) != 1:
        raise ValueError(f"Alice files differ in labels or sampling rate: {labels_by_file}, {fsample}")
    labels = [*next(iter(label_sets)), "AUX"]

    m10 = mne.channels.make_standard_montage("easycap-M10").get_positions()["ch_pos"]
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    targets = np.array([std[name] for name in task_scalp])
    candidates = []
    for label in labels:
        if label in m10:
            distances = np.linalg.norm(targets - m10[label], axis=1) * 1000
            index = int(distances.argmin())
            candidates.append((float(distances[index]), label, task_scalp[index]))
    mapped, used = {}, set()
    for distance, label, name in sorted(candidates):
        if distance <= ALICE_MAX_MM and name not in used:
            mapped[label] = (name, round(distance, 1))
            used.add(name)
    return {
        "sfreq_recorded": fsample.pop(),
        "sfreq": 128.0,
        "unit_samples": "variable (one word, zero-padded to the longest word in the file)",
        "channel_names": labels + [f"pad{i}" for i in range(128 - len(labels))],
        "mapped_names": [mapped[label][0] if label in mapped else None for label in labels]
        + [None] * (128 - len(labels)),
        "mapping_distance_mm": {label: mapped[label][1] for label in mapped},
        "mapping_rule": f"EasyCap M10 numbered electrode -> nearest task 10-10 electrode within "
                        f"{ALICE_MAX_MM:.0f} mm (one-to-one), MNE template positions",
        "preprocessing": "resampled 500->128 Hz, 4th-order Butterworth 0.5 Hz to 0.99*Nyquist, "
                         "average reference over all 62 recorded channels (incl. VEOG, Aux5), "
                         "zero-padded to 128 channels (alice_ict_reader_v2.Preprocess)",
        "checked_files": sorted(labels_by_file),
    }


def nieuwland_montage() -> dict:
    headers = sorted(NIEUWLAND_DIR.glob("*.vhdr"))
    fields = {key: Counter() for key in ("NumberOfChannels", "SamplingInterval", "DataOrientation",
                                         "BinaryFormat")}
    channel_sets = Counter()
    for header in headers:
        text = header.read_text(encoding="utf-8", errors="ignore")
        for key in fields:
            match = re.search(rf"^{key}=(.+)$", text, re.MULTILINE)
            if match:
                fields[key][match.group(1).strip()] += 1
        channel_section = text.split("[Channel Infos]", 1)[-1].split("\n[", 1)[0]
        channel_sets[tuple(re.findall(r"^Ch\d+=([^,]+),", channel_section, re.MULTILINE))] += 1
    history = sorted(NIEUWLAND_DIR.glob("*.ehst2"))
    from nieuwland_continuous_reader import CHANNELS, SFREQ, WINDOW_SAMPLES
    return {
        "status": "rebuilt from continuous recordings (nieuwland_continuous_reader.py)",
        "old_pair_files": "nieuwland_ict_pairs_RUNTIME_MASKING.npy was built from 1.5 s article/noun epochs "
                          "read as int16 multiplexed data; do not use",
        "segmented_file_format": {key: dict(counter) for key, counter in fields.items()},
        "headers_checked": len(headers),
        "history_files_without_vhdr": len(history),
        "channel_sets": [{"count": count, "channels": list(names)} for names, count in channel_sets.items()],
        "sfreq": SFREQ,
        "unit_samples": WINDOW_SAMPLES,
        "channel_names": CHANNELS,
        "mapped_names": CHANNELS,
        "preprocessing": "0.5-100 Hz band-pass, 50 Hz notch, average reference over the 22 channels",
    }


def write_report(sources: dict) -> None:
    task, alice, nieuwland = sources["authentic"], sources["alice"], sources["nieuwland"]
    alice_mapped = [name for name in alice["mapped_names"] if name]
    lines = [
        "EEG MONTAGE AND SAMPLING-RATE CHECK",
        "===================================",
        "",
        "Task (reading, typing, imagining)",
        f"- {task['sfreq']:.0f} Hz; units are {task['unit_samples']} samples (1 s).",
        f"- {len(task['channel_names'])} channels: {len(task['scalp_channels'])} scalp (10-10 names) "
        f"+ LEFT EAR / RIGHT EAR (used as A1 / A2).",
        f"- Filters in the files: highpass {task['filters']['highpass']} Hz, lowpass {task['filters']['lowpass']} Hz.",
        f"- Reference: {task['reference']}.",
        "",
        "Alice (audio)",
        f"- Recorded at {alice['sfreq_recorded']:.0f} Hz; the pair files are at {alice['sfreq']:.0f} Hz.",
        f"- {alice['preprocessing']}.",
        f"- {len([c for c in alice['channel_names'] if not c.startswith('pad')])} recorded channels, "
        f"numbered EasyCap M10 positions (no 10-10 names).",
        f"- {len(alice_mapped)} electrodes map to a task 10-10 electrode ({alice['mapping_rule']}).",
        "",
        "Nieuwland (visual)",
        f"- Rebuilt from the continuous recordings: {nieuwland['sfreq']:.0f} Hz, "
        f"{nieuwland['unit_samples']}-sample word windows, {len(nieuwland['channel_names'])} channels "
        f"({' '.join(nieuwland['channel_names'])}); {nieuwland['preprocessing']}.",
        f"- The old pair files: {nieuwland['old_pair_files']}.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    task = task_montage()
    sources = {"authentic": task, "alice": alice_montage(task["scalp_channels"]), "nieuwland": nieuwland_montage()}
    OUT_JSON.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    write_report(sources)
    print(OUT_JSON)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
