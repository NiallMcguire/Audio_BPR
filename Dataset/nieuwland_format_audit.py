"""Audit how nieuwland_ict_reader_v2 decoded each Nieuwland recording against the files' own format.

For every participant the reader used, compares:
- the reader's decode (from the regeneration log): int16, multiplexed, guessed channel count, 500 Hz;
- the declared format (BrainVision .vhdr header or Analyzer .ehst2 history): data type, orientation,
  channel names, points per segment;
- a data check: the file decoded as declared (int32, one channel after another) must fill the file
  exactly and give smooth signals (high lag-1 autocorrelation), as EEG does.

Writes findings/nieuwland_format_audit.csv and findings/nieuwland_format_audit.txt.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "Processed segmented data"
REGEN_LOG = Path.home() / "BPR_mean_pooling_inputs" / "nieuwland_regen.log"
FINDINGS = Path(__file__).resolve().parents[2] / "findings"
SEGMENT_SECONDS = 1.5  # every .vhdr pairs SegmentDataPoints with a sampling interval giving 1.5 s
SUFFIX = {"INT_32": "int32", "INT_16": "int16", "IEEE_FLOAT_32": "float32"}


def reader_decodes() -> dict:
    text = REGEN_LOG.read_text(encoding="utf-8", errors="ignore")
    pattern = r"Participant: (\S+)\n(?:.*\n){0,3}?.*EEG data loaded: \((\d+), (\d+)\) \(\d+ch, <class 'numpy\.(\w+)'>\)"
    return {pid: {"reader_channels": int(ch), "reader_samples": int(n), "reader_dtype": dtype}
            for pid, ch, n, dtype in re.findall(pattern, text)}


def declared_format(pid: str) -> dict:
    header = DATA_DIR / f"seg_{pid}.vhdr"
    if header.is_file():
        text = header.read_text(encoding="utf-8", errors="ignore")
        channels = re.findall(r"^Ch\d+=([^,]+),", text.split("[Channel Infos]", 1)[-1].split("\n[", 1)[0], re.M)
        field = lambda key: (re.search(rf"^{key}=(.+)$", text, re.M) or [None, ""])[1].strip()  # noqa: E731
        return {"format_source": "vhdr", "declared_dtype": SUFFIX.get(field("BinaryFormat"), field("BinaryFormat")),
                "declared_orientation": field("DataOrientation").lower(), "channel_names": channels,
                "segment_points": int(field("SegmentDataPoints") or 0)}
    history = DATA_DIR / f"seg_{pid}.ehst2"
    if history.is_file():
        text = history.read_bytes().decode("utf-8", errors="ignore")
        names = list(dict.fromkeys(re.findall(r"<ChannelName>([^<]+)</ChannelName>", text)))
        multiplexed = re.search(r"<Multiplexed>(\w+)</Multiplexed>", text)
        dtype = re.search(r"<DataFormat>(\w+)</DataFormat>", text)
        points = re.search(r"<SegmentDataPoints>(\d+)</SegmentDataPoints>", text)
        return {"format_source": "ehst2", "declared_dtype": SUFFIX.get(dtype[1], dtype[1]) if dtype else "",
                "declared_orientation": ("multiplexed" if multiplexed and multiplexed[1] == "true" else
                                         "vectorized" if multiplexed else ""),
                "channel_names": names, "segment_points": int(points[1]) if points else 0}
    return {"format_source": "none", "declared_dtype": "", "declared_orientation": "", "channel_names": [],
            "segment_points": 0}


def data_check(path: Path, n_channels: int, dtype: str, segment_points: int) -> dict:
    size = path.stat().st_size
    if not n_channels or not dtype or size % (n_channels * np.dtype(dtype).itemsize):
        return {"declared_fills_file": False, "declared_lag1_autocorr": "", "segments": ""}
    data = np.memmap(path, dtype=dtype, mode="r").reshape(n_channels, -1)
    window = np.asarray(data[:, :5000], dtype=np.float64)
    whole = segment_points and data.shape[1] % segment_points == 0
    return {"declared_fills_file": bool(whole), "segments": data.shape[1] // segment_points if whole else "",
            "declared_lag1_autocorr": round(float(np.median(
                [np.corrcoef(ch[:-1], ch[1:])[0, 1] for ch in window if ch.std() > 0])), 3)}


def reader_check(path: Path, channels: int) -> float:
    data = np.memmap(path, dtype=np.int16, mode="r")
    data = np.asarray(data[:channels * 5000], dtype=np.float64).reshape(-1, channels).T
    return round(float(np.median([np.corrcoef(ch[:-1], ch[1:])[0, 1] for ch in data if ch.std() > 0])), 3)


def lab(pid: str) -> str:
    return re.match(r"[A-Za-z]+", pid)[0].lower()


def main() -> None:
    reader = reader_decodes()
    declared_by_pid = {pid: declared_format(pid) for pid in reader}
    # Participants without a header or history file: use the layout of the same lab's other files.
    lab_layouts = {}
    for pid, declared in declared_by_pid.items():
        if declared["channel_names"]:
            lab_layouts.setdefault(lab(pid), Counter())[
                (declared["declared_dtype"], declared["declared_orientation"],
                 tuple(declared["channel_names"]), declared["segment_points"])] += 1
    rows = []
    for pid, decode in sorted(reader.items()):
        declared = declared_by_pid[pid]
        if declared["format_source"] == "none" and lab(pid) in lab_layouts:
            dtype, orientation, names, points = lab_layouts[lab(pid)].most_common(1)[0][0]
            declared = {"format_source": "same lab", "declared_dtype": dtype, "declared_orientation": orientation,
                        "channel_names": list(names), "segment_points": points}
        dat = DATA_DIR / f"seg_{pid}.dat"
        n_channels = len(declared["channel_names"])
        row = {"participant": pid, **decode, **{k: v for k, v in declared.items() if k != "channel_names"},
               "declared_channels": n_channels, "channel_names": " ".join(declared["channel_names"]),
               "declared_sfreq": round(declared["segment_points"] / SEGMENT_SECONDS, 2)
               if declared["segment_points"] else "",
               "file_bytes": dat.stat().st_size}
        row.update(data_check(dat, n_channels, declared["declared_dtype"], declared["segment_points"]))
        row["reader_lag1_autocorr"] = reader_check(dat, decode["reader_channels"])
        row["reader_matches_declared"] = (decode["reader_dtype"] == declared["declared_dtype"]
                                          and decode["reader_channels"] == n_channels
                                          and declared["declared_orientation"] == "multiplexed")
        rows.append(row)

    FINDINGS.mkdir(exist_ok=True)
    with (FINDINGS / "nieuwland_format_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with_format = [r for r in rows if r["format_source"] != "none"]
    lines = [
        "NIEUWLAND FORMAT AUDIT",
        "======================",
        f"Participants the reader used (others had no sentence triggers and were skipped): {len(rows)}",
        f"Reader decodes: {dict(Counter((r['reader_dtype'], r['reader_channels']) for r in rows))} "
        "(dtype, channels); always multiplexed, always 500 Hz.",
        f"Format information available: {dict(Counter(r['format_source'] for r in rows))}",
        f"Declared formats: {dict(Counter((r['declared_dtype'], r['declared_orientation']) for r in with_format))}",
        f"Declared channel counts: {dict(Counter(r['declared_channels'] for r in with_format))}",
        f"Declared sampling rates (Hz): {dict(Counter(r['declared_sfreq'] for r in with_format))}",
        f"Reader decode matches the declared format: {sum(r['reader_matches_declared'] for r in rows)} "
        f"of {len(with_format)} with format information.",
        f"Declared format fills the file exactly (whole 1.5 s segments): "
        f"{sum(r['declared_fills_file'] is True for r in with_format)} of {len(with_format)}.",
        f"1.5 s segments per participant: {sorted(r['segments'] for r in with_format if r['segments'] != '')}",
        "Median lag-1 autocorrelation (EEG is smooth, so close to 1 means a correct decode):",
        f"  declared decode: {np.median([r['declared_lag1_autocorr'] for r in with_format if r['declared_lag1_autocorr'] != '']):.3f}",
        f"  reader decode:   {np.median([r['reader_lag1_autocorr'] for r in rows]):.3f}",
        "",
        "Notes",
        "- Files that do not fill whole segments (Kent, Stirling STIR7, Oxford) most likely hold 2 more",
        "  channels (eye channels) than their history file lists; their exact channel count is unconfirmed.",
        "- Every file is a set of 1.5 s epochs (0.5 s before to 1 s after a marked word), not a continuous",
        "  recording, so sentence-long word sequences cannot be cut from these files.",
        "",
        "Per-participant details: nieuwland_format_audit.csv",
    ]
    (FINDINGS / "nieuwland_format_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
