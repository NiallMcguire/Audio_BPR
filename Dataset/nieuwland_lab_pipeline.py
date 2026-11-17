"""Download one Nieuwland lab's continuous recordings from OSF, build per-participant word EEG, delete the raw files.

    python nieuwland_lab_pipeline.py index                # list OSF files -> ~/nieuwland_raw/osf_index.json
    python nieuwland_lab_pipeline.py lab BRIS             # download + words for one lab
    python nieuwland_lab_pipeline.py pairs [max_pairs] [name]  # all labs -> {name}_ict_pairs_*.npy
"""

from __future__ import annotations

import json
import http.client
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from nieuwland_continuous_reader import build_pairs, build_words

ROOT = Path.home() / "nieuwland_raw"
WORDS = ROOT / "words"
OUTPUT = Path.home() / "BPR_mean_pooling_inputs"
OSF_DATA = "https://api.osf.io/v2/nodes/eyzaq/files/osfstorage/5abe45e6a2409b00119f8869/"
LABS = ("BRIS", "KENT", "EDIN", "LOND", "STIR", "GLAS", "OXFO", "YORK", "BIRM")
# Files not needed to rebuild the main-experiment word EEG.
SKIP = re.compile(r"(control|E-Prime|EPrime|Extra\.zip|\.rar$|\.ehst2$|\.edat2$|\.log$|\.txt$|\.fst$|\.ftg$"
                  r"|preproc_segments/|\.xml$|\.es2$|\.ebs2$|\.wndpos$|\.hfinf2$)", re.I)


def _listing(url: str) -> list[dict]:
    items = []
    while url:
        page = json.load(urllib.request.urlopen(url, timeout=120))
        items += page["data"]
        url = page["links"].get("next")
    return items


def build_index() -> dict:
    labs = {}
    raw = next(n for n in _listing(OSF_DATA) if n["attributes"]["name"] == "Raw data")
    for lab in _listing(raw["relationships"]["files"]["links"]["related"]["href"]):
        files = {}
        stack = [("", lab["relationships"]["files"]["links"]["related"]["href"])]
        while stack:
            prefix, url = stack.pop()
            for node in _listing(url):
                name = prefix + node["attributes"]["name"]
                if node["attributes"]["kind"] == "folder":
                    stack.append((name + "/", node["relationships"]["files"]["links"]["related"]["href"]))
                else:
                    files[name] = [node["links"]["download"], node["attributes"]["size"]]
        labs[lab["attributes"]["name"]] = files
    (ROOT / "osf_index.json").write_text(json.dumps(labs, indent=1))
    return labs


def wanted(lab: str, name: str) -> bool:
    if SKIP.search(name):
        return False
    if lab == "OXFO":  # the 2048 Hz .bdf files are 35 GB; the resampled BrainVision copies hold the same data
        return name.startswith("resampled/")
    return True


def _download(url: str, target: Path, size: int) -> None:
    for attempt in range(8):
        offset = target.stat().st_size if target.exists() else 0
        if offset == size:
            return
        if offset > size:
            target.unlink()
            offset = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content_range = response.headers.get("Content-Range", "")
                resume = offset > 0 and response.status == 206
                if resume and not content_range.startswith(f"bytes {offset}-"):
                    raise OSError(f"Unexpected Content-Range: {content_range}")
                with target.open("ab" if resume else "wb") as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
            actual = target.stat().st_size
            if actual == size:
                return
            if actual > size:
                target.unlink()
                raise OSError(f"Downloaded {actual} bytes, expected {size}")
            raise EOFError(f"Downloaded {actual} of {size} bytes")
        except (OSError, EOFError, TimeoutError, urllib.error.URLError, http.client.HTTPException):
            if attempt == 7:
                raise
            time.sleep(min(attempt + 1, 10))


def download_lab(lab: str, index: dict) -> Path:
    folder = ROOT / lab
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, size) in sorted(index[lab].items()):
        if not wanted(lab, name):
            continue
        target = folder / Path(name).name
        if target.exists() and target.stat().st_size == size:
            continue
        _download(url, target, size)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as archive:
                for member in archive.namelist():
                    if member.lower().endswith((".vhdr", ".vmrk")):
                        (folder / Path(member).name).write_bytes(archive.read(member))
    return folder


def recordings(folder: Path) -> list[Path]:
    found = []
    for header in sorted(folder.glob("*.vhdr")):
        text = header.read_text(encoding="utf-8", errors="ignore")
        data = re.search(r"^DataFile=(.+)$", text, re.M)
        marker = re.search(r"^MarkerFile=(.+)$", text, re.M)
        if data and (folder / data[1].strip()).exists() and marker and (folder / marker[1].strip()).exists():
            found.append(header)
    found += sorted(folder.glob("*.bdf")) + sorted(folder.glob("*.cnt"))
    return found


def run_lab(lab: str) -> None:
    index = json.loads((ROOT / "osf_index.json").read_text())
    folder = download_lab(lab, index)
    WORDS.mkdir(exist_ok=True)
    reports = []
    for recording in recordings(folder):
        participant = f"{lab.lower()}_{re.sub(r'[^A-Za-z0-9]+', '', recording.stem)}"
        output = WORDS / f"{participant}.npz"
        if output.exists():
            continue
        try:
            report = build_words(recording, output)
        except Exception as error:  # one unreadable recording must not stop the lab
            report = {"recording": str(recording), "status": f"error: {error}"}
            output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(participant, report.get("status"), report.get("sentences"), report.get("channels_found"), flush=True)
    (ROOT / f"{lab}_report.json").write_text(json.dumps(reports, indent=2))
    shutil.rmtree(folder)


def make_pairs(max_pairs: int | None, name: str = "nieuwland_continuous") -> None:
    OUTPUT.mkdir(exist_ok=True)
    full = OUTPUT / f"{name}_ict_pairs_all.npy"
    print(build_pairs(WORDS, full))
    if max_pairs:
        import numpy as np
        import random
        store = np.load(full, allow_pickle=True).item()
        pairs = store["ict_pairs"]
        keep = sorted(random.Random(42).sample(range(len(pairs)), min(max_pairs, len(pairs))))
        metadata = {**store["metadata"], "generated_ict_pairs": len(keep),
                    "subsample": f"{len(keep)} of {len(pairs)} pairs, random.Random(42), to match the earlier "
                                 f"Nieuwland set size"}
        target = OUTPUT / f"{name}_ict_pairs_RUNTIME_MASKING.npy"
        np.save(target, {"ict_pairs": [pairs[i] for i in keep], "metadata": metadata}, allow_pickle=True)
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print(metadata)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "index":
        print({lab: len(files) for lab, files in build_index().items()})
    elif command == "lab":
        run_lab(sys.argv[2])
    elif command == "pairs":
        make_pairs(int(sys.argv[2]) if len(sys.argv) > 2 else None, *sys.argv[3:4])
