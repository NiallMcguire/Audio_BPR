"""EEG foundation-model encoders (LaBraM, CBraMod, REVE, EEGPT, BIOT) for the BPR dual encoder.

Each EEG unit (one fixation, keypress, word or 1 s window) is encoded on its own by the
pretrained model; the unit vectors are then pooled exactly like the MLP encoder's outputs.

FMInputAdapter (dataloader side) turns a source's units into the model's input layout:
channel mapping by electrode name, resampling, a fixed unit length and per-channel z-scoring.
FoundationUnitEncoder (model side) runs the pretrained backbone and returns one vector per unit.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

DEFAULT_MONTAGES = Path(__file__).resolve().parent / "Dataset" / "eeg_source_montages.json"
BIOT_PAIRS = [("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"), ("FP2", "F8"), ("F8", "T8"),
              ("T8", "P8"), ("P8", "O2"), ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
              ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("C3", "A2"), ("C4", "A1")]
FM_SPECS = {
    "labram": {"class": "Labram", "repo": "braindecode/labram-pretrained", "sfreq": 200, "n_times": 200},
    "cbramod": {"class": "CBraMod", "repo": "braindecode/cbramod-pretrained", "sfreq": 200, "n_times": 200},
    "reve": {"class": "REVE", "repo": "brain-bzh/reve-base", "sfreq": 200, "n_times": 200,
             "kwargs": {"on_non_divisible": "pad"}},
    "eegpt": {"class": "EEGPT", "repo": "braindecode/eegpt-pretrained", "sfreq": 256, "n_times": 256},
    "biot": {"class": "BIOT", "repo": "braindecode/biot-pretrained-six-datasets-18chs", "sfreq": 200,
             "n_times": 200},
}
FINETUNE_MODES = ("frozen", "top", "full")
CLIP_SD = 15.0


def load_montages(path=DEFAULT_MONTAGES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def montage_key(dataset_source: str) -> str:
    source = str(dataset_source)
    if source.startswith("authentic_"):
        return "authentic"
    if source in ("alice", "nieuwland"):
        return source
    raise ValueError(f"No montage is defined for dataset_source={source!r}")


def channel_layout(model_name: str, montages: dict) -> list[tuple[str, str | None]]:
    """Input channels for the model as (electrode, reference electrode or None), upper-case names."""
    scalp = [name.upper() for name in montages["authentic"]["scalp_channels"]]
    if model_name == "biot":
        return list(BIOT_PAIRS)
    if model_name == "labram":
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER
        vocabulary = {name.upper() for name in LABRAM_CHANNEL_ORDER}
    elif model_name == "eegpt":
        from braindecode.models.eegpt import EEGPT_CHANNELS
        vocabulary = set(EEGPT_CHANNELS)
    else:
        vocabulary = set(scalp)
    return [(name, None) for name in scalp if name in vocabulary]


class FMInputAdapter:
    """Map one pair's EEG units ([units, time, channels], source layout) to the model's layout."""

    def __init__(self, model_name: str, montages_path=DEFAULT_MONTAGES):
        if model_name not in FM_SPECS:
            raise ValueError(f"Unknown foundation model {model_name!r}")
        self.model_name = model_name
        self.sfreq = FM_SPECS[model_name]["sfreq"]
        self.n_times = FM_SPECS[model_name]["n_times"]
        self.montages = load_montages(montages_path)
        self.layout = channel_layout(model_name, self.montages)
        self.n_channels = len(self.layout)
        self._sources = {}

    def _source(self, key: str) -> dict:
        if key not in self._sources:
            montage = self.montages[key]
            if not montage.get("mapped_names") or not montage.get("sfreq"):
                raise ValueError(f"Source {key!r} has no usable montage "
                                 f"(status: {montage.get('status', 'unknown')}); see eeg_montage_report.txt")
            names = [name.upper() if name else None for name in montage["mapped_names"]]
            index = {name: i for i, name in enumerate(names) if name}
            scalp = [i for i, name in enumerate(names) if name and name not in ("A1", "A2")]
            ratio = Fraction(self.sfreq) / Fraction(montage["sfreq"]).limit_denominator(1000)
            self._sources[key] = {"index": index, "scalp": scalp, "n_source_channels": len(names),
                                  "up": ratio.numerator, "down": ratio.denominator}
        return self._sources[key]

    def __call__(self, units: np.ndarray, dataset_source: str) -> np.ndarray:
        source = self._source(montage_key(dataset_source))
        out = np.zeros((len(units), self.n_times, self.n_channels), dtype=np.float32)
        for u, unit in enumerate(np.asarray(units, dtype=np.float32)):
            unit = unit[:, :source["n_source_channels"]]
            # Units shorter than the file's longest are zero-padded at the end; drop that padding.
            active = np.flatnonzero(np.abs(unit).max(axis=1) > 0)
            if active.size == 0:
                continue
            unit = unit[:active[-1] + 1].astype(np.float64)
            unit = unit - unit[:, source["scalp"]].mean(axis=1, keepdims=True)
            if source["up"] != source["down"]:
                unit = resample_poly(unit, source["up"], source["down"], axis=0)
            unit = unit[:self.n_times]
            for c, (electrode, reference) in enumerate(self.layout):
                if electrode not in source["index"] or (reference and reference not in source["index"]):
                    continue
                signal = unit[:, source["index"][electrode]]
                if reference:
                    signal = signal - unit[:, source["index"][reference]]
                signal = signal - signal.mean()
                std = signal.std()
                if std > 0:
                    out[u, :len(signal), c] = np.clip(signal / std, -CLIP_SD, CLIP_SD)
        return out


def _block_list(module: nn.Module) -> nn.ModuleList | None:
    """The backbone's largest list of repeated transformer blocks."""
    best, best_size = None, 0
    for child in module.modules():
        if isinstance(child, nn.ModuleList) and len(child) >= 2 \
                and len({type(block) for block in child}) == 1:
            size = sum(p.numel() for p in child.parameters())
            if size > best_size:
                best, best_size = child, size
    return best


class FoundationUnitEncoder(nn.Module):
    """Pretrained EEG backbone that maps units [N, time, channels] to vectors [N, hidden_dim]."""

    def __init__(self, model_name: str, hidden_dim: int, finetune: str = "frozen", unfreeze_blocks: int = 2,
                 use_amp: bool = False, chunk_size: int = 1024, montages_path=DEFAULT_MONTAGES):
        super().__init__()
        import braindecode.models as bd_models

        if finetune not in FINETUNE_MODES:
            raise ValueError(f"finetune must be one of {FINETUNE_MODES}, got {finetune!r}")
        spec = FM_SPECS[model_name]
        self.model_name, self.finetune, self.use_amp, self.chunk_size = model_name, finetune, use_amp, chunk_size
        layout = channel_layout(model_name, load_montages(montages_path))
        self.channel_names = [electrode if reference is None else f"{electrode}-{reference}"
                              for electrode, reference in layout]
        kwargs = {"n_chans": len(layout), "n_times": spec["n_times"], "sfreq": spec["sfreq"], "n_outputs": 1,
                  **spec.get("kwargs", {})}
        if model_name in ("labram", "reve", "eegpt"):
            kwargs["chs_info"] = [{"ch_name": name} for name in self.channel_names]
        self.backbone = getattr(bd_models, spec["class"]).from_pretrained(spec["repo"], **kwargs)
        self._forward_kwargs = {"ch_names": self.channel_names} if model_name == "labram" else {}

        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(2, len(layout), spec["n_times"])
            self.feature_dim = self._features(dummy).shape[-1]
        self.projection = nn.Linear(self.feature_dim, hidden_dim)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = finetune == "full"
        if finetune == "top":
            blocks = _block_list(self.backbone)
            if blocks is None:
                raise ValueError(f"Could not find transformer blocks in {model_name}")
            for block in list(blocks)[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for name, child in self.backbone.named_children():
                if name in ("norm", "fc_norm", "proj_out"):
                    for parameter in child.parameters():
                        parameter.requires_grad = True
        self.trainable_backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.finetune == "frozen":
            self.backbone.eval()
        return self

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, return_features=True, **self._forward_kwargs)["features"]
        return features.flatten(1, -2).mean(1) if features.dim() > 2 else features

    def forward(self, units: torch.Tensor) -> torch.Tensor:
        x = units.transpose(1, 2)  # [N, channels, time]
        outputs = []
        for start in range(0, x.size(0), self.chunk_size):
            chunk = x[start:start + self.chunk_size]
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.finetune != "frozen"), \
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp and chunk.is_cuda):
                outputs.append(self._features(chunk).float())
        return self.projection(torch.cat(outputs))
