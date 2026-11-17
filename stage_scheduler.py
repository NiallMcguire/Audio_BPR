"""Prepare (never submit) Slurm jobs for the next stages of the plan.

setup   MLP baselines with the fixed task split and real best-epoch checkpoints: arm A per modality,
        plus + Alice, + Nieuwland and + both (balanced), 3 seeds.
stage1a Foundation models (LaBraM, CBraMod, REVE, EEGPT, BIOT) x fine-tuning level x modality (arm A).
        --pilot keeps reading and seed 42 only (the "test on one modality first" step).
stage1b The chosen foundation-model setup per modality + Alice, + Nieuwland and + both (balanced).
stage2  EEG pooling options x encoder x modality (arm A).

Example:
    python stage_scheduler.py setup
    python stage_scheduler.py stage1a --pilot
    python stage_scheduler.py stage1b --best reading=reve:frozen imagining=labram:top
    python stage_scheduler.py stage2 --best reading=reve:frozen
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiment_scheduler import (AUDIO_BPR_DIR, IMAGINING_SETTINGS, TYPING_V2_SETTINGS, _controller_command,
                                  _slurm_script, source_paths)
from fm_encoders import FINETUNE_MODES, FM_SPECS

DATA_ROOT = AUDIO_BPR_DIR.parent / "dataset"
TASK_DATA = DATA_ROOT / "authentic_task_ict_query_response" / "authentic_task_ict_aligned_pairs.npy"
EXPANDED_EVAL_INDEX = DATA_ROOT / "authentic_task_ict_query_response" / "authentic_task_expanded_eval_index.json"
EXTERNAL_DIR = Path.home() / "BPR_mean_pooling_inputs"
AUDIO_DATA = EXTERNAL_DIR / "alice_ict_pairs_RUNTIME_MASKING.npy"
VISUAL_DATA = EXTERNAL_DIR / "nieuwland_all_labs_ict_pairs_RUNTIME_MASKING.npy"
FM_PYTHON = Path.home() / "envs" / "bpr_fm" / "bin" / "python"
SEEDS = (42, 43, 44)
READING_SETTINGS = {"learning_rate": 1e-5, "patience": 5, "primary_eval_source": "authentic_query"}
MODALITIES = {
    "reading": (("query_response", "question_ict"), READING_SETTINGS),
    **{f"typing_{fmt}": ((f"typing_{fmt}_query_response", f"typing_{fmt}_question_ict"), TYPING_V2_SETTINGS)
       for fmt in ("continuous", "keypress")},
    "imagining": (("imagining_window_response", "imagining_ict_position"), IMAGINING_SETTINGS),
}
EXTERNAL = {"": (), "_audio": ("audio",), "_visual": ("visual",), "_audio_visual": ("audio", "visual")}
POOLING = {"mean": ("mean", ()), "max": ("max", ()), "cls": ("cls", ()), "attention": ("attention", ()),
           "mil": ("mean", ("--eeg-window-units", "2", "--eeg-window-stride", "1"))}
COMMON_ARGS = ("--best-checkpoint-copy",)
ENVIRONMENT = {"HF_HUB_OFFLINE": 1, "TRANSFORMERS_OFFLINE": 1, "TOKENIZERS_PARALLELISM": "false"}


def experiment(name: str, modality: str, external: tuple, seed: int, encoder_args: tuple = (),
               paths: dict | None = None) -> dict:
    task_sources, settings = MODALITIES[modality]
    sources = (*task_sources, *external)
    overrides = {**settings, "seed": seed, "extra_args": (*COMMON_ARGS, *encoder_args)}
    if external:
        overrides["balance_sources"] = True
    return {"name": name, "modality": modality, "sources": list(sources), "split_strategy": "original_fixed",
            "overrides": overrides, "data_paths": [paths[source] for source in sources],
            "expanded_eval_index": str(EXPANDED_EVAL_INDEX) if modality == "reading" else None}


def fm_args(model: str, finetune: str) -> tuple:
    return ("--eeg_arch", model, "--fm-finetune", finetune, "--fm-amp")


def parse_best(entries: list[str]) -> dict:
    """'reading=reve:frozen' -> {'reading': ('reve', 'frozen')}."""
    best = {}
    for entry in entries or ():
        modality, _, setup = entry.partition("=")
        model, _, finetune = setup.partition(":")
        if modality not in MODALITIES or model not in FM_SPECS or finetune not in FINETUNE_MODES:
            raise ValueError(f"Bad --best entry {entry!r}; use modality=model:finetune")
        best[modality] = (model, finetune)
    return best


def build(stage: str, args) -> tuple[list[dict], list[str]]:
    paths = source_paths(AUDIO_DATA, VISUAL_DATA, TASK_DATA)
    seeds = (42,) if getattr(args, "pilot", False) else SEEDS
    typing = f"typing_{args.typing_format}"
    plan, notes = [], []
    if stage == "setup":
        for modality in MODALITIES:
            for suffix, external in EXTERNAL.items():
                for seed in seeds:
                    plan.append(experiment(f"setup_{modality}{suffix}_seed{seed}", modality, external, seed,
                                           paths=paths))
    elif stage == "stage1a":
        modalities = ("reading",) if args.pilot else ("reading", typing, "imagining")
        for model in args.models:
            for finetune in FINETUNE_MODES:
                for modality in modalities:
                    for seed in seeds:
                        plan.append(experiment(f"s1a_{model}_{finetune}_{modality}_seed{seed}", modality, (), seed,
                                               fm_args(model, finetune), paths))
    elif stage == "stage1b":
        for modality, (model, finetune) in parse_best(args.best).items():
            for suffix, external in list(EXTERNAL.items())[1:]:
                for seed in seeds:
                    plan.append(experiment(f"s1b_{model}_{finetune}_{modality}{suffix}_seed{seed}", modality,
                                           external, seed, fm_args(model, finetune), paths))
    elif stage == "stage2":
        encoders = {modality: ("mlp", ()) for modality in ("reading", typing, "imagining")}
        fm = {modality: (f"{model}-{finetune}", fm_args(model, finetune))
              for modality, (model, finetune) in parse_best(args.best).items()}
        for modality in encoders:
            for encoder, encoder_args in [encoders[modality], *([fm[modality]] if modality in fm else [])]:
                if args.only_fm and encoder == "mlp":
                    continue
                for pooling, (eeg_pooling, pooling_args) in POOLING.items():
                    for seed in seeds:
                        plan.append(experiment(
                            f"s2_{encoder}_{pooling}_{modality}_seed{seed}", modality, (), seed,
                            (*encoder_args, "--eeg-pooling", eeg_pooling, *pooling_args), paths))
    else:
        raise ValueError(stage)
    return plan, notes


def resources(item: dict) -> tuple[str, str]:
    """(memory, time) for one job."""
    fm = "--eeg_arch" in item["overrides"]["extra_args"]
    memory = "256G" if {"audio", "visual"} <= set(item["sources"]) else "128G"
    external = {"audio", "visual"} & set(item["sources"])
    time = "2-00:00:00" if fm or external else "1-00:00:00"
    return memory, time


def prepare(stage: str, args) -> Path:
    plan, notes = build(stage, args)
    if not plan:
        raise ValueError("Nothing to prepare")
    output_dir = (Path(args.output_dir) if args.output_dir else
                  AUDIO_BPR_DIR / f"prepared_experiments_{args.date}_{stage}{'_pilot' if args.pilot else ''}")
    plan_path = output_dir / "experiment_matrix.json"
    if plan_path.exists() and not args.overwrite:
        raise FileExistsError(f"{plan_path} exists; pass --overwrite to regenerate")
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted({path for item in plan for path in item["data_paths"] if not Path(path).is_file()})
    if missing:
        raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))
    for item in plan:
        result_dir = output_dir / "results" / item["name"]
        command = _controller_command(item, result_dir, str(args.python), 42, 200, 10, 32, 1e-5,
                                      True, False, "mean", args.wandb_project)
        memory, time_limit = resources(item)
        script = jobs_dir / f"{item['name']}.sbatch"
        script.write_text(_slurm_script(item, command, output_dir, "gpu", time_limit, memory, 12, "gpu:1",
                                        args.exclude_nodes, ENVIRONMENT), encoding="utf-8")
        script.chmod(0o755)
        item.update({"job_script": str(script), "output_dir": str(result_dir), "memory": memory,
                     "time": time_limit})
    plan_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "submitted": False,
        "split_policy": "original Audio BPR split (seed 42, 80% train, test = second half of the 20%) applied "
                        "to each source separately, so the task test set is identical in every arm",
        "checkpoint_policy": "early stopping on the task validation loss; the best epoch's weights are tested",
        "text_encoder_policy": "pretrained ColBERT frozen (LoRA disabled)",
        "python": str(args.python), "notes": notes, "matrix": plan,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "PREPARED_NOT_SUBMITTED.txt").write_text(
        f"{len(plan)} Slurm job files are in jobs/. No jobs were submitted.\n"
        + "".join(f"Note: {note}\n" for note in notes)
        + "Submit all with:\n  for f in jobs/*.sbatch; do sbatch \"$f\"; done\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("setup", "stage1a", "stage1b", "stage2"))
    parser.add_argument("--pilot", action="store_true", help="stage1a: reading only, seed 42")
    parser.add_argument("--models", nargs="+", choices=tuple(FM_SPECS), default=tuple(FM_SPECS))
    parser.add_argument("--best", nargs="+", default=None, help="modality=model:finetune (stage1b, stage2)")
    parser.add_argument("--only-fm", action="store_true", help="stage2: skip the MLP pooling jobs")
    parser.add_argument("--typing-format", choices=("continuous", "keypress"), default="continuous",
                        help="Typing format carried into stage 1 and 2 (choose from the setup results)")
    parser.add_argument("--python", type=Path, default=FM_PYTHON)
    parser.add_argument("--exclude-nodes", nargs="+", default=["gpu04", "gpu05"])
    parser.add_argument("--wandb-project", default="Authentic Queries BPR")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "stage1b" and not args.best:
        parser.error("stage1b needs --best")
    output_dir = prepare(args.stage, args)
    print(f"Prepared jobs in {output_dir}; nothing was submitted.")


if __name__ == "__main__":
    main()
