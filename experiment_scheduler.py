"""Prepare (but never submit) Slurm jobs for the mixed-modality BPR matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


AUDIO_BPR_DIR = Path(__file__).resolve().parent
# Typing-only training settings; reading arms keep the original Audio BPR flags.
TYPING_V2_SETTINGS = {"learning_rate": 3e-5, "patience": 20, "min_epochs": 10, "grad_clip": 5.0,
                      "max_eeg_len": 100, "primary_eval_source": "authentic_typing"}
TYPING_V2_SEEDS = (42, 43, 44)
TYPING_FORMATS = ("continuous", "keypress", "word")
IMAGINING_SETTINGS = {"learning_rate": 3e-5, "patience": 20, "min_epochs": 10, "grad_clip": 5.0,
                      "primary_eval_source": "authentic_imagining"}
IMAGINING_TARGETS = ("response",)
IMAGINING_ICT_VARIANTS = ("position", "pace", "random")
READING_ARM_A_CHECKPOINT = (AUDIO_BPR_DIR / "prepared_experiments_20261022_audio_bpr_originalsplit" / "results"
                            / "query_response_plus_question_ict" / "model_mean_dual.pt")
# method -> (EEG format file, extra controller flags)
IMAGINING_METHODS = {
    "window": ("window", ()),
    "crop": ("window", ("--eeg-train-crop-units", "2", "4")),
    "cropavg": ("window", ("--eeg-train-crop-units", "2", "4", "--eeg-test-crop-units", "3",
                            "--eeg-test-crops", "5")),
    "transfer": ("window", ("--init-checkpoint", str(READING_ARM_A_CHECKPOINT))),
    "mil": ("window", ("--eeg-window-units", "2", "--eeg-window-stride", "1")),
    "pace": ("pace", ()),
    "uniform": ("uniform", ()),
    "shuffled": ("shuffled", ()),
}
EXPERIMENTS = (
    ("query_response_only", ("query_response",)),
    ("query_response_plus_question_ict", ("query_response", "question_ict")),
    ("query_response_plus_response_ict", ("query_response", "response_ict")),
    ("all_task_sources", ("query_response", "question_ict", "response_ict")),
    ("all_task_sources_audio", ("query_response", "question_ict", "response_ict", "audio")),
    ("all_task_sources_visual", ("query_response", "question_ict", "response_ict", "visual")),
    ("all_task_sources_audio_visual",
     ("query_response", "question_ict", "response_ict", "audio", "visual")),
    ("query_response_question_ict_audio", ("query_response", "question_ict", "audio")),
    ("query_response_question_ict_visual", ("query_response", "question_ict", "visual")),
    ("query_response_question_ict_audio_visual", ("query_response", "question_ict", "audio", "visual")),
    *((f"typing_{fmt}{suffix}", (f"typing_{fmt}_query_response", f"typing_{fmt}_ict", *extra))
      for fmt in ("continuous", "keypress")
      for suffix, extra in (("", ()), ("_audio", ("audio",)), ("_visual", ("visual",)),
                            ("_audio_visual", ("audio", "visual")))),
    *((f"typing_v2_{fmt}_seed{seed}", (f"typing_{fmt}_query_response", f"typing_{fmt}_question_ict"),
       {**TYPING_V2_SETTINGS, "seed": seed})
      for fmt in TYPING_FORMATS for seed in TYPING_V2_SEEDS),
    *((f"typing_v2_{fmt}{suffix}", (f"typing_{fmt}_query_response", f"typing_{fmt}_question_ict", *extra),
       {**TYPING_V2_SETTINGS, "balance_sources": True})
      for fmt in TYPING_FORMATS
      for suffix, extra in (("_audio", ("audio",)), ("_visual", ("visual",)),
                            ("_audio_visual", ("audio", "visual")))),
    *((f"imagining_{method}_{target}", (f"imagining_{fmt}_{target}",),
       {**IMAGINING_SETTINGS, "extra_args": extra})
      for method, (fmt, extra) in IMAGINING_METHODS.items() for target in IMAGINING_TARGETS),
    *((f"imagining_arm_a_ict_{variant}", ("imagining_window_response", f"imagining_ict_{variant}"),
       IMAGINING_SETTINGS) for variant in IMAGINING_ICT_VARIANTS),
    *((f"imagining_arm_{arm}", ("imagining_window_response", "imagining_ict_position", *extra),
       {**IMAGINING_SETTINGS, "balance_sources": True})
      for arm, extra in (("b_audio", ("audio",)), ("c_visual", ("visual",)),
                         ("d_audio_visual", ("audio", "visual")))),
)
READING_TASK_SOURCES = {"query_response", "question_ict", "response_ict"}


def build_plan(audio_data: Path, visual_data: Path, task_data: Path,
               experiment_names: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    task_dir = Path(task_data).resolve().parent
    expanded_eval_index = str(task_dir / "authentic_task_expanded_eval_index.json")
    sources = {
        "audio": str(Path(audio_data).resolve()),
        "visual": str(Path(visual_data).resolve()),
        "query_response": str(task_dir / "authentic_query_response_aligned_pairs.npy"),
        "question_ict": str(task_dir / "authentic_question_ict_aligned_pairs.npy"),
        "response_ict": str(task_dir / "authentic_response_ict_aligned_pairs.npy"),
        **{f"typing_{fmt}_{kind}": str(task_dir.parent / "authentic_typing_pairs"
                                        / f"authentic_typing_{fmt}_{kind}_pairs.npy")
           for fmt in TYPING_FORMATS for kind in ("query_response", "ict", "question_ict")},
        **{f"imagining_{fmt}_{target}": str(task_dir.parent / "authentic_imagining_pairs"
                                             / f"authentic_imagining_{fmt}_{target}_pairs.npy")
           for fmt in ("window", "pace", "uniform", "shuffled") for target in IMAGINING_TARGETS},
        **{f"imagining_ict_{variant}": str(task_dir.parent / "authentic_imagining_pairs"
                                           / f"authentic_imagining_ict_{variant}_question_pairs.npy")
           for variant in IMAGINING_ICT_VARIANTS},
    }
    selected_names = set(experiment_names) if experiment_names is not None else None
    plan = []
    for name, source_names, *rest in EXPERIMENTS:
        if selected_names is not None and name not in selected_names:
            continue
        plan.append({"name": name, "sources": list(source_names), "split_strategy": "original",
                     "overrides": dict(rest[0]) if rest else {},
                     "expanded_eval_index": (expanded_eval_index
                                             if READING_TASK_SOURCES & set(source_names) else None),
                     "data_paths": [sources[source] for source in source_names]})
    if selected_names is not None and {row["name"] for row in plan} != selected_names:
        raise ValueError(f"Unknown experiment names: {sorted(selected_names - {row['name'] for row in plan})}")
    return plan


def _controller_command(experiment: dict, output_path: Path, python_executable: str,
                        seed: int, epochs: int, patience: int, batch_size: int, learning_rate: float,
                        use_pretrained_text: bool, use_lora: bool,
                        pooling_strategy: str, wandb_project: str) -> list[str]:
    command = [python_executable, str(AUDIO_BPR_DIR / "controller.py")]
    overrides = experiment.get("overrides", {})
    seed = overrides.get("seed", seed)
    patience = overrides.get("patience", patience)
    learning_rate = overrides.get("learning_rate", learning_rate)
    paths = experiment["data_paths"]
    if len(paths) == 1:
        command.extend(["--data_path", paths[0]])
    else:
        command.append("--data_paths")
        command.extend(paths)
    command.extend([
        "--pooling_strategy", pooling_strategy,
        "--encoder_type", "dual",
        "--query_type", "eeg",
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--seed", str(seed),
        "--training_masking_level", "90",
        "--enable_multi_masking_validation",
        "--validation_masking_levels", "0", "25", "50", "75", "90", "100",
        "--primary_masking_level", "90",
        "--output_dir", str(output_path),
        "--experiment-name", experiment["name"],
        "--ablation-sources", *experiment["sources"],
        *(["--expanded-eval-index", experiment["expanded_eval_index"]]
          if experiment["expanded_eval_index"] else []),
        "--wandb-project", wandb_project,
        "--wandb-run-name", f"aqbpr_{experiment['name']}_mean_seed{seed}_originalsplit",
        "--split-strategy", experiment["split_strategy"],
    ])
    if "grad_clip" in overrides:
        command.extend(["--grad-clip", str(overrides["grad_clip"])])
    if "min_epochs" in overrides:
        command.extend(["--min-epochs", str(overrides["min_epochs"])])
    if "max_eeg_len" in overrides:
        command.extend(["--max_eeg_len", str(overrides["max_eeg_len"])])
    if "primary_eval_source" in overrides:
        command.extend(["--primary-eval-source", overrides["primary_eval_source"]])
    if overrides.get("balance_sources") and len(paths) > 1:
        command.append("--balance_sources")
    command.extend(overrides.get("extra_args", ()))
    command.append("--multi-positive-loss")
    if use_pretrained_text:
        command.append("--use_pretrained_text")
    if not use_lora:
        command.append("--no_lora")
    return command


def _slurm_script(experiment: dict, command: list[str], output_dir: Path, partition: str,
                  time_limit: str, memory: str, cpus: int, gpu_request: str,
                  exclude_nodes: list[str] | None = None) -> str:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name=bpr-{experiment['name']}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --gres={gpu_request}",
        *([f"#SBATCH --exclude={','.join(exclude_nodes)}"] if exclude_nodes else []),
        f"#SBATCH --output={shlex.quote(str(log_dir / (experiment['name'] + '-%j.out')))}",
        f"#SBATCH --error={shlex.quote(str(log_dir / (experiment['name'] + '-%j.err')))}",
        "set -euo pipefail",
        f"cd {shlex.quote(str(AUDIO_BPR_DIR))}",
        "export PYTHONUNBUFFERED=1",
        f"{command_text}",
        "",
    ])


def prepare_jobs(audio_data: Path, visual_data: Path, task_data: Path, output_dir: Path,
                 partition: str = "gpu", time_limit: str = "24:00:00", memory: str = "128G",
                 cpus: int = 12, gpu_request: str = "gpu:1", python_executable: str = sys.executable,
                 seed: int = 42, epochs: int = 200, batch_size: int = 32,
                 patience: int = 10, experiment_names: list[str] | tuple[str, ...] | None = None,
                 learning_rate: float = 1e-5, use_pretrained_text: bool = True,
                 use_lora: bool = False, pooling_strategy: str = "mean", validate_inputs: bool = False,
                 overwrite: bool = False, exclude_nodes: list[str] | None = None,
                 wandb_project: str = "Authentic Queries BPR") -> dict:
    task_dir = Path(task_data).resolve().parent
    inputs = {
        "audio": Path(audio_data), "visual": Path(visual_data),
        "task_union": Path(task_data),
        "query_response": task_dir / "authentic_query_response_aligned_pairs.npy",
        "question_ict": task_dir / "authentic_question_ict_aligned_pairs.npy",
        "response_ict": task_dir / "authentic_response_ict_aligned_pairs.npy",
        "expanded_eval_index": task_dir / "authentic_task_expanded_eval_index.json",
        "expanded_eval_index": task_dir / "authentic_task_expanded_eval_index.json",
    }
    if validate_inputs:
        missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))
    expanded_eval_metadata = json.loads(inputs["expanded_eval_index"].read_text(encoding="utf-8"))

    output_dir = Path(output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(inputs["audio"], inputs["visual"], inputs["task_union"], experiment_names)
    if not plan:
        raise ValueError("Select at least one experiment")
    if validate_inputs:
        missing = sorted({path for experiment in plan for path in experiment["data_paths"]
                          if not Path(path).is_file()})
        if missing:
            raise FileNotFoundError("Missing experiment data files:\n" + "\n".join(missing))
    plan_path = output_dir / "experiment_matrix.json"
    if not overwrite and plan_path.exists():
        raise FileExistsError(f"{plan_path} exists; pass --overwrite to regenerate prepared jobs")

    for experiment in plan:
        experiment_output = output_dir / "results" / experiment["name"]
        command = _controller_command(experiment, experiment_output, python_executable, seed,
                          epochs, patience, batch_size, learning_rate, use_pretrained_text, use_lora,
                          pooling_strategy, wandb_project)
        script_path = jobs_dir / f"{experiment['name']}.sbatch"
        script_path.write_text(_slurm_script(experiment, command, output_dir, partition,
                                             time_limit, memory, cpus, gpu_request, exclude_nodes), encoding="utf-8")
        script_path.chmod(0o755)
        experiment["job_script"] = str(script_path)
        experiment["output_dir"] = str(experiment_output)
        experiment["balance_sources"] = True
        experiment["task_pair_sources"] = [source for source in experiment["sources"]
                            if source in {"query_response", "question_ict", "response_ict"}]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "split_policy": "Task: deterministic 80/10/10 by trial (participant/session/part/trial/question), so documents recur across splits but EEG does not; Alice/Nieuwland: existing pair-level split",
        "training_policy": "each arm trains only on its listed sources (balanced); all three Task sources are always loaded so val/test are identical across arms",
        "checkpoint_policy": "early stopping on validation loss over all three Task sources",
        "multi_positive_all_conditions": True,
        "positive_policy": "question-reading EEG (full question or span): its full question and response_seen are positives; response-reading EEG: response_seen",
        "expanded_eval_policy": "All Task test queries ranked together against one fixed index of all aligned full questions and displayed responses, with per-source breakdown and Gaussian-noise baseline",
        "expanded_eval_candidate_count": expanded_eval_metadata.get("candidate_count", 0),
        "pooling_strategy": pooling_strategy,
        "wandb_project": wandb_project,
        "excluded_nodes": exclude_nodes or [],
        "patience": patience,
        "text_encoder_policy": "pretrained ColBERT frozen (LoRA disabled)" if use_pretrained_text and not use_lora
        else "pretrained ColBERT with LoRA" if use_pretrained_text else "text encoder trained from scratch",
        "evaluation_policy": "source-specific held-out candidate pools; primary comparison is authentic_query_response",
        "primary_alignment_policy": "Primary jobs use only aligned_word_index Task EEG; unverified P001/P002 offset pairs remain in full files for separate sensitivity analysis.",
        "resources": {"partition": partition, "time": time_limit, "memory": memory,
                      "cpus_per_task": cpus, "gres": gpu_request},
        "matrix": plan,
    }
    plan_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = output_dir / "PREPARED_NOT_SUBMITTED.txt"
    readme.write_text(
        f"{len(plan)} Slurm job files have been generated under jobs/. No jobs were submitted.\n"
        "Review experiment_matrix.json and the data paths/resources before manually submitting any job.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-data", type=Path, required=True)
    parser.add_argument("--visual-data", type=Path, required=True)
    parser.add_argument("--task-data", type=Path,
                        default=AUDIO_BPR_DIR.parent / "dataset" / "authentic_task_ict_query_response"
                        / "authentic_task_ict_aligned_pairs.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--memory", default="128G")
    parser.add_argument("--cpus", type=int, default=12)
    parser.add_argument("--gpu-request", default="gpu:1")
    parser.add_argument("--exclude-nodes", nargs="+", default=None,
                        help="Slurm nodes to avoid, e.g. gpu04")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--experiments", nargs="+", choices=[name for name, *_ in EXPERIMENTS],
                        default=None, help="Subset of experiment conditions to prepare")
    parser.add_argument("--wandb-project", default="Authentic Queries BPR")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--pooling-strategy", choices=("cls", "mean", "max", "multi"), default="mean")
    parser.add_argument("--no-pretrained-text", action="store_true")
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA adapters; disabled by default for cluster compatibility")
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_jobs(
        args.audio_data, args.visual_data, args.task_data, args.output_dir,
        partition=args.partition, time_limit=args.time, memory=args.memory,
        cpus=args.cpus, gpu_request=args.gpu_request, python_executable=args.python,
        seed=args.seed, epochs=args.epochs, patience=args.patience, experiment_names=args.experiments,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate, use_pretrained_text=not args.no_pretrained_text,
        use_lora=args.use_lora, pooling_strategy=args.pooling_strategy, validate_inputs=args.validate_inputs,
        overwrite=args.overwrite, exclude_nodes=args.exclude_nodes,
        wandb_project=args.wandb_project,
    )
    print(f"Prepared {len(report['matrix'])} Slurm job scripts; no jobs submitted.")
    print(f"Plan: {Path(args.output_dir).resolve() / 'experiment_matrix.json'}")


if __name__ == "__main__":
    main()