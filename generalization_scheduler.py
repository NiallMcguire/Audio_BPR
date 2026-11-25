"""Prepare paired 10-fold subject- and query-generalization runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiment_scheduler import AUDIO_BPR_DIR, _controller_command, _slurm_script, source_paths
from stage_scheduler import AUDIO_DATA, FM_PYTHON, MODALITIES, TASK_DATA, VISUAL_DATA

FOLDS = range(1, 11)
SEEDS = (42, 43, 44)
EXCLUDE_NODES = ("gpu04", "gpu05")
BEST_MODELS = {
    "reading": ("eegpt", "full"),
    "typing_continuous": ("reve", "full"),
    "imagining": ("eegpt", "full"),
}
SPLITS = {
    "subject_10fold": "Participants are divided into 10 balanced groups; fold i is test, fold i+1 is validation, "
                      "and the other eight groups are training.",
    "query_10fold": "Shared question_document_id groups (falling back to normalized question_text) are divided "
                    "into 10 balanced groups; fold i is test, fold i+1 is validation, and the other eight are training.",
}


def build_plan(output_dir: Path, selected_modalities=None) -> list[dict]:
    paths = source_paths(AUDIO_DATA, VISUAL_DATA, TASK_DATA)
    selected_modalities = set(selected_modalities or BEST_MODELS)
    unknown = selected_modalities - set(BEST_MODELS)
    if unknown:
        raise ValueError(f"Unsupported modalities: {sorted(unknown)}")
    plan = []
    for split_strategy in SPLITS:
        for modality, (source_names, settings) in MODALITIES.items():
            if modality not in selected_modalities:
                continue
            fm_model, fm_tuning = BEST_MODELS[modality]
            mixed_sources = (*source_names, "audio", "visual")
            encoders = (
                ("transformer", ("--best-checkpoint-copy", "--eeg_arch", "transformer"), False),
                (f"{fm_model}_{fm_tuning}",
                 ("--best-checkpoint-copy", "--eeg_arch", fm_model, "--fm-finetune", fm_tuning, "--fm-amp"),
                 True),
            )
            for encoder_name, encoder_args, is_fm in encoders:
                for fold in FOLDS:
                    for seed in SEEDS:
                        name = f"gen_{split_strategy}_{encoder_name}_{modality}_fold{fold:02d}_seed{seed}"
                        item = {
                            "name": name,
                            "modality": modality,
                            "encoder": encoder_name,
                            "sources": list(mixed_sources),
                            "data_paths": [paths[source] for source in mixed_sources],
                            "split_strategy": split_strategy,
                            "cv_fold": fold,
                            "expanded_eval_index": None,
                            "overrides": {
                                **settings,
                                "seed": seed,
                                "split_seed": 42,
                                "balance_sources": True,
                                "extra_args": encoder_args,
                            },
                        }
                        result_dir = output_dir / "results" / name
                        command = _controller_command(
                            item, result_dir, str(FM_PYTHON), seed, 200,
                            settings["patience"], 32, settings["learning_rate"],
                            True, False, "mean", "Authentic Queries BPR",
                        )
                        command.extend(("--cv-fold", str(fold)))
                        memory, time_limit = "256G", "2-00:00:00"
                        item.update({"output_dir": str(result_dir), "memory": memory, "time": time_limit})
                        plan.append(item)
    return plan


def prepare(output_dir: Path, selected_modalities=None) -> Path:
    output_dir = output_dir.resolve()
    matrix_path = output_dir / "experiment_matrix.json"
    if matrix_path.exists():
        raise FileExistsError(f"{matrix_path} exists")
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    selected_modalities = tuple(selected_modalities or BEST_MODELS)
    plan = build_plan(output_dir, selected_modalities)
    by_name = {item["name"]: item for item in plan}
    for item in plan:
        command = _controller_command(
            item, Path(item["output_dir"]), str(FM_PYTHON), item["overrides"]["seed"], 200,
            item["overrides"]["patience"], 32, item["overrides"]["learning_rate"],
            True, False, "mean", "Authentic Queries BPR",
        )
        command.extend(("--cv-fold", str(item["cv_fold"])))
        script = jobs_dir / f"{item['name']}.sbatch"
        is_fm = item["encoder"] != "transformer"
        script.write_text(
            _slurm_script(item, command, output_dir, "gpu", item["time"], item["memory"], 12,
                          "gpu:1", list(EXCLUDE_NODES),
                          {"HF_HUB_OFFLINE": 1, "TRANSFORMERS_OFFLINE": 1,
                           "TOKENIZERS_PARALLELISM": "false"}),
            encoding="utf-8",
        )
        script.chmod(0o755)
        item["job_script"] = str(script)
    expected_count = len(SPLITS) * len(selected_modalities) * 2 * len(FOLDS) * len(SEEDS)
    assert len(by_name) == len(plan) == expected_count
    matrix_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "folds": 10,
        "seeds": SEEDS,
        "best_foundation_models": BEST_MODELS,
        "split_policies": SPLITS,
        "baseline": "Audio BPR Transformer (eeg_arch=transformer), trained from scratch",
        "foundation_models": "EEGPT-full for reading and imagining; REVE-full for continuous typing",
        "selected_modalities": selected_modalities,
        "external_training_data": True,
        "data_sources": ["task", "Alice", "Nieuwland all-labs (332 participants)"],
        "source_balancing": "Equal-probability source sampling for training; all sources appear in each split.",
        "candidate_index": "Test-fold documents from task, Alice and Nieuwland sources are ranked together.",
        "metrics": "All-source and per-source test ranking at 0/25/50/75/90/100% masking; Gaussian-noise metrics at every level",
        "matrix": plan,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "PREPARED_NOT_SUBMITTED.txt").write_text(
        f"{len(plan)} job scripts are in jobs/.\n"
        "Submit with: for f in jobs/*.sbatch; do sbatch \"$f\"; done\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modalities", nargs="+", choices=tuple(BEST_MODELS), default=None,
                        help="Modalities to run; defaults to reading, continuous typing and imagining")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    output = args.output_dir or AUDIO_BPR_DIR / f"prepared_experiments_{args.date}_generalization"
    print(f"Prepared {len(build_plan(output, args.modalities))} runs in {output}; nothing was submitted.")
    prepare(output, args.modalities)


if __name__ == "__main__":
    main()