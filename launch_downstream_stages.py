"""Select FM configurations by validation loss, then launch Stage 1b and FM Stage 2."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE1A = ROOT / "prepared_experiments_20261117_stage1a_selected"
SEEDS = (42, 43, 44)
MODELS = ("eegpt", "labram", "reve")
FINETUNES = ("frozen", "top", "full")
MODALITIES = ("reading", "typing_continuous", "imagining")
LOSS = re.compile(r"Best validation loss:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")


def best_validation_loss(log: Path) -> float | None:
    matches = LOSS.findall(log.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def select_best() -> dict:
    selected = {}
    diagnostics = {}
    log_dirs = (STAGE1A / "logs", STAGE1A / STAGE1A.name / "logs")
    for modality in MODALITIES:
        candidates = []
        for model in MODELS:
            for finetune in FINETUNES:
                losses = []
                for seed in SEEDS:
                    pattern = f"s1a_{model}_{finetune}_{modality}_seed{seed}-*.out"
                    logs = [path for log_dir in log_dirs for path in log_dir.glob(pattern)]
                    if len(logs) == 1:
                        loss = best_validation_loss(logs[0])
                        if loss is not None:
                            losses.append(loss)
                if len(losses) >= 2:
                    candidates.append((sum(losses) / len(losses), model, finetune, losses))
        if not candidates:
            raise RuntimeError(f"No configuration has validation losses for at least two seeds: {modality}")
        mean_loss, model, finetune, losses = min(candidates)
        selected[modality] = f"{model}:{finetune}"
        diagnostics[modality] = {
            "model": model, "finetune": finetune, "mean_best_validation_loss": mean_loss,
            "seed_losses": losses, "successful_seeds": len(losses),
        }
    return {"selection": selected, "diagnostics": diagnostics}


def prepare_and_submit(stage: str, selection: dict, output_name: str, only_fm: bool = False) -> list[str]:
    output = ROOT / output_name
    command = [sys.executable, str(ROOT / "stage_scheduler.py"), stage, "--best"]
    command += [f"{modality}={model}" for modality, model in selection.items()]
    command += ["--date", "20261117", "--output-dir", str(output)]
    if only_fm:
        command.append("--only-fm")
    subprocess.run(command, cwd=ROOT, check=True)
    submitted = []
    for script in sorted((output / "jobs").glob("*.sbatch")):
        result = subprocess.run(["sbatch", str(script)], cwd=ROOT, text=True, capture_output=True, check=True)
        submitted.append(result.stdout.strip())
    (output / "SUBMITTED_JOBS.txt").write_text("\n".join(submitted) + "\n")
    return submitted


def main() -> None:
    result = select_best()
    selection_file = STAGE1A / "downstream_selection.json"
    selection_file.write_text(json.dumps(result, indent=2) + "\n")
    selection = result["selection"]
    stage1b = prepare_and_submit("stage1b", selection, "prepared_experiments_20261117_stage1b_fm")
    stage2 = prepare_and_submit("stage2", selection, "prepared_experiments_20261117_stage2_fm", only_fm=True)
    print(json.dumps({"selected": result, "stage1b_jobs": len(stage1b), "stage2_fm_jobs": len(stage2)}, indent=2))


if __name__ == "__main__":
    main()