"""Bayesian search (Optuna/TPE) over independent per-stage ATPT keep fractions.

The existing pruning sweep (scripts/prune_sweep/run_full_sweep.py) applies ONE
--prune-target-keep uniformly, which the model then compounds geometrically
across the 3 ATPT stages (stage_j_target = target ** (j/3) — see
ct_atpt.model.CTATPTConfig.prune_stage_keep_targets / _stage_final_cum_frac).
This script instead searches the 3 stage keep fractions INDEPENDENTLY —
(stage1_keep, stage2_keep, stage3_keep), each in [0.1, 0.9] with the
progressive constraint stage1 >= stage2 >= stage3 — to see whether a
non-uniform pruning schedule beats the best uniform one, using the SAME
AdamW optimizer, dataset split and everything else.

Each trial trains a low-fidelity surrogate (fewer epochs than a full sweep
run) via a subprocess call to scripts/train_ct_atpt_ddp.py with
--prune-keep-stage1/2/3, then reads the resulting best_checkpoint.pt's own
saved validation metrics (not stdout parsing — the checkpoint's "metrics"
dict is the exact same BinaryMetrics computed by the training script itself,
so this can't drift from whatever format the training script prints).

Warm-starts the study with the 9 existing uniform AdamW sweep points from
results/all_prune_metrics.csv, correctly converted to the per-stage targets
those runs actually used (target ** (j/3) per stage — NOT the same fraction
repeated 3x, which would misrepresent what those checkpoints did and would
corrupt the surrogate Optuna fits).

Usage:
    python scripts/bayesian_prune_search.py \
        --splits-dir scripts/prune_sweep/splits_tvt \
        --n-trials 15 --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = PROJECT_ROOT / "scripts" / "prune_sweep"
DEFAULT_SPLITS_DIR = SWEEP_DIR / "splits_tvt"
DEFAULT_UNIFORM_CSV = SWEEP_DIR / "results" / "all_prune_metrics.csv"
DEFAULT_STUDY_DB = PROJECT_ROOT / "results" / "optuna_prune_search.db"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "bayesian_search_summary.csv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "bayesian_search"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_ct_atpt_ddp.py"
NUM_STAGES = 3  # matches the model's fixed 3-stage staging (blocks depth//4, depth//2, 3*depth//4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian (Optuna/TPE) search over per-stage ATPT keep fractions.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--uniform-csv", type=Path, default=DEFAULT_UNIFORM_CSV,
                   help="Existing uniform AdamW sweep results, used to warm-start the study.")
    p.add_argument("--study-db", type=Path, default=DEFAULT_STUDY_DB)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--n-trials", type=int, default=15, help="New trials beyond the warm-start points.")
    p.add_argument("--epochs", type=int, default=20, help="Low-fidelity surrogate epoch count.")
    p.add_argument("--pruning-warmup-epochs", type=int, default=5,
                   help="Scaled down from the full-sweep default (10) to fit the shorter surrogate run.")
    p.add_argument("--prune-ramp-epochs", type=int, default=8,
                   help="Scaled down from the full-sweep default (15) to fit the shorter surrogate run.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--depth", type=int, default=96)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--patch-z", type=int, default=8)
    p.add_argument("--patch-y", type=int, default=16)
    p.add_argument("--patch-x", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--transformer-depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--metric", choices=["accuracy", "roc_auc", "balanced_accuracy", "f1"], default="accuracy",
                   help="Which field of the checkpoint's saved val metrics to optimize.")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def uniform_stage_targets(keep_frac: float) -> tuple[float, float, float]:
    """The per-stage cumulative targets a uniform --prune-target-keep run
    ACTUALLY used, per ct_atpt.model.CTATPT._stage_final_cum_frac's default
    (compounding) formula: target ** (stage_rank / num_stages)."""
    return tuple(keep_frac ** (j / NUM_STAGES) for j in range(1, NUM_STAGES + 1))


def _common_train_flags(args: argparse.Namespace, run_dir: Path, splits_dir_rel: str) -> list[str]:
    return [
        "--train-manifest", f"{splits_dir_rel}/train.csv",
        "--val-manifest", f"{splits_dir_rel}/val.csv",
        "--output-dir", str(run_dir),
        "--pretrained",
        "--optimizer", "adamw",
        "--embed-dim", str(args.embed_dim),
        "--transformer-depth", str(args.transformer_depth),
        "--heads", str(args.heads),
        "--depth", str(args.depth),
        "--height", str(args.height),
        "--width", str(args.width),
        "--patch-z", str(args.patch_z),
        "--patch-y", str(args.patch_y),
        "--patch-x", str(args.patch_x),
        "--pruning-mode", "adaptive",
        "--pruning-warmup-epochs", str(args.pruning_warmup_epochs),
        "--prune-ramp-epochs", str(args.prune_ramp_epochs),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--cls-loss", "ce",
        "--label-smoothing", str(args.label_smoothing),
        "--drop-path", str(args.drop_path),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--det-weight", "0",
        "--amp", args.amp,
        "--tta",
        "--augment",
        "--best-metric", "roc_auc",
    ]


def run_trial_training(
    args: argparse.Namespace, trial_number: int, stage1: float, stage2: float, stage3: float, splits_dir_rel: str,
) -> Path:
    run_dir = args.runs_root / f"trial_{trial_number}"
    cmd = [
        args.python, str(TRAIN_SCRIPT),
        *_common_train_flags(args, run_dir, splits_dir_rel),
        "--prune-keep-stage1", str(stage1),
        "--prune-keep-stage2", str(stage2),
        "--prune-keep-stage3", str(stage3),
    ]
    print(f"\n{'='*70}\nTrial {trial_number}: stage keeps = ({stage1:.3f}, {stage2:.3f}, {stage3:.3f})\n{'='*70}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return run_dir / "best_checkpoint.pt"


def read_checkpoint_metric(checkpoint_path: Path, metric: str) -> float:
    if not checkpoint_path.exists():
        print(f"  WARNING: {checkpoint_path} not found (training likely failed) — scoring 0.0")
        return 0.0
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics")
    if not metrics or metric not in metrics or metrics[metric] is None:
        print(f"  WARNING: checkpoint has no '{metric}' metric — scoring 0.0")
        return 0.0
    return float(metrics[metric])


class SummaryCsvWriter:
    """Appends one row per trial (warm-start or searched) as they complete,
    so progress survives an interrupted search."""

    FIELDS = ["trial_number", "stage1_keep", "stage2_keep", "stage3_keep", "accuracy", "is_warmstart"]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header = not self.path.exists()

    def append(self, trial_number: int, stage1: float, stage2: float, stage3: float, value: float, is_warmstart: bool) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow({
                "trial_number": trial_number,
                "stage1_keep": stage1,
                "stage2_keep": stage2,
                "stage3_keep": stage3,
                "accuracy": value,
                "is_warmstart": is_warmstart,
            })


def load_warmstart_points(uniform_csv: Path, metric: str) -> list[tuple[tuple[float, float, float], float]]:
    if not uniform_csv.exists():
        print(f"No existing uniform sweep CSV at {uniform_csv} — skipping warm-start.")
        return []
    points = []
    with open(uniform_csv) as f:
        for row in csv.DictReader(f):
            if row.get("optimizer") not in (None, "", "adamw"):
                continue  # warm-start only from the AdamW uniform sweep, matching this search's optimizer
            if metric not in row or row[metric] in (None, ""):
                continue
            prune_pct = float(row["prune_percent"])
            keep_frac = 1.0 - prune_pct / 100.0
            stages = uniform_stage_targets(keep_frac)
            points.append((stages, float(row[metric])))
    print(f"Loaded {len(points)} warm-start point(s) from {uniform_csv}")
    return points


def main() -> None:
    args = parse_args()
    try:
        splits_dir_rel = str(args.splits_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        splits_dir_rel = str(args.splits_dir)

    missing = [f for f in ("train.csv", "val.csv") if not (args.splits_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {args.splits_dir}. Run scripts/prune_sweep/run_full_sweep.py "
            "(or make_train_val_test.py) first to build the split this search reuses."
        )

    args.study_db.parent.mkdir(parents=True, exist_ok=True)
    summary = SummaryCsvWriter(args.summary_csv)

    study = optuna.create_study(
        direction="maximize",
        study_name="ct_atpt_prune_search",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{args.study_db}",
        load_if_exists=True,
    )

    # ── Warm-start from the existing uniform AdamW sweep ────────────────
    # The search space is bounded to [0.1, 0.9] per stage, but the uniform
    # sweep's compounding formula (target ** (j/3)) can land slightly outside
    # that band for an extreme prune_percent (e.g. keep_frac=0.8 -> stage1
    # target 0.928). Clip into the search bounds so optuna.trial.create_trial
    # doesn't reject the point; a clipped warm-start is still a much better
    # prior than skipping it.
    STAGE_LO, STAGE_HI = 0.1, 0.9
    warmstart_points = load_warmstart_points(args.uniform_csv, args.metric)
    already_seeded = {t.params.get("stage1_keep") for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE}
    for (s1, s2, s3), value in warmstart_points:
        s1c, s2c, s3c = (min(max(v, STAGE_LO), STAGE_HI) for v in (s1, s2, s3))
        if (s1c, s2c, s3c) != (s1, s2, s3):
            print(f"  warm-start point ({s1:.3f},{s2:.3f},{s3:.3f}) clipped to "
                  f"({s1c:.3f},{s2c:.3f},{s3c:.3f}) to fit the [{STAGE_LO},{STAGE_HI}] search bounds")
        s1, s2, s3 = s1c, s2c, s3c
        if s1 in already_seeded:
            continue  # study.db already has this warm-start point from a previous run of this script
        study.add_trial(
            optuna.trial.create_trial(
                params={"stage1_keep": s1, "stage2_keep": s2, "stage3_keep": s3},
                distributions={
                    "stage1_keep": optuna.distributions.FloatDistribution(0.1, 0.9),
                    "stage2_keep": optuna.distributions.FloatDistribution(0.1, 0.9),
                    "stage3_keep": optuna.distributions.FloatDistribution(0.1, 0.9),
                },
                value=value,
            )
        )
        summary.append(len(study.trials) - 1, s1, s2, s3, value, is_warmstart=True)

    # ── New trials ────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        stage1_keep = trial.suggest_float("stage1_keep", 0.1, 0.9)
        stage2_keep = trial.suggest_float("stage2_keep", 0.1, 0.9)
        stage3_keep = trial.suggest_float("stage3_keep", 0.1, 0.9)

        # Progressive-pruning constraint: prune more in later stages. Violating
        # combos are scored 0.0 without training (Optuna's TPE learns to avoid
        # that region of the search space from the score alone).
        if not (stage1_keep >= stage2_keep >= stage3_keep):
            value = 0.0
        else:
            try:
                ckpt_path = run_trial_training(args, trial.number, stage1_keep, stage2_keep, stage3_keep, splits_dir_rel)
                value = read_checkpoint_metric(ckpt_path, args.metric)
            except subprocess.CalledProcessError as e:
                print(f"  Trial {trial.number} training failed (exit {e.returncode}) — scoring 0.0")
                value = 0.0

        summary.append(trial.number, stage1_keep, stage2_keep, stage3_keep, value, is_warmstart=False)
        return value

    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n{'='*70}\nBest trial\n{'='*70}")
    print(f"  {args.metric}: {study.best_trial.value:.4f}")
    print(f"  Stage 1 keep: {study.best_trial.params['stage1_keep']:.3f}")
    print(f"  Stage 2 keep: {study.best_trial.params['stage2_keep']:.3f}")
    print(f"  Stage 3 keep: {study.best_trial.params['stage3_keep']:.3f}")
    print(f"\nStudy DB: {args.study_db}")
    print(f"Summary CSV: {args.summary_csv}")
    print("\nNext: python scripts/retrain_best_config.py")


if __name__ == "__main__":
    main()
