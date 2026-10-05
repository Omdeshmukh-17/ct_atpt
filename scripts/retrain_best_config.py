"""Retrain the best (stage1_keep, stage2_keep, stage3_keep) config found by
scripts/bayesian_prune_search.py at FULL epochs, then evaluate it exactly
like a normal sweep run.

Reads the Optuna study's best trial directly from its sqlite storage (the
source of truth — more reliable than re-parsing the summary CSV, since the
CSV also contains any progressive-pruning-constraint violations logged at
accuracy 0.0). Retrains with the SAME AdamW hyperparameters as the existing
sweep's full runs, then calls scripts/prune_sweep/eval_and_save_metrics.py
so the output format (metrics CSV row, ROC curve PNG, raw-score .npz) is
byte-for-byte the same shape as every other sweep run, just filed under
results/bayesian_optimal/ instead of results/.

Usage:
    python scripts/retrain_best_config.py --epochs 45
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = PROJECT_ROOT / "scripts" / "prune_sweep"
DEFAULT_SPLITS_DIR = SWEEP_DIR / "splits_tvt"
DEFAULT_STUDY_DB = PROJECT_ROOT / "results" / "optuna_prune_search_v2.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "bayesian_optimal"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "bayesian_optimal"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_ct_atpt_ddp.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "prune_sweep" / "eval_and_save_metrics.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain the Bayesian search's best per-stage config at full epochs.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--study-db", type=Path, default=DEFAULT_STUDY_DB)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=45, help="Full epoch count (matches run_full_sweep.py's default).")
    p.add_argument("--pruning-warmup-epochs", type=int, default=10)
    p.add_argument("--prune-ramp-epochs", type=int, default=15)
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
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def load_best_config(study_db: Path) -> dict:
    if not study_db.exists():
        raise FileNotFoundError(f"No study database at {study_db} — run scripts/bayesian_prune_search.py first.")
    study = optuna.load_study(study_name="ct_atpt_prune_search", storage=f"sqlite:///{study_db}")

    def score(t):
        if t.value is not None:
            return float(t.value)
        return float(t.intermediate_values[max(t.intermediate_values)]) if t.intermediate_values else None

    # Warm-start rows come from the full 45-epoch uniform sweep, a different fidelity than the short
    # search trials, so they would always win. Pick the best config the search itself trained.
    ok = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    searched = [t for t in study.trials if t.user_attrs.get("searched") and t.state in ok and score(t) is not None]
    if not searched:
        raise RuntimeError("No trained search trial found in the study - run scripts/bayesian_prune_search.py first.")
    best = max(searched, key=score)
    print(f"Best searched trial #{best.number}: value={score(best):.4f} params={best.params}")
    return best.params


def main() -> None:
    args = parse_args()
    try:
        splits_dir_rel = str(args.splits_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        splits_dir_rel = str(args.splits_dir)

    for f in ("train.csv", "val.csv", "test.csv"):
        if not (args.splits_dir / f).exists():
            raise FileNotFoundError(f"Missing {args.splits_dir / f} — this needs the same split the search used.")

    best_params = load_best_config(args.study_db)
    stage1, stage2, stage3 = best_params["stage1_keep"], best_params["stage2_keep"], best_params["stage3_keep"]

    train_cmd = [
        args.python, str(TRAIN_SCRIPT),
        "--train-manifest", f"{splits_dir_rel}/train.csv",
        "--val-manifest", f"{splits_dir_rel}/val.csv",
        "--output-dir", str(args.output_dir),
        "--pretrained",
        "--optimizer", "adamw",
        "--prune-keep-stage1", str(stage1),
        "--prune-keep-stage2", str(stage2),
        "--prune-keep-stage3", str(stage3),
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
    print("\n=== Retraining best config at full epochs ===")
    print(" ".join(train_cmd))
    subprocess.run(train_cmd, check=True, cwd=str(PROJECT_ROOT))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        args.python, str(EVAL_SCRIPT),
        "--checkpoint", str(args.output_dir / "best_checkpoint.pt"),
        "--val-manifest", f"{splits_dir_rel}/test.csv",
        "--prune-percent", "0",  # not part of the uniform sweep; stage config is what matters here
        "--optimizer-name", "adamw_bayesian_optimal",
        "--csv-path", str(args.results_dir / "metrics.csv"),
        "--plots-dir", str(args.results_dir / "plots"),
        "--scores-dir", str(args.results_dir / "scores"),
        "--amp", "bf16",
        "--tta",
    ]
    print("\n=== Evaluating on the held-out test set ===")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True, cwd=str(PROJECT_ROOT))

    print(f"\nDone. Results under {args.results_dir}")
    print(f"  stage1_keep={stage1:.3f} stage2_keep={stage2:.3f} stage3_keep={stage3:.3f}")


if __name__ == "__main__":
    main()
