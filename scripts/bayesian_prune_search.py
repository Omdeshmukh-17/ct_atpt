"""Bayesian search (Optuna/TPE + Hyperband) over independent per-stage ATPT keep fractions.

Each trial trains a cheap surrogate IN-PROCESS (no subprocess, so no GPU memory
piling up across trials) for a few epochs at half the normal batch size, reports
validation accuracy after every epoch, and lets Optuna's Hyperband pruner kill
unpromising configs at epoch 5 (--early-stop-epoch). Warm-started from the 9 uniform AdamW sweep
points. Everything lives in an sqlite study, so a crash/restart resumes where it
stopped and only runs the trials still missing.

Search space (categorical, progressive constraint stage1 >= stage2 >= stage3):
    stage1_keep in {0.3, 0.5, 0.7, 0.9}
    stage2_keep in {0.2, 0.4, 0.6, 0.8}
    stage3_keep in {0.1, 0.3, 0.5, 0.7}
Only 20 of the 64 combinations satisfy the constraint. A violating combo is
scored 0.0 without training (free), and a combo already trained in this study
reuses its stored score instead of retraining, so "--n-trials 8" means 8 configs
that were actually trained.

Usage (Colab: keep the DB local, back it up to Drive after every trial):
    python scripts/bayesian_prune_search.py --splits-dir scripts/prune_sweep/splits_tvt \
        --backup-dir /content/drive/MyDrive/ct_atpt_bo
"""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch

import scripts.train_ct_atpt_ddp as train_script

SWEEP_DIR = PROJECT_ROOT / "scripts" / "prune_sweep"
DEFAULT_SPLITS_DIR = SWEEP_DIR / "splits_tvt"
DEFAULT_UNIFORM_CSV = SWEEP_DIR / "results" / "all_prune_metrics.csv"
# v2: the earlier float-space study (if one exists) is incompatible with the categorical space.
DEFAULT_STUDY_DB = PROJECT_ROOT / "results" / "optuna_prune_search_v2.db"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "bayesian_search_summary.csv"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "bayesian_search"

STUDY_NAME = "ct_atpt_prune_search"
NUM_STAGES = 3  # fixed 3-stage staging (after the 4th, 7th and 10th blocks)
GRIDS = {
    "stage1_keep": [0.3, 0.5, 0.7, 0.9],
    "stage2_keep": [0.2, 0.4, 0.6, 0.8],
    "stage3_keep": [0.1, 0.3, 0.5, 0.7],
}
N_VALID_COMBOS = sum(
    a >= b >= c for a in GRIDS["stage1_keep"] for b in GRIDS["stage2_keep"] for c in GRIDS["stage3_keep"]
)
SUMMARY_FIELDS = ["trial_number", "stage1_keep", "stage2_keep", "stage3_keep", "accuracy", "is_warmstart", "state"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian (Optuna/TPE + Hyperband) search over per-stage ATPT keep fractions.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--uniform-csv", type=Path, default=DEFAULT_UNIFORM_CSV,
                   help="Existing uniform AdamW sweep results, used to warm-start the study.")
    p.add_argument("--study-db", type=Path, default=DEFAULT_STUDY_DB)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                   help="Scratch dir for per-trial checkpoints (deleted after every trial).")
    p.add_argument("--backup-dir", type=Path, default=None,
                   help="Optional persistent dir (e.g. on Drive). The study DB and summary CSV are copied "
                        "here after every trial, and restored from here if the local DB is missing.")
    p.add_argument("--n-trials", type=int, default=8, help="New configs to actually train (beyond the warm-start points).")
    p.add_argument("--epochs", type=int, default=10, help="Max epochs per trial (Hyperband max_resource).")
    p.add_argument("--early-stop-epoch", type=int, default=5,
                   help="Epochs every trial trains before Hyperband may stop it (Hyperband min_resource).")
    # The 8-epoch surrogate needs a compressed schedule, otherwise pruning would never switch on
    # before the epoch-3 pruning decision and every config would look identical.
    p.add_argument("--pruning-warmup-epochs", type=int, default=5,
                   help="Pruning-free epochs at the start of each trial.")
    p.add_argument("--prune-ramp-epochs", type=int, default=1)
    p.add_argument("--warmup-epochs", type=int, default=2, help="LR warmup epochs (scaled down for the short run).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Normal training batch size. Search trials use HALF of this (retrain_best_config.py uses the full value).")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
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
                   help="Validation metric reported to Optuna after every epoch and optimized.")
    return p.parse_args()


# ── warm start ──────────────────────────────────────────────────────────
def uniform_stage_targets(keep_frac: float) -> tuple[float, float, float]:
    """Per-stage cumulative targets a uniform --prune-target-keep run ACTUALLY used
    (ct_atpt.model._stage_final_cum_frac default): target ** (stage_rank / num_stages)."""
    return tuple(keep_frac ** (j / NUM_STAGES) for j in range(1, NUM_STAGES + 1))


def snap(value: float, grid: list[float]) -> float:
    return min(grid, key=lambda g: abs(g - value))


def load_warmstart_points(uniform_csv: Path, metric: str) -> list[tuple[float, tuple[float, float, float], float]]:
    """(uniform keep fraction, per-stage targets, metric) for every AdamW row in the sweep CSV."""
    if not uniform_csv.exists():
        print(f"No existing uniform sweep CSV at {uniform_csv} - skipping warm-start.")
        return []
    points = []
    with open(uniform_csv) as f:
        for row in csv.DictReader(f):
            if row.get("optimizer") not in (None, "", "adamw"):
                continue
            if metric not in row or row[metric] in (None, ""):
                continue
            keep_frac = 1.0 - float(row["prune_percent"]) / 100.0
            points.append((keep_frac, uniform_stage_targets(keep_frac), float(row[metric])))
    print(f"Loaded {len(points)} warm-start point(s) from {uniform_csv}")
    return points


def add_warmstart_trials(study: optuna.Study, points) -> None:
    """The sweep's continuous per-stage targets are snapped to the nearest grid value (the study is
    categorical). They were measured with the full 45-epoch recipe, so they are tagged
    fidelity='full_sweep' and never counted as searched trials."""
    seeded = {t.user_attrs.get("uniform_keep") for t in study.trials if t.user_attrs.get("is_warmstart")}
    dists = {k: optuna.distributions.CategoricalDistribution(v) for k, v in GRIDS.items()}
    for keep_frac, stages, value in points:
        if round(keep_frac, 4) in seeded:
            continue
        params = {name: snap(v, GRIDS[name]) for name, v in zip(GRIDS, stages)}
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=dists, value=value,
            user_attrs={"is_warmstart": True, "uniform_keep": round(keep_frac, 4), "fidelity": "full_sweep",
                        "unsnapped_stages": [round(v, 4) for v in stages]},
        ))


# ── study bookkeeping ───────────────────────────────────────────────────
def trial_score(t: optuna.trial.FrozenTrial) -> float | None:
    if t.value is not None:
        return float(t.value)
    if t.intermediate_values:
        return float(t.intermediate_values[max(t.intermediate_values)])
    return None


def searched_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    ok = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return [t for t in study.trials if t.user_attrs.get("searched") and t.state in ok]


def best_searched(study: optuna.Study):
    done = [t for t in searched_trials(study) if trial_score(t) is not None]
    return max(done, key=trial_score) if done else None


def write_summary(study: optuna.Study, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for t in sorted(study.trials, key=lambda t: t.number):
            score = trial_score(t)
            w.writerow({
                "trial_number": t.number,
                "stage1_keep": t.params.get("stage1_keep"),
                "stage2_keep": t.params.get("stage2_keep"),
                "stage3_keep": t.params.get("stage3_keep"),
                "accuracy": "" if score is None else score,
                "is_warmstart": bool(t.user_attrs.get("is_warmstart")),
                "state": t.state.name,
            })


def backup(args: argparse.Namespace) -> None:
    if args.backup_dir is None:
        return
    try:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        for src in (args.study_db, args.summary_csv):
            if src.exists():
                shutil.copy2(src, args.backup_dir / src.name)
    except OSError as e:
        print(f"  WARNING: backup to {args.backup_dir} failed: {e}")


def restore_from_backup(args: argparse.Namespace) -> None:
    if args.backup_dir is None or args.study_db.exists():
        return
    saved = args.backup_dir / args.study_db.name
    if saved.exists():
        args.study_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, args.study_db)
        print(f"Restored study DB from {saved}")


# ── one in-process trial ────────────────────────────────────────────────
def build_train_args(args, run_dir: Path, splits_dir: Path, s1: float, s2: float, s3: float, batch_size: int):
    p = lambda name: os.path.abspath(splits_dir / name)  # lexical: keeps symlinked split CSVs as-is
    argv = [
        "--train-manifest", p("train.csv"),
        "--val-manifest", p("val.csv"),
        "--output-dir", str(run_dir),
        "--pretrained",
        "--optimizer", "adamw",
        "--prune-keep-stage1", str(s1), "--prune-keep-stage2", str(s2), "--prune-keep-stage3", str(s3),
        "--embed-dim", str(args.embed_dim), "--transformer-depth", str(args.transformer_depth), "--heads", str(args.heads),
        "--depth", str(args.depth), "--height", str(args.height), "--width", str(args.width),
        "--patch-z", str(args.patch_z), "--patch-y", str(args.patch_y), "--patch-x", str(args.patch_x),
        "--pruning-mode", "adaptive",
        "--pruning-warmup-epochs", str(args.pruning_warmup_epochs),
        "--prune-ramp-epochs", str(args.prune_ramp_epochs),
        "--warmup-epochs", str(args.warmup_epochs),
        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size), "--grad-accum", str(args.grad_accum),
        "--num-workers", str(args.num_workers),
        "--cls-loss", "ce", "--label-smoothing", str(args.label_smoothing),
        "--drop-path", str(args.drop_path), "--lr", str(args.lr), "--seed", str(args.seed),
        "--det-weight", "0", "--amp", args.amp, "--tta", "--augment", "--best-metric", "roc_auc",
    ]
    return train_script.parse_args(argv)


def release_gpu() -> None:
    try:
        train_script.cleanup_distributed()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)


def main() -> None:
    args = parse_args()
    splits_dir = args.splits_dir
    missing = [f for f in ("train.csv", "val.csv") if not (splits_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {splits_dir}. Run scripts/prune_sweep/run_full_sweep.py "
            "(or make_train_val_test.py) first to build the split this search reuses."
        )
    if not 1 <= args.early_stop_epoch < args.epochs:
        raise ValueError("--early-stop-epoch must be at least 1 and smaller than --epochs")
    first_informative = args.pruning_warmup_epochs + args.prune_ramp_epochs + 1
    if args.early_stop_epoch < first_informative:
        print(f"WARNING: pruning reaches full strength only in epoch {first_informative} (1-indexed), but the first "
              f"early-stop check is after {args.early_stop_epoch} epochs, so every config looks identical at that "
              f"check and it cannot rank them. Use --early-stop-epoch >= {first_informative} for it to be informative.")
    search_batch = max(1, args.batch_size // 2)
    print(f"Search trials: batch_size={search_batch} (half of {args.batch_size}), grad_accum={args.grad_accum}, "
          f"max {args.epochs} epochs, metric={args.metric}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found - each trial will be extremely slow on CPU.")

    restore_from_backup(args)
    args.study_db.parent.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=args.early_stop_epoch, max_resource=args.epochs, reduction_factor=3),
        storage=f"sqlite:///{args.study_db}",
        load_if_exists=True,
    )
    add_warmstart_trials(study, load_warmstart_points(args.uniform_csv, args.metric))

    done = len(searched_trials(study))
    remaining = args.n_trials - done
    print(f"Searched trials already done: {done}; remaining: {max(remaining, 0)}")

    def objective(trial: optuna.Trial) -> float:
        s1 = trial.suggest_categorical("stage1_keep", GRIDS["stage1_keep"])
        s2 = trial.suggest_categorical("stage2_keep", GRIDS["stage2_keep"])
        s3 = trial.suggest_categorical("stage3_keep", GRIDS["stage3_keep"])

        # Progressive pruning (cumulative keep can only shrink). Violations cost nothing: no training.
        if not (s1 >= s2 >= s3):
            trial.set_user_attr("invalid", True)
            return 0.0

        # Same config already trained in this study: reuse its (deterministic, seeded) score.
        for t in searched_trials(study):
            if t.number != trial.number and t.params == trial.params and trial_score(t) is not None:
                trial.set_user_attr("duplicate_of", t.number)
                return trial_score(t)

        trial.set_user_attr("searched", True)
        run_dir = args.runs_root / f"trial_{trial.number}"
        last = {"value": 0.0}

        def on_epoch(epoch: int, metrics) -> None:
            value = float(getattr(metrics, args.metric))
            last["value"] = value
            trial.report(value, step=epoch + 1)  # step = epochs completed, so the first Hyperband rung is "after --early-stop-epoch epochs"
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after epoch {epoch + 1} ({args.metric}={value:.4f})")

        print(f"\n{'=' * 70}\nTrial {trial.number}: stage keeps = ({s1}, {s2}, {s3})\n{'=' * 70}")
        try:
            train_args = build_train_args(args, run_dir, splits_dir, s1, s2, s3, search_batch)
            train_script.main(train_args, epoch_callback=on_epoch)
            return last["value"]
        except optuna.TrialPruned as e:
            traceback.clear_frames(e.__traceback__)  # drop references to the model/optimizer held by the traceback
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            traceback.clear_frames(e.__traceback__)
            return 0.0
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
            release_gpu()

    def after_trial(study_: optuna.Study, frozen: optuna.trial.FrozenTrial) -> None:
        write_summary(study_, args.summary_csv)
        backup(args)
        n_done = len(searched_trials(study_))
        if n_done >= min(args.n_trials, N_VALID_COMBOS):
            study_.stop()

    if remaining > 0:
        kwargs = {}
        if "gc_after_trial" in inspect.signature(study.optimize).parameters:
            kwargs["gc_after_trial"] = True
        # TPE on a 64-cell grid often re-proposes invalid or already-trained combos. Those cost nothing
        # (no training), so allow many attempts; the callback stops the study as soon as n_trials configs
        # have really been trained.
        study.optimize(objective, n_trials=max(remaining * 60, 300), callbacks=[after_trial], **kwargs)
        if len(searched_trials(study)) < min(args.n_trials, N_VALID_COMBOS):
            print("WARNING: attempt limit reached before all requested configs were trained; re-run to continue.")
    write_summary(study, args.summary_csv)
    backup(args)

    print(f"\n{'=' * 70}\nSearch finished: {len(searched_trials(study))} trained config(s)\n{'=' * 70}")
    n_err = sum("error" in t.user_attrs for t in study.trials)
    if n_err:
        print(f"WARNING: {n_err} trial(s) crashed and were scored 0.0 (see 'error' user_attr / log above).")
    best = best_searched(study)
    if best is None:
        print("No trained trial finished - nothing to report.")
    else:
        print(f"Best searched config ({args.metric}={trial_score(best):.4f}, trial #{best.number}, "
              f"{best.state.name.lower()}):")
        print(f"  Stage 1 keep: {best.params['stage1_keep']}")
        print(f"  Stage 2 keep: {best.params['stage2_keep']}")
        print(f"  Stage 3 keep: {best.params['stage3_keep']}")
    print(f"\nStudy DB: {args.study_db}\nSummary CSV: {args.summary_csv}")
    print("\nNext: python scripts/retrain_best_config.py")


if __name__ == "__main__":
    main()
