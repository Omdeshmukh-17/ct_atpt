"""Master setup script for the SAM+AdamW side of the pruning-percentage sweep.

Mirrors ../run_full_sweep.py (AdamW) and ../phyadam/run_phyadam_sweep.py, but:
  - does NOT recreate the train/val/test split — it reuses the existing
    splits_tvt/{train,val,test}.csv produced by the AdamW sweep and errors
    out with instructions if they're missing.
  - trains with --optimizer sam_adamw (+ --sam-rho), AdamW base hyperparams
    lr=3e-4 / weight_decay=0.01 (SAM's own recommended range, separate from
    the plain-AdamW sweep's lr=1e-4).
  - forces --grad-accum 1 (SAM needs two full forward/backward passes per
    batch; accumulation across multiple batches is not defined for its
    ascent step — see ct_atpt/sam.py and the SAM branch in
    scripts/train_ct_atpt_ddp.py's training loop).
  - writes to separate output dirs (runs/sam_adamw_prune_XX) and a separate
    results CSV (results/all_sam_adamw_metrics.csv) so AdamW/PhyAdam/Lion/SAM
    runs never collide.

Generated scripts live directly in this directory
(scripts/prune_sweep/sam_adamw/), named run_sam_adamw_prune_XX.bat (Windows)
or .sh (Linux/Mac).

Usage:
    python scripts/prune_sweep/sam_adamw/run_sam_adamw_sweep.py --splits-dir splits_tvt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SAM_DIR = Path(__file__).resolve().parent
SWEEP_DIR = SAM_DIR.parent
PROJECT_ROOT = SWEEP_DIR.parents[1]
DEFAULT_SPLITS_DIR = SWEEP_DIR / "splits_tvt"

PRUNE_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

TRAIN_SCRIPT = "scripts/train_ct_atpt_ddp.py"
EVAL_SCRIPT = "scripts/prune_sweep/eval_and_save_metrics.py"
SAM_CSV = "results/all_sam_adamw_metrics.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set up the SAM+AdamW side of the CT-ATPT pruning-percentage sweep.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                   help="Directory containing the existing train.csv/val.csv/test.csv "
                        "(produced by the AdamW sweep's make_train_val_test.py). Not regenerated here.")
    # SAM does two forward/backward passes per batch already; keep the
    # per-batch size modest by default so a single pass at --grad-accum 1
    # still fits, rather than compensating with accumulation (unsupported).
    p.add_argument("--batch-size", type=int, default=4, help="Per-step batch size (laptop-GPU friendly default).")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--pruning-warmup-epochs", type=int, default=10)
    p.add_argument("--prune-ramp-epochs", type=int, default=15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["none", "bf16"], default="bf16",
                   help="SAM does not support fp16 (see ct_atpt/sam.py / the training loop's SAM branch).")
    p.add_argument("--depth", type=int, default=96, help="Crop depth (Z) — 96 for the ViT-B/16-compatible recipe.")
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--patch-z", type=int, default=8)
    p.add_argument("--patch-y", type=int, default=16)
    p.add_argument("--patch-x", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--transformer-depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    # SAM-specific hyperparameters. Base optimizer is AdamW; SAM's own
    # published recipes typically use a higher base lr than plain AdamW here
    # (lr=3e-4 vs. the AdamW sweep's 1e-4) since the ascent step already adds
    # regularization pressure.
    p.add_argument("--sam-rho", type=float, default=0.05, help="SAM: neighborhood size rho")
    p.add_argument("--lr", type=float, default=3e-4, help="SAM: base-optimizer (AdamW) learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01, help="SAM: base-optimizer (AdamW) weight decay")
    p.add_argument("--generated-dir", type=Path, default=SAM_DIR,
                   help="Where to write the generated run_sam_adamw_prune_XX scripts.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to invoke in generated scripts.")
    return p.parse_args()


def verify_splits(splits_dir: Path) -> None:
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    missing = [str(p) for p in (train_csv, val_csv, test_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split file(s): " + ", ".join(missing) + "\n"
            "Run the AdamW sweep's split step first:\n"
            "  python scripts/prune_sweep/run_full_sweep.py --manifest /path/to/lidc_manifest.csv\n"
            "(or, if you already have a manifest processed, just run "
            "scripts/prune_sweep/make_train_val_test.py directly) before running this script."
        )
    print(f"Found existing split: {train_csv}, {val_csv}, {test_csv}")


def _common_train_flags(args: argparse.Namespace, prune_percent: int, run_dir: str, splits_dir_rel: str) -> list[str]:
    keep_target = round(1.0 - prune_percent / 100.0, 4)
    return [
        "--train-manifest", f"{splits_dir_rel}/train.csv",
        "--val-manifest", f"{splits_dir_rel}/val.csv",
        "--output-dir", run_dir,
        "--pretrained",
        "--optimizer", "sam_adamw",
        "--sam-rho", str(args.sam_rho),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
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
        "--prune-target-keep", str(keep_target),
        "--pruning-warmup-epochs", str(args.pruning_warmup_epochs),
        "--prune-ramp-epochs", str(args.prune_ramp_epochs),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", "1",  # required for SAM — see module docstring
        "--cls-loss", "ce",
        "--label-smoothing", str(args.label_smoothing),
        "--drop-path", str(args.drop_path),
        "--seed", str(args.seed),
        "--det-weight", "0",
        "--amp", args.amp,
        "--tta",
        "--augment",
        "--best-metric", "roc_auc",
    ]


def _eval_flags(prune_percent: int, run_dir: str, splits_dir_rel: str) -> list[str]:
    return [
        "--checkpoint", f"{run_dir}/best_checkpoint.pt",
        "--val-manifest", f"{splits_dir_rel}/test.csv",
        "--prune-percent", str(prune_percent),
        "--optimizer-name", "sam_adamw",
        "--csv-path", SAM_CSV,
        "--amp", "bf16",
        "--tta",
    ]


def _quote_win(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def _quote_posix(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if (" " in part or any(c in part for c in "()")) else part for part in parts)


def generate_scripts(args: argparse.Namespace, splits_dir_rel: str) -> list[Path]:
    print("=" * 70)
    print("Generating 9 SAM+AdamW per-percentage sweep scripts")
    print("=" * 70)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    is_windows = sys.platform.startswith("win")
    written: list[Path] = []

    for pct in PRUNE_PERCENTS:
        run_dir = f"runs/sam_adamw_prune_{pct}"
        train_cmd = [args.python, TRAIN_SCRIPT] + _common_train_flags(args, pct, run_dir, splits_dir_rel)
        eval_cmd = [args.python, EVAL_SCRIPT] + _eval_flags(pct, run_dir, splits_dir_rel)

        if is_windows:
            script_path = args.generated_dir / f"run_sam_adamw_prune_{pct}.bat"
            lines = [
                "@echo off",
                f"REM SAM+AdamW pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}%% pruned)",
                "cd /d %~dp0..\\..\\..",
                "",
                "echo === Training sam_adamw_prune_%s ===" % pct,
                _quote_win(train_cmd),
                "if errorlevel 1 (",
                f"    echo Training failed for sam_adamw_prune_{pct}, aborting.",
                "    exit /b 1",
                ")",
                "",
                "echo === Evaluating sam_adamw_prune_%s on held-out test set ===" % pct,
                _quote_win(eval_cmd),
                "if errorlevel 1 (",
                f"    echo Evaluation failed for sam_adamw_prune_{pct}.",
                "    exit /b 1",
                ")",
                "",
                "echo Done: sam_adamw_prune_%s" % pct,
            ]
            script_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        else:
            script_path = args.generated_dir / f"run_sam_adamw_prune_{pct}.sh"
            lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"# SAM+AdamW pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}% pruned)",
                'cd "$(dirname "$0")/../../.."',
                "",
                f'echo "=== Training sam_adamw_prune_{pct} ==="',
                _quote_posix(train_cmd),
                "",
                f'echo "=== Evaluating sam_adamw_prune_{pct} on held-out test set ==="',
                _quote_posix(eval_cmd),
                "",
                f'echo "Done: sam_adamw_prune_{pct}"',
            ]
            script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            script_path.chmod(0o755)

        written.append(script_path)
        print(f"  wrote {_rel_to_root(script_path)}")

    return written


def _rel_to_root(path: Path) -> Path | str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    args = parse_args()
    verify_splits(args.splits_dir)

    try:
        splits_dir_rel = str(args.splits_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        splits_dir_rel = str(args.splits_dir)

    scripts = generate_scripts(args, splits_dir_rel)

    print()
    print("=" * 70)
    print("SAM+ADAMW SWEEP SETUP COMPLETE")
    print("=" * 70)
    print(f"Reused split from: {args.splits_dir}")
    print(f"Generated {len(scripts)} scripts in: {args.generated_dir}")
    print()
    print("Run them ONE AT A TIME (each is a full training + eval job, and each")
    print("training step itself runs two forward/backward passes — expect roughly")
    print("2x the wall-clock time of an equivalent AdamW run), in order:")
    for s in scripts:
        print(f"  {_rel_to_root(s)}")
    print()
    print("Each script trains a model at its pruning percentage with SAM+AdamW")
    print("(checkpoint selected on val.csv), then evaluates the resulting")
    print("checkpoint on the held-out test.csv, appending one row to:")
    print(f"  {SWEEP_DIR / 'results' / 'all_sam_adamw_metrics.csv'}")
    print("and saving an ROC curve to:")
    print(f"  {SWEEP_DIR / 'results' / 'plots' / 'sam_adamw' / 'prune<pct>.png'}")
    print()
    print("After all 9 AdamW/PhyAdam/Lion/SAM+AdamW runs have finished, compare with:")
    print("  python scripts/prune_sweep/plot_comparison.py")


if __name__ == "__main__":
    main()
