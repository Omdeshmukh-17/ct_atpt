"""Master setup script for the pruning-percentage sweep.

Does two things:
  1. Runs make_train_val_test.py to build the strict, patient-level
     train/val/test split (70/15/15) under splits_tvt/.
  2. Generates 9 standalone runnable scripts, one per pruning percentage
     (10, 20, ..., 90). Each generated script (a) trains a model with
     --prune-target-keep = 1 - prune_percent/100, using val.csv for
     checkpoint selection, then (b) evaluates the resulting checkpoint on
     the held-out test.csv via eval_and_save_metrics.py, appending a row to
     results/all_prune_metrics.csv and writing a per-run ROC curve PNG.

On Windows this emits .bat files; on Linux/Mac it emits .sh files. Nothing is
executed automatically beyond the split step — run the generated scripts
yourself, one at a time (they're heavy full training runs).

Usage:
    python scripts/prune_sweep/run_full_sweep.py \
        --manifest /path/to/lidc_manifest.csv \
        --batch-size 4 --grad-accum 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SWEEP_DIR.parents[1]
SPLITS_DIR = SWEEP_DIR / "splits_tvt"
RUNS_DIR_NAME = "runs" / Path("prune_sweep")
GENERATED_DIR = SWEEP_DIR / "generated_scripts"

PRUNE_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

TRAIN_SCRIPT = "scripts/train_ct_atpt_ddp.py"
EVAL_SCRIPT = "scripts/prune_sweep/eval_and_save_metrics.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set up the full CT-ATPT pruning-percentage sweep.")
    p.add_argument("--manifest", type=Path, required=True, help="Full manifest CSV from prepare_lidc.py")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                   help="Where to write/read train.csv, val.csv, test.csv")
    p.add_argument("--skip-split", action="store_true",
                   help="Skip regenerating the train/val/test split (use an existing splits-dir).")
    p.add_argument("--batch-size", type=int, default=4, help="Per-step batch size (laptop-GPU friendly default).")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--pruning-warmup-epochs", type=int, default=10)
    p.add_argument("--prune-ramp-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--depth", type=int, default=96, help="Crop depth (Z) — 96 for the ViT-B/16-compatible recipe.")
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--patch-z", type=int, default=8)
    p.add_argument("--patch-y", type=int, default=16)
    p.add_argument("--patch-x", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--transformer-depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to invoke in generated scripts.")
    return p.parse_args()


def make_split(args: argparse.Namespace) -> None:
    if args.skip_split:
        print(f"--skip-split set: using existing split at {args.splits_dir}")
        return
    print("=" * 70)
    print("Step 1/2: building strict patient-level train/val/test split")
    print("=" * 70)
    cmd = [
        sys.executable,
        str(SWEEP_DIR / "make_train_val_test.py"),
        "--manifest", str(args.manifest),
        "--out-dir", str(args.splits_dir),
        "--seed", str(args.seed),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _common_train_flags(args: argparse.Namespace, prune_percent: int, run_dir: str) -> list[str]:
    keep_target = round(1.0 - prune_percent / 100.0, 4)
    return [
        "--train-manifest", "scripts/prune_sweep/splits_tvt/train.csv",
        "--val-manifest", "scripts/prune_sweep/splits_tvt/val.csv",
        "--output-dir", run_dir,
        "--pretrained",
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


def _eval_flags(prune_percent: int, run_dir: str) -> list[str]:
    return [
        "--checkpoint", f"{run_dir}/best_checkpoint.pt",
        "--val-manifest", "scripts/prune_sweep/splits_tvt/test.csv",
        "--prune-percent", str(prune_percent),
        "--amp", "bf16",
        "--tta",
    ]


def _quote_win(parts: list[str]) -> str:
    out = []
    for part in parts:
        if " " in part:
            out.append(f'"{part}"')
        else:
            out.append(part)
    return " ".join(out)


def _quote_posix(parts: list[str]) -> str:
    out = []
    for part in parts:
        if " " in part or any(c in part for c in "()"):
            out.append(f'"{part}"')
        else:
            out.append(part)
    return " ".join(out)


def generate_scripts(args: argparse.Namespace) -> list[Path]:
    print("=" * 70)
    print("Step 2/2: generating 9 per-percentage sweep scripts")
    print("=" * 70)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    is_windows = sys.platform.startswith("win")
    written: list[Path] = []

    for pct in PRUNE_PERCENTS:
        run_dir = f"runs/prune_sweep/prune{pct}"
        train_cmd = [args.python, TRAIN_SCRIPT] + _common_train_flags(args, pct, run_dir)
        eval_cmd = [args.python, EVAL_SCRIPT] + _eval_flags(pct, run_dir)

        if is_windows:
            script_path = args.generated_dir / f"run_prune{pct}.bat"
            lines = [
                "@echo off",
                f"REM Pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}%% pruned)",
                "cd /d %~dp0..\\..\\..",
                "",
                "echo === Training prune%s ===" % pct,
                _quote_win(train_cmd),
                "if errorlevel 1 (",
                "    echo Training failed for prune%s, aborting.",
                "    exit /b 1",
                ")",
                "",
                "echo === Evaluating prune%s on held-out test set ===" % pct,
                _quote_win(eval_cmd),
                "if errorlevel 1 (",
                "    echo Evaluation failed for prune%s.",
                "    exit /b 1",
                ")",
                "",
                "echo Done: prune%s" % pct,
            ]
            script_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        else:
            script_path = args.generated_dir / f"run_prune{pct}.sh"
            lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"# Pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}% pruned)",
                'cd "$(dirname "$0")/../../.."',
                "",
                f'echo "=== Training prune{pct} ==="',
                _quote_posix(train_cmd),
                "",
                f'echo "=== Evaluating prune{pct} on held-out test set ==="',
                _quote_posix(eval_cmd),
                "",
                f'echo "Done: prune{pct}"',
            ]
            script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            script_path.chmod(0o755)

        written.append(script_path)
        print(f"  wrote {script_path.relative_to(PROJECT_ROOT)}")

    return written


def main() -> None:
    args = parse_args()
    make_split(args)
    scripts = generate_scripts(args)

    print()
    print("=" * 70)
    print("SWEEP SETUP COMPLETE")
    print("=" * 70)
    print(f"Split written to: {args.splits_dir}")
    print(f"Generated {len(scripts)} scripts in: {args.generated_dir}")
    print()
    print("Run them ONE AT A TIME (each is a full training + eval job), in order:")
    for s in scripts:
        print(f"  {s.relative_to(PROJECT_ROOT)}")
    print()
    print("Each script trains a model at its pruning percentage (checkpoint")
    print("selected on val.csv), then evaluates the resulting checkpoint on the")
    print("held-out test.csv, appending one row to:")
    print(f"  {SWEEP_DIR / 'results' / 'all_prune_metrics.csv'}")
    print("and saving an ROC curve to:")
    print(f"  {SWEEP_DIR / 'results' / 'plots' / 'prune<pct>.png'}")
    print()
    print("After all 9 have finished, summarize with:")
    print("  python scripts/prune_sweep/plot_summary.py")


if __name__ == "__main__":
    main()
