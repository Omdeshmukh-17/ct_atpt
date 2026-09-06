"""Master setup script for the PhyAdam side of the pruning-percentage sweep.

Mirrors ../run_full_sweep.py but:
  - does NOT recreate the train/val/test split — it reuses the existing
    splits_tvt/{train,val,test}.csv produced by the AdamW sweep and errors
    out with instructions if they're missing.
  - trains with --optimizer phyadam (+ --base-mass/--mass-scale/--friction)
    instead of AdamW.
  - writes to separate output dirs (runs/phyadam_prune_XX) and a separate
    results CSV (results/all_phyadam_metrics.csv) so AdamW and PhyAdam runs
    never collide.

Generated scripts live directly in this directory (scripts/prune_sweep/phyadam/),
named run_phyadam_prune_XX.bat (Windows) or .sh (Linux/Mac).

Usage:
    python scripts/prune_sweep/phyadam/run_phyadam_sweep.py --splits-dir splits_tvt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHYADAM_DIR = Path(__file__).resolve().parent
SWEEP_DIR = PHYADAM_DIR.parent
PROJECT_ROOT = SWEEP_DIR.parents[1]
DEFAULT_SPLITS_DIR = SWEEP_DIR / "splits_tvt"

PRUNE_PERCENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

TRAIN_SCRIPT = "scripts/train_ct_atpt_ddp.py"
EVAL_SCRIPT = "scripts/prune_sweep/eval_and_save_metrics.py"
PHYADAM_CSV = "results/all_phyadam_metrics.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set up the PhyAdam side of the CT-ATPT pruning-percentage sweep.")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                   help="Directory containing the existing train.csv/val.csv/test.csv "
                        "(produced by the AdamW sweep's make_train_val_test.py). Not regenerated here.")
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
    # PhyAdam-specific hyperparameters.
    p.add_argument("--base-mass", type=float, default=1.0, help="PhyAdam: base particle mass M0")
    p.add_argument("--mass-scale", type=float, default=0.1, help="PhyAdam: mass scaling coefficient alpha")
    p.add_argument("--friction", type=float, default=0.1, help="PhyAdam: friction coefficient mu")
    p.add_argument("--generated-dir", type=Path, default=PHYADAM_DIR,
                   help="Where to write the generated run_phyadam_prune_XX scripts.")
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
        "--optimizer", "phyadam",
        "--base-mass", str(args.base_mass),
        "--mass-scale", str(args.mass_scale),
        "--friction", str(args.friction),
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


def _eval_flags(prune_percent: int, run_dir: str, splits_dir_rel: str) -> list[str]:
    return [
        "--checkpoint", f"{run_dir}/best_checkpoint.pt",
        "--val-manifest", f"{splits_dir_rel}/test.csv",
        "--prune-percent", str(prune_percent),
        "--optimizer-name", "phyadam",
        "--csv-path", PHYADAM_CSV,
        "--amp", "bf16",
        "--tta",
    ]


def _quote_win(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def _quote_posix(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if (" " in part or any(c in part for c in "()")) else part for part in parts)


def generate_scripts(args: argparse.Namespace, splits_dir_rel: str) -> list[Path]:
    print("=" * 70)
    print("Generating 9 PhyAdam per-percentage sweep scripts")
    print("=" * 70)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    is_windows = sys.platform.startswith("win")
    written: list[Path] = []

    for pct in PRUNE_PERCENTS:
        run_dir = f"runs/phyadam_prune_{pct}"
        train_cmd = [args.python, TRAIN_SCRIPT] + _common_train_flags(args, pct, run_dir, splits_dir_rel)
        eval_cmd = [args.python, EVAL_SCRIPT] + _eval_flags(pct, run_dir, splits_dir_rel)

        if is_windows:
            script_path = args.generated_dir / f"run_phyadam_prune_{pct}.bat"
            lines = [
                "@echo off",
                f"REM PhyAdam pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}%% pruned)",
                "cd /d %~dp0..\\..\\..",
                "",
                "echo === Training phyadam_prune_%s ===" % pct,
                _quote_win(train_cmd),
                "if errorlevel 1 (",
                f"    echo Training failed for phyadam_prune_{pct}, aborting.",
                "    exit /b 1",
                ")",
                "",
                "echo === Evaluating phyadam_prune_%s on held-out test set ===" % pct,
                _quote_win(eval_cmd),
                "if errorlevel 1 (",
                f"    echo Evaluation failed for phyadam_prune_{pct}.",
                "    exit /b 1",
                ")",
                "",
                "echo Done: phyadam_prune_%s" % pct,
            ]
            script_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        else:
            script_path = args.generated_dir / f"run_phyadam_prune_{pct}.sh"
            lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"# PhyAdam pruning sweep step: keep target = {round(1 - pct/100, 2)} ({pct}% pruned)",
                'cd "$(dirname "$0")/../../.."',
                "",
                f'echo "=== Training phyadam_prune_{pct} ==="',
                _quote_posix(train_cmd),
                "",
                f'echo "=== Evaluating phyadam_prune_{pct} on held-out test set ==="',
                _quote_posix(eval_cmd),
                "",
                f'echo "Done: phyadam_prune_{pct}"',
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
        # splits_dir isn't under PROJECT_ROOT (e.g. an absolute path elsewhere) — use it as-is.
        splits_dir_rel = str(args.splits_dir)

    scripts = generate_scripts(args, splits_dir_rel)

    print()
    print("=" * 70)
    print("PHYADAM SWEEP SETUP COMPLETE")
    print("=" * 70)
    print(f"Reused split from: {args.splits_dir}")
    print(f"Generated {len(scripts)} scripts in: {args.generated_dir}")
    print()
    print("Run them ONE AT A TIME (each is a full training + eval job), in order:")
    for s in scripts:
        print(f"  {_rel_to_root(s)}")
    print()
    print("Each script trains a model at its pruning percentage with PhyAdam")
    print("(checkpoint selected on val.csv), then evaluates the resulting")
    print("checkpoint on the held-out test.csv, appending one row to:")
    print(f"  {SWEEP_DIR / 'results' / 'all_phyadam_metrics.csv'}")
    print("and saving an ROC curve to:")
    print(f"  {SWEEP_DIR / 'results' / 'plots' / 'phyadam_prune<pct>.png'}")
    print()
    print("After all 9 AdamW AND all 9 PhyAdam runs have finished, compare with:")
    print("  python scripts/prune_sweep/phyadam/plot_comparison.py")


if __name__ == "__main__":
    main()
