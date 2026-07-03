"""Aggregate per-fold best checkpoints into mean +/- std CV metrics.

Reads the `metrics` dict saved in each fold's best_checkpoint.pt and reports
mean and standard deviation across folds for the headline metrics. Use this to
produce the reportable cross-validation number for the paper.

Usage:
    python scripts/aggregate_folds.py --runs-root /content/drive/MyDrive/runs/cv_strict
    # expects <runs-root>/fold*/best_checkpoint.pt

    # or list checkpoints explicitly:
    python scripts/aggregate_folds.py --checkpoints runA/best_checkpoint.pt runB/best_checkpoint.pt
"""
from __future__ import annotations

import argparse
import pickle
import statistics
from pathlib import Path

import torch

HEADLINE = [
    "roc_auc", "pr_auc", "balanced_accuracy", "best_balanced_accuracy",
    "sensitivity", "specificity", "f1", "accuracy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate CV fold metrics.")
    p.add_argument("--runs-root", type=Path, default=None,
                   help="Directory containing fold*/best_checkpoint.pt")
    p.add_argument("--checkpoints", type=Path, nargs="+", default=None,
                   help="Explicit list of best_checkpoint.pt paths")
    p.add_argument("--ckpt-name", default="best_checkpoint.pt",
                   help="Checkpoint filename to look for under each fold dir")
    return p.parse_args()


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [Path(c) for c in args.checkpoints]
    if args.runs_root:
        found = sorted(args.runs_root.glob(f"*/{args.ckpt_name}"))
        if not found:
            raise SystemExit(f"No '{args.ckpt_name}' found under {args.runs_root}/*/")
        return found
    raise SystemExit("Provide --runs-root or --checkpoints")


def main() -> None:
    args = parse_args()
    ckpts = collect_checkpoints(args)

    per_metric: dict[str, list[float]] = {k: [] for k in HEADLINE}
    print(f"{'fold':<28} {'roc_auc':>9} {'best_bal':>9} {'sens':>7} {'spec':>7} {'src':>4}")
    print("-" * 70)
    loaded = 0
    for ckpt_path in ckpts:
        # weights_only=False: checkpoints store an args dict with pathlib.Path
        # objects, which the PyTorch>=2.6 safe loader rejects. These are our
        # own trusted checkpoints.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except (EOFError, RuntimeError, pickle.UnpicklingError) as exc:
            print(f"{ckpt_path.parent.name:<28} (CORRUPT/truncated checkpoint — skipped: "
                  f"{type(exc).__name__}; size={ckpt_path.stat().st_size} bytes — rerun this fold)")
            continue
        loaded += 1
        metrics = ckpt.get("metrics") or {}
        if not metrics:
            print(f"{ckpt_path.parent.name:<28} (no metrics saved — skipped)")
            continue
        for k in HEADLINE:
            if metrics.get(k) is not None:
                per_metric[k].append(float(metrics[k]))
        src = "ema" if ckpt.get("best_is_ema") else "raw"
        print(f"{ckpt_path.parent.name:<28} "
              f"{metrics.get('roc_auc', float('nan')):>9.4f} "
              f"{metrics.get('best_balanced_accuracy', float('nan')):>9.4f} "
              f"{metrics.get('sensitivity', float('nan')):>7.4f} "
              f"{metrics.get('specificity', float('nan')):>7.4f} "
              f"{src:>4}")

    print("\n" + "=" * 70)
    print(f"Cross-validation summary over {loaded} of {len(ckpts)} folds")
    print("=" * 70)
    for k in HEADLINE:
        vals = per_metric[k]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {k:<24} {mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
