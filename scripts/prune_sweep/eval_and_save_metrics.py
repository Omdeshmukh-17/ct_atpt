"""Evaluate a saved CT-ATPT checkpoint and append its metrics to a shared CSV.

Same checkpoint/config-reconstruction pattern as scripts/eval_checkpoint.py, plus:
  - a --prune-percent tag written into the output row (identifies which sweep
    point this checkpoint belongs to)
  - one appended row in results/all_prune_metrics.csv (header written once)
  - an individual ROC curve PNG in results/plots/prune{percent}.png

Usage:
    python scripts/prune_sweep/eval_and_save_metrics.py \
        --checkpoint runs/prune10/best_checkpoint.pt \
        --val-manifest scripts/prune_sweep/splits_tvt/test.csv \
        --prune-percent 10 \
        --amp bf16 --tta
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, SequentialSampler

from ct_atpt.data import CTVolumeDataset
from ct_atpt.inference import tta_average_proba
from ct_atpt.metrics import compute_binary_metrics, format_binary_metrics
from ct_atpt.model import CTATPT, CTATPTConfig

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "all_prune_metrics.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

CSV_FIELDS = [
    "optimizer",
    "prune_percent",
    "checkpoint",
    "manifest",
    "n_samples",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "specificity",
    "balanced_accuracy",
    "best_balanced_accuracy",
    "best_threshold",
    "tp",
    "tn",
    "fp",
    "fn",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CT-ATPT checkpoint and log metrics for the prune sweep.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    # Named --val-manifest to match the existing scripts/eval_checkpoint.py
    # convention, even though it's pointed at the held-out test split here.
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--prune-percent", type=float, required=True,
                   help="Pruning percentage this checkpoint was trained for (e.g. 10, 20, ... 90). Tags the output row.")
    p.add_argument("--optimizer-name", type=str, default="adamw",
                   help="Name of optimizer used (for CSV column)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--tta", action="store_true",
                   help="Flip test-time augmentation (averages probs over 8 axis-flip views).")
    p.add_argument("--csv-path", type=Path, default=CSV_PATH, help="Shared metrics CSV to append to.")
    p.add_argument("--plots-dir", type=Path, default=PLOTS_DIR, help="Directory for per-run ROC curve PNGs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    # weights_only=False: our checkpoints store an args dict with pathlib.Path
    # objects that the PyTorch>=2.6 safe loader rejects. Trusted source.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    saved_args = ckpt.get("args", {})
    print(f"Checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Saved best metric value: {ckpt.get('best_metric_value', '?')}")

    # ── Reconstruct model config from saved args ─────────────────────
    config = CTATPTConfig(
        input_shape=(
            saved_args.get("depth", 96),
            saved_args.get("height", 96),
            saved_args.get("width", 96),
        ),
        patch_size=(
            saved_args.get("patch_z", 8),
            saved_args.get("patch_y", 16),
            saved_args.get("patch_x", 16),
        ),
        embed_dim=saved_args.get("embed_dim", 768),
        depth=saved_args.get("transformer_depth", 12),
        num_heads=saved_args.get("heads", 12),
        dropout=saved_args.get("dropout", 0.1),
        min_keep_tokens=saved_args.get("min_keep_tokens", 48),
        min_keep_ratio=saved_args.get("min_keep_ratio", 0.40),
        max_prune_fraction_per_block=saved_args.get("max_prune_fraction_per_block", 0.30),
        pruning_mode=saved_args.get("pruning_mode", "adaptive"),
        scale_kept_tokens=saved_args.get("scale_kept_tokens", False),
        pruning_warmup_epochs=saved_args.get("pruning_warmup_epochs", 20),
        # Soft-mode fields: gate_sharpness is a plain float (not in the state
        # dict) and prune_target_keep sets the eval-time lambda schedule, so
        # omitting them silently evaluates soft checkpoints with wrong gates.
        prune_target_keep=saved_args.get("prune_target_keep", 0.5),
        prune_ramp_epochs=saved_args.get("prune_ramp_epochs", 15),
        soft_lambda_init=saved_args.get("soft_lambda_init", -2.0),
        gate_sharpness=saved_args.get("gate_sharpness", 10.0),
        drop_path_rate=saved_args.get("drop_path", 0.0),
    )
    print(f"Model config: dim={config.embed_dim}, depth={config.depth}, "
          f"heads={config.num_heads}, patch={config.patch_size}, "
          f"prune_target_keep={config.prune_target_keep}")

    model = CTATPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    # Set epoch high so pruning is active (past warmup).
    model.set_epoch(999)
    model.eval()

    # ── Load evaluation data ─────────────────────────────────────────
    eval_ds = CTVolumeDataset(
        args.val_manifest,
        target_shape=config.input_shape,
        augment=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(eval_ds),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluation samples: {len(eval_ds)}")

    # ── Run evaluation ───────────────────────────────────────────────
    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    labels: list[int] = []
    scores: list[float] = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if args.tta:
                    probs = tta_average_proba(model, volume)[:, 1]
                else:
                    logits, _aux = model(volume)
                    probs = torch.softmax(logits.float(), dim=1)[:, 1]
            labels.extend(label.int().tolist())
            scores.extend(probs.cpu().float().tolist())

            if (i + 1) % 5 == 0:
                print(f"  Processed {(i+1) * args.batch_size}/{len(eval_ds)} samples")

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = compute_binary_metrics(labels, scores)
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS (prune_percent={args.prune_percent})")
    print(f"{'='*70}")
    print(format_binary_metrics(metrics))
    print(f"{'='*70}")

    # ── Append row to shared CSV ──────────────────────────────────────
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.csv_path.exists()
    row = {
        "optimizer": args.optimizer_name,
        "prune_percent": args.prune_percent,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.val_manifest),
        "n_samples": len(labels),
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.sensitivity,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "specificity": metrics.specificity,
        "balanced_accuracy": metrics.balanced_accuracy,
        "best_balanced_accuracy": metrics.best_balanced_accuracy,
        "best_threshold": metrics.best_threshold,
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
    }
    with open(args.csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nAppended row to {args.csv_path}")

    # ── ROC curve plot ────────────────────────────────────────────────
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, scores)
    # Keep the original "prune{N}.png" naming for the default (adamw) sweep so
    # existing outputs/paths don't change; prefix other optimizers so their
    # plots never collide with adamw's in the shared results/plots/ directory.
    plot_stem = f"prune{int(args.prune_percent)}" if args.optimizer_name == "adamw" else f"{args.optimizer_name}_prune{int(args.prune_percent)}"
    plot_path = args.plots_dir / f"{plot_stem}.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {metrics.roc_auc:.3f})", color="C0", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {args.optimizer_name.upper()} — {int(args.prune_percent)}% Pruning")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve to {plot_path}")


if __name__ == "__main__":
    main()
