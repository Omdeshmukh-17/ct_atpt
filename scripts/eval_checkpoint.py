"""Evaluate a saved CT-ATPT checkpoint on a validation manifest.

Usage:
    python eval_checkpoint.py \
        --checkpoint /path/to/best_checkpoint.pt \
        --val-manifest /path/to/val_manifest.csv \
        --amp bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, SequentialSampler

from ct_atpt.data import CTVolumeDataset
from ct_atpt.inference import tta_average_proba
from ct_atpt.metrics import compute_binary_metrics, format_binary_metrics
from ct_atpt.model import CTATPT, CTATPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CT-ATPT checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--tta", action="store_true",
                   help="Flip test-time augmentation (averages probs over 8 axis-flip views).")
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
    if ckpt.get("metrics"):
        print(f"Saved metrics: {ckpt['metrics']}")

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
    print(f"\nModel config: dim={config.embed_dim}, depth={config.depth}, "
          f"heads={config.num_heads}, patch={config.patch_size}")

    model = CTATPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    # Set epoch high so pruning is active (past warmup)
    model.set_epoch(999)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Load validation data ─────────────────────────────────────────
    val_ds = CTVolumeDataset(
        args.val_manifest,
        target_shape=config.input_shape,
        augment=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_ds),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Validation samples: {len(val_ds)}")

    # ── Run evaluation ───────────────────────────────────────────────
    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    labels: list[int] = []
    scores: list[float] = []
    keep_ratios_sum = None

    print("\nRunning evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, aux = model(volume)
                if args.tta:
                    probs = tta_average_proba(model, volume)[:, 1]
                else:
                    probs = torch.softmax(logits.float(), dim=1)[:, 1]
            labels.extend(label.int().tolist())
            scores.extend(probs.cpu().float().tolist())

            kr = aux["keep_ratios"].detach().float().cpu()
            if keep_ratios_sum is None:
                keep_ratios_sum = kr
            else:
                keep_ratios_sum = keep_ratios_sum + kr

            if (i + 1) % 5 == 0:
                print(f"  Processed {(i+1) * args.batch_size}/{len(val_ds)} samples")

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = compute_binary_metrics(labels, scores)
    n_batches = len(val_loader)
    avg_keep = (keep_ratios_sum / n_batches).tolist()

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(format_binary_metrics(metrics))
    print(f"\nAvg token keep ratios: {[round(k, 3) for k in avg_keep]}")
    print(f"{'='*70}")

    # Compare with saved metrics
    saved_auc = ckpt.get("best_metric_value", None)
    if saved_auc is not None:
        print(f"\nSaved AUC:      {saved_auc:.4f}")
        print(f"Reproduced AUC: {metrics.roc_auc:.4f}")
        diff = abs(metrics.roc_auc - saved_auc)
        if diff < 0.01:
            print("✅ MATCH — checkpoint reproduces within 1% tolerance")
        else:
            print(f"⚠️  MISMATCH — difference = {diff:.4f}")


if __name__ == "__main__":
    main()
