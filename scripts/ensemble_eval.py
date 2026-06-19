"""Ensemble several CT-ATPT checkpoints by averaging their softmax probabilities.

Averaging predictions from independently trained models is a near-guaranteed
ROC-AUC gain and a standard way to produce a strong headline number. Two common
uses:

  * Multi-seed ensemble — several models trained on the SAME split with different
    seeds, evaluated on that split's held-out val set.
  * Cross-fold ensemble — the 5 fold `best_checkpoint.pt` models, evaluated on a
    single common held-out TEST manifest (NOT the per-fold val sets, which differ).

All checkpoints are run on the one `--val-manifest` you pass, so the sample order
(SequentialSampler) and labels line up across models before averaging.

Usage:
    python scripts/ensemble_eval.py \
        --checkpoints runs/fold0/best_checkpoint.pt runs/fold1/best_checkpoint.pt ... \
        --val-manifest /path/to/test_manifest.csv \
        --tta --amp bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from ct_atpt.data import CTVolumeDataset
from ct_atpt.inference import tta_average_proba
from ct_atpt.metrics import compute_binary_metrics, format_binary_metrics
from ct_atpt.model import CTATPT, CTATPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble CT-ATPT checkpoints by probability averaging.")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="Two or more .pt checkpoints to average.")
    p.add_argument("--val-manifest", type=Path, required=True,
                   help="Common manifest all checkpoints are evaluated on.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--tta", action="store_true",
                   help="Flip test-time augmentation (averages probs over 8 axis-flip views).")
    return p.parse_args()


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> CTATPT:
    """Reconstruct a CT-ATPT model from a checkpoint's saved args and load weights."""
    saved_args = ckpt.get("args", {})
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
    )
    model = CTATPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_epoch(999)  # past warmup so pruning (if any) is active at eval
    model.eval()
    return model, config


@torch.no_grad()
def predict_scores(
    model: CTATPT,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[int], list[float]]:
    """Return (labels, positive-class probabilities) in dataset order."""
    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    labels: list[int] = []
    scores: list[float] = []
    for batch in loader:
        volume = batch["volume"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if args.tta:
                probs = tta_average_proba(model, volume)[:, 1]
            else:
                logits, _ = model(volume)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
        labels.extend(batch["label"].int().tolist())
        scores.extend(probs.cpu().float().tolist())
    return labels, scores


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        print("⚠️  Only one checkpoint given — this is a plain eval, not an ensemble.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  TTA: {'on' if args.tta else 'off'}")

    # The dataset is identical across checkpoints, so load it once. We still
    # confirm each checkpoint's input_shape matches it (mismatched crops can't be
    # ensembled meaningfully).
    first_ckpt = torch.load(args.checkpoints[0], map_location=device, weights_only=False)
    _, first_config = build_model_from_checkpoint(first_ckpt, device)
    val_ds = CTVolumeDataset(args.val_manifest, target_shape=first_config.input_shape, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_ds),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Validation samples: {len(val_ds)}  |  input_shape={first_config.input_shape}")

    ref_labels: list[int] | None = None
    prob_matrix: list[list[float]] = []

    for idx, ckpt_path in enumerate(args.checkpoints):
        print(f"\n[{idx + 1}/{len(args.checkpoints)}] {ckpt_path}")
        ckpt = first_ckpt if idx == 0 else torch.load(ckpt_path, map_location=device, weights_only=False)
        model, config = build_model_from_checkpoint(ckpt, device)
        if config.input_shape != first_config.input_shape:
            raise ValueError(
                f"Checkpoint {ckpt_path} input_shape {config.input_shape} != "
                f"{first_config.input_shape}; cannot ensemble different crop sizes."
            )

        labels, scores = predict_scores(model, val_loader, device, args)
        if ref_labels is None:
            ref_labels = labels
        elif labels != ref_labels:
            raise RuntimeError("Label order differs across checkpoints — manifest/order mismatch.")

        member_metrics = compute_binary_metrics(labels, scores)
        print(f"  member AUC={member_metrics.roc_auc:.4f}  pr_auc={member_metrics.pr_auc:.4f}")
        prob_matrix.append(scores)

        # Free GPU memory before loading the next model.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert ref_labels is not None
    mean_scores = np.mean(np.asarray(prob_matrix, dtype=np.float64), axis=0).tolist()
    ensemble_metrics = compute_binary_metrics(ref_labels, mean_scores)

    member_aucs = [compute_binary_metrics(ref_labels, s).roc_auc for s in prob_matrix]
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE OF {len(args.checkpoints)} CHECKPOINTS")
    print(f"{'=' * 70}")
    print(f"Member AUCs:   {[round(a, 4) for a in member_aucs]}")
    print(f"Mean member:   {np.mean(member_aucs):.4f}")
    print(f"Ensemble AUC:  {ensemble_metrics.roc_auc:.4f}  "
          f"(Δ vs mean member = {ensemble_metrics.roc_auc - float(np.mean(member_aucs)):+.4f})")
    print(format_binary_metrics(ensemble_metrics))
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
