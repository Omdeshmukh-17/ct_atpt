from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from ct_atpt.data import CTVolumeDataset
from ct_atpt.inference import tta_average_proba
from ct_atpt.losses import detection_loss, focal_loss
from ct_atpt.metrics import BinaryMetrics, compute_binary_metrics, format_binary_metrics
from ct_atpt.model import CTATPT, CTATPTConfig
from ct_atpt.phyadam import PhyAdam
from scripts.load_pretrained import load_imagenet_vit_into_ctatpt


class ModelEma:
    """Exponential moving average of model parameters and buffers.

    EMA weights typically track a smoother optimum than the raw weights and
    are robust to the late-training instability seen on this small dataset.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        m = model.module if isinstance(model, DistributedDataParallel) else model
        return getattr(m, "_orig_mod", m)  # unwrap torch.compile

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_sd = self.module.state_dict()
        model_sd = self._unwrap(model).state_dict()
        for key, ema_v in ema_sd.items():
            model_v = model_sd[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v.to(ema_v.dtype), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay — standard for ViT training."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CT-ATPT with PyTorch DDP.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ct_atpt")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize transformer backbone with ViT-B/16 ImageNet weights. "
                             "Requires embed_dim=768, transformer_depth=12, heads=12.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 per GPU for dynamic CT token compaction.")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--patch-z", type=int, default=8)
    parser.add_argument("--patch-y", type=int, default=32)
    parser.add_argument("--patch-x", type=int, default=32)

    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.0,
                        help="Stochastic-depth rate (linear 0->value across blocks). 0.1 is a good default for fine-tuning.")
    parser.add_argument("--min-keep-tokens", type=int, default=64)
    parser.add_argument("--min-keep-ratio", type=float, default=0.25)
    parser.add_argument("--max-prune-fraction-per-block", type=float, default=0.5)
    parser.add_argument("--pruning-mode", choices=["adaptive", "none", "soft"], default="adaptive")
    parser.add_argument("--pruning-warmup-epochs", type=int, default=10,
                        help="Train without pruning for this many epochs so the backbone learns features first.")
    parser.add_argument("--prune-target-keep", type=float, default=0.5,
                        help="(soft & adaptive modes) final fraction of patch tokens to keep after the ramp.")
    parser.add_argument("--prune-ramp-epochs", type=int, default=15,
                        help="(soft & adaptive modes) epochs to ramp keep-ratio from 1.0 down to --prune-target-keep, starting after warmup.")
    parser.add_argument("--sparsity-weight", type=float, default=0.5,
                        help="Weight on the keep-ratio sparsity penalty. Soft mode: one-sided keep>target "
                             "penalty. Adaptive mode: centres the pre-clamp soft keep fraction on the "
                             "budget track so lambda/tau learn to decide counts inside the band.")
    parser.add_argument("--soft-lambda-init", type=float, default=-2.0,
                        help="(soft mode) init for the unconstrained threshold scalar. Less negative (e.g. -1.0, 0.0) => gates start responsive so they can actually prune.")
    parser.add_argument("--gate-sharpness", type=float, default=10.0,
                        help="(soft mode) sigmoid sharpness kappa. Lower (e.g. 4-5) keeps gates differentiable instead of saturating at 0/1.")
    parser.add_argument(
        "--scale-kept-tokens",
        action="store_true",
        help="Use the old soft-gated kept-token scaling. Default preserves kept tokens at full strength.",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=["adamw", "phyadam"], default="adamw",
                        help="Optimizer to use")
    parser.add_argument("--base-mass", type=float, default=1.0,
                        help="PhyAdam: base particle mass M0")
    parser.add_argument("--mass-scale", type=float, default=0.1,
                        help="PhyAdam: mass scaling coefficient alpha")
    parser.add_argument("--friction", type=float, default=0.1,
                        help="PhyAdam: friction coefficient mu")
    parser.add_argument("--prune-lr-mult", type=float, default=1.0,
                        help="LR multiplier for the pruning scalars (importance_logits/α,β,γ, "
                             "soft_lambda_raw, lambda_raw, temperature_raw). These also get "
                             "weight_decay=0 so decay can't pin α/β/γ at uniform. Use 10-50 in "
                             "soft mode so the pruning policy actually moves.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (0 = disabled).")
    parser.add_argument("--det-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--cls-loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for cross-entropy (ignored for focal). 0.05-0.1 is typical.")
    parser.add_argument("--soft-labels", action="store_true",
                        help="Train on continuous targets from mean_malignancy (LIDC 1-5 → [0,1]) "
                             "via soft-target cross-entropy. Reduces label noise at the benign/malignant "
                             "boundary. Validation metrics still use the hard label.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="MixUp Beta(a,a) strength (0 = off). Mixes volumes and classification "
                             "targets within a batch; 0.2-0.4 is a good starting range. Requires batch-size>1.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma for class imbalance.")
    parser.add_argument("--focal-alpha", type=float, default=0.5, help="Positive-class weight in focal loss.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Weight-EMA decay (0 = disabled). 0.999 is a good default; best checkpoint is taken across raw and EMA.")
    parser.add_argument("--tta", action="store_true",
                        help="Flip test-time augmentation during validation (averages probs over 8 axis-flip views).")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training-time augmentation (flips, noise, brightness). "
                             "Disable for overfit sanity tests.")
    parser.add_argument(
        "--best-metric",
        choices=["roc_auc", "balanced_accuracy", "sensitivity", "f1", "pr_auc"],
        default="roc_auc",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight init, dropout, shuffling and MixUp. "
                             "Without this, runs are not comparable (warmup AUC alone can "
                             "swing by 0.2 on this dataset).")
    return parser.parse_args()


def set_seed(seed: int, rank: int) -> None:
    """Seed all RNGs. Offset by rank so DDP workers don't correlate."""
    import random as _random
    _random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def build_target_probs(batch: dict, args: argparse.Namespace, num_classes: int = 2) -> torch.Tensor:
    """Build a [B, num_classes] target distribution for soft-target cross-entropy.

    Uses the continuous `soft_label` (from mean_malignancy) when --soft-labels is
    set, otherwise the hard label as a one-hot. Label smoothing is folded in so it
    composes with the soft-target path.
    """
    if args.soft_labels:
        s = batch["soft_label"].float().clamp(0.0, 1.0)
    else:
        s = batch["label"].float()
    probs = torch.stack([1.0 - s, s], dim=1)  # [B, 2]; binary task
    if args.label_smoothing > 0:
        probs = probs * (1.0 - args.label_smoothing) + args.label_smoothing / num_classes
    return probs


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a target probability distribution (soft labels / MixUp)."""
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    epoch: int,
    ema: ModelEma | None = None,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        # MixUp mixes the input volume, so it must happen before the forward pass.
        volume = batch["volume"]
        mixup_active = args.mixup_alpha > 0.0 and volume.shape[0] > 1
        if mixup_active:
            lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            perm = torch.randperm(volume.shape[0], device=volume.device)
            volume = lam * volume + (1.0 - lam) * volume[perm]

        # Soft-target path covers soft labels and MixUp (both need a target
        # distribution rather than integer classes); focal is incompatible with both.
        soft_path = args.soft_labels or mixup_active

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(volume)

            if soft_path:
                target_probs = build_target_probs(batch, args)
                if mixup_active:
                    target_probs = lam * target_probs + (1.0 - lam) * target_probs[perm]
                cls_loss = soft_cross_entropy(logits, target_probs)
            elif args.cls_loss == "focal":
                cls_loss = focal_loss(
                    logits, batch["label"],
                    gamma=args.focal_gamma,
                    alpha=args.focal_alpha,
                )
            else:
                cls_loss = F.cross_entropy(
                    logits, batch["label"], label_smoothing=args.label_smoothing
                )
            # Detection loss uses the original (unmixed) targets; it is degenerate
            # on this dataset and typically run with --det-weight 0.
            det_loss = detection_loss(
                aux["det_pred"],
                aux["grid_shape"],
                batch["det_target"],
                batch["has_det"],
                focal_gamma=args.focal_gamma,
            )
            loss = (
                cls_loss
                + args.det_weight * det_loss
                + args.entropy_weight * aux["entropy_loss"]
                + args.consistency_weight * aux["consistency_loss"]
                + args.sparsity_weight * aux["sparsity_loss"]
            )
            loss = loss / args.grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            if scaler is not None:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        if is_main_process(rank) and step % 10 == 0:
            keep = aux["keep_ratios"].detach().float().cpu().tolist()
            abg  = aux["alpha_beta_gamma"].detach().float().cpu().tolist()
            fb   = int(aux["fallback_count"].item())
            lr   = scheduler.get_last_lr()[0]
            print(
                f"epoch={epoch:03d} step={step:04d} lr={lr:.2e} "
                f"loss={(loss.item() * args.grad_accum):.4f} "
                f"cls={cls_loss.item():.4f} det={det_loss.item():.4f} "
                f"ent={aux['entropy_loss'].item():.4f} "
                f"con={aux['consistency_loss'].item():.4f} "
                f"spar={aux['sparsity_loss'].item():.4f} "
                f"lambda={float(aux['lambda']):.3f} "
                f"keep={[round(v, 3) for v in keep]} "
                f"abg={[round(v, 3) for v in abg]} "
                f"fallbacks={fb}"
            )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    tta: bool = False,
) -> BinaryMetrics:
    model.eval()
    labels: list[int] = []
    positive_scores: list[float] = []
    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if tta:
                probs = tta_average_proba(model, batch["volume"])[:, 1]
            else:
                logits, _ = model(batch["volume"])
                probs = torch.softmax(logits.float(), dim=1)[:, 1]

        labels.extend(batch["label"].detach().cpu().int().tolist())
        positive_scores.extend(probs.detach().cpu().float().tolist())

    if dist.is_available() and dist.is_initialized():
        gathered: list[tuple[list[int], list[float]] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (labels, positive_scores))
        labels = []
        positive_scores = []
        for item in gathered:
            if item is None:
                continue
            item_labels, item_scores = item
            labels.extend(item_labels)
            positive_scores.extend(item_scores)

    return compute_binary_metrics(labels, positive_scores)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    rank: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    filename: str | None = None,
    metrics: BinaryMetrics | None = None,
    best_metric_value: float = float("-inf"),
    model_state: dict | None = None,
    ema_state: dict | None = None,
    best_is_ema: bool = False,
) -> None:
    if not is_main_process(rank):
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    ckpt = {
        "epoch": epoch,
        # "model" holds the weights that achieved best_metric_value (raw or EMA),
        # so eval_checkpoint.py loads the right ones with no extra flags.
        "model": model_state if model_state is not None else raw_model.state_dict(),
        "ema": ema_state,
        "best_is_ema": best_is_ema,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "metrics": metrics.__dict__ if metrics is not None else None,
        "best_metric_value": best_metric_value,
    }
    torch.save(ckpt, args.output_dir / (filename or f"checkpoint_epoch_{epoch:03d}.pt"))


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()
    set_seed(args.seed, rank)

    train_ds = CTVolumeDataset(
        args.train_manifest,
        target_shape=(args.depth, args.height, args.width),
        augment=args.augment,
    )
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_manifest is not None:
        val_ds = CTVolumeDataset(
            args.val_manifest,
            target_shape=(args.depth, args.height, args.width),
            augment=False,
        )
        # Validation uses SequentialSampler on rank-0 only to avoid DDP padding
        # artefacts that would repeat samples and corrupt metrics.
        val_sampler = SequentialSampler(val_ds) if world_size == 1 else DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    config = CTATPTConfig(
        input_shape=(args.depth, args.height, args.width),
        patch_size=(args.patch_z, args.patch_y, args.patch_x),
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        num_heads=args.heads,
        dropout=args.dropout,
        drop_path_rate=args.drop_path,
        min_keep_tokens=args.min_keep_tokens,
        min_keep_ratio=args.min_keep_ratio,
        max_prune_fraction_per_block=args.max_prune_fraction_per_block,
        pruning_mode=args.pruning_mode,
        scale_kept_tokens=args.scale_kept_tokens,
        pruning_warmup_epochs=args.pruning_warmup_epochs,
        prune_target_keep=args.prune_target_keep,
        prune_ramp_epochs=args.prune_ramp_epochs,
        soft_lambda_init=args.soft_lambda_init,
        gate_sharpness=args.gate_sharpness,
    )
    model: torch.nn.Module = CTATPT(config).to(device)

    # Load pretrained ViT-B/16 ImageNet weights if requested
    if args.pretrained and args.resume is None:
        if is_main_process(rank):
            print("Loading pretrained ViT-B/16 ImageNet weights...")
        load_imagenet_vit_into_ctatpt(model, verbose=is_main_process(rank))

    # EMA tracks the unwrapped model; create it before compile/DDP wrapping.
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0.0 else None

    if args.compile:
        model = torch.compile(model)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # Split the pruning policy scalars into their own param-group: no weight
    # decay (so decay can't drag importance_logits back to zero → softmax pinned
    # at uniform [0.333,0.333,0.333]) and an optional higher LR (their gradients
    # are weak vs the backbone). Everything else keeps the original settings.
    PRUNE_PARAM_NAMES = {
        "importance_logits", "soft_lambda_raw", "lambda_raw", "temperature_raw",
    }
    core = model.module if isinstance(model, DistributedDataParallel) else model
    core = getattr(core, "_orig_mod", core)  # unwrap torch.compile
    prune_param_ids = {
        id(p) for name, p in core.named_parameters()
        if name.split(".")[-1] in PRUNE_PARAM_NAMES
    }
    prune_params = [p for p in model.parameters() if id(p) in prune_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in prune_param_ids]
    if args.optimizer == "phyadam":
        optimizer = PhyAdam(
            [
                {"params": other_params, "weight_decay": args.weight_decay, "lr": args.lr},
                {"params": prune_params, "weight_decay": 0.0, "lr": args.lr * args.prune_lr_mult},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
            base_mass=args.base_mass,
            mass_scale=args.mass_scale,
            friction=args.friction,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": args.weight_decay, "lr": args.lr},
                {"params": prune_params, "weight_decay": 0.0, "lr": args.lr * args.prune_lr_mult},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if is_main_process(rank):
        print(f"Optimizer: {args.optimizer.upper()} | {len(other_params)} backbone params @ lr={args.lr:.2e} (wd={args.weight_decay}) | "
              f"{len(prune_params)} pruning scalars @ lr={args.lr * args.prune_lr_mult:.2e} (wd=0)")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    if args.amp != "fp16":
        scaler = None

    # Steps-based cosine schedule with linear warmup
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(total_steps, steps_per_epoch * args.warmup_epochs)
    scheduler = build_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_metric_value = float("-inf")

    # Checkpoint resume
    if args.resume is not None and args.resume.is_file():
        # weights_only=False: checkpoint stores an args dict with pathlib.Path
        # objects that the PyTorch>=2.6 safe loader rejects. Trusted source.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric_value = ckpt.get("best_metric_value", float("-inf"))
        if ema is not None:
            if ckpt.get("ema") is not None:
                ema.module.load_state_dict(ckpt["ema"])
            else:
                # No EMA stored (e.g. resuming a pre-EMA run): seed from current weights.
                ema = ModelEma(ModelEma._unwrap(model), args.ema_decay)
        if is_main_process(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if is_main_process(rank):
        effective_batch = args.batch_size * world_size * args.grad_accum
        print(f"Training on world_size={world_size}, device={device}, effective_batch={effective_batch}")
        print(f"Transformer: CT-ATPT, dim={args.embed_dim}, depth={args.transformer_depth}, heads={args.heads}")
        print(
            "Pruning: "
            f"mode={args.pruning_mode}, "
            f"warmup_epochs={args.pruning_warmup_epochs}, "
            f"min_keep_tokens={args.min_keep_tokens}, "
            f"min_keep_ratio={args.min_keep_ratio:.2f}, "
            f"max_prune_fraction_per_block={args.max_prune_fraction_per_block:.2f}, "
            f"scale_kept_tokens={args.scale_kept_tokens}"
        )
        if args.pruning_mode == "soft":
            print(
                "Soft pruning: "
                f"target_keep={args.prune_target_keep:.2f}, "
                f"ramp_epochs={args.prune_ramp_epochs} (after warmup), "
                f"sparsity_weight={args.sparsity_weight}"
            )
        print(f"LR schedule: warmup={args.warmup_epochs} epochs, cosine decay over {args.epochs} epochs")
        if args.soft_labels:
            print("Classification loss: soft-target cross_entropy (mean_malignancy → [0,1]), "
                  f"label_smoothing={args.label_smoothing}")
        elif args.cls_loss == "focal":
            print(f"Classification loss: focal, gamma={args.focal_gamma}, alpha={args.focal_alpha}")
        else:
            print(f"Classification loss: cross_entropy, label_smoothing={args.label_smoothing}")
        if args.mixup_alpha > 0:
            print(f"MixUp: enabled (alpha={args.mixup_alpha})")
        print(
            f"Regularization: drop_path={args.drop_path} | "
            f"EMA: {'decay=' + str(args.ema_decay) if args.ema_decay > 0 else 'off'} | "
            f"val TTA: {'on (8 flips)' if args.tta else 'off'}"
        )

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)  # reseed augmentation RNG for this epoch

        # Tell the model which epoch we're in so pruning warmup works correctly.
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model.set_epoch(epoch)
        if is_main_process(rank) and args.pruning_mode == "adaptive":
            if epoch < args.pruning_warmup_epochs:
                print(f"epoch={epoch:03d} [pruning warmup {epoch+1}/{args.pruning_warmup_epochs} — pruning disabled]")
            elif epoch == args.pruning_warmup_epochs:
                print(f"epoch={epoch:03d} [pruning warmup complete — adaptive pruning enabled]")

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args, rank, epoch, ema=ema)

        if val_loader is not None:
            raw_metrics = evaluate(model, val_loader, device, args, tta=args.tta)
            raw_value = float(getattr(raw_metrics, args.best_metric))

            # Candidate = better of raw vs EMA weights by the selection metric.
            best_value = raw_value
            best_metrics = raw_metrics
            best_is_ema = False
            if ema is not None:
                ema.module.set_epoch(epoch)
                ema_metrics = evaluate(ema.module, val_loader, device, args, tta=args.tta)
                ema_value = float(getattr(ema_metrics, args.best_metric))
                if ema_value > raw_value:
                    best_value, best_metrics, best_is_ema = ema_value, ema_metrics, True

            if is_main_process(rank):
                print(f"epoch={epoch:03d} [raw] {format_binary_metrics(raw_metrics)}")
                if ema is not None:
                    print(f"epoch={epoch:03d} [ema] {format_binary_metrics(ema_metrics)}")

                if best_value > best_metric_value:
                    best_metric_value = best_value
                    raw_core = ModelEma._unwrap(model)
                    model_state = ema.module.state_dict() if best_is_ema else raw_core.state_dict()
                    save_checkpoint(
                        model, optimizer, args, epoch, rank,
                        scheduler=scheduler,
                        filename="best_checkpoint.pt",
                        metrics=best_metrics,
                        best_metric_value=best_metric_value,
                        model_state=model_state,
                        ema_state=ema.module.state_dict() if ema is not None else None,
                        best_is_ema=best_is_ema,
                    )
                    tag = "ema" if best_is_ema else "raw"
                    print(f"new_best_{args.best_metric}={best_value:.4f} ({tag})")

        # Only save best checkpoint — no per-epoch saves to conserve storage.

    cleanup_distributed()


if __name__ == "__main__":
    main()
