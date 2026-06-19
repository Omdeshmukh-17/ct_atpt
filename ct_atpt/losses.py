from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """Focal loss for class-imbalanced classification.

    Focal loss down-weights easy negatives via (1-p_t)^gamma, focusing
    training on hard examples. Essential for LUNA16 where positives are
    ~0.2% of candidates.

    Args:
        logits: [B, C] unnormalized class scores.
        targets: [B] integer class labels.
        gamma: focusing parameter — higher = more focus on hard examples.
        alpha: optional per-sample class weight for the positive class.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = (1.0 - pt) ** gamma * ce
    if alpha is not None:
        weight = targets.float() * alpha + (1.0 - targets.float()) * (1.0 - alpha)
        loss = loss * weight
    return loss.mean()


def make_grid_centers(
    grid_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gz, gy, gx = grid_shape
    z = (torch.arange(gz, device=device, dtype=dtype) + 0.5) / gz
    y = (torch.arange(gy, device=device, dtype=dtype) + 0.5) / gy
    x = (torch.arange(gx, device=device, dtype=dtype) + 0.5) / gx
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)


def detection_loss(
    det_pred: torch.Tensor,
    grid_shape: tuple[int, int, int],
    det_target: torch.Tensor,
    has_det: torch.Tensor,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Anchor-free detection loss supporting multiple nodules per volume.

    det_pred shape: [B, N, 4]
      channel 0: objectness logit
      channels 1-3: center offsets (tanh-bounded, scaled by grid)

    det_target shape: [B, 4], normalized [center_z, center_y, center_x, radius].
    The radius column is unused by this loss but retained in the manifest for
    downstream analysis.
    has_det shape: [B] bool — whether this sample has an annotated nodule.

    Objectness uses focal BCE to handle the severe 1-positive / N-1-negative
    imbalance across patch tokens.
    """
    device = det_pred.device
    dtype = det_pred.dtype
    batch_size, num_tokens, _ = det_pred.shape
    centers = make_grid_centers(grid_shape, device=device, dtype=dtype)

    total = det_pred.new_tensor(0.0)
    valid_count = 0

    for b in range(batch_size):
        if not bool(has_det[b].item()):
            continue

        target_center = det_target[b, :3].to(device=device, dtype=dtype).clamp(0.0, 1.0)

        dist = torch.sum((centers - target_center.unsqueeze(0)) ** 2, dim=-1)
        positive_idx = torch.argmin(dist)

        # Focal BCE for objectness — reduces easy-negative dominance
        obj_logits = det_pred[b, :, 0]
        obj_probs  = torch.sigmoid(obj_logits)
        obj_target = torch.zeros(num_tokens, device=device, dtype=dtype)
        obj_target[positive_idx] = 1.0
        bce = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
        pt  = obj_target * obj_probs + (1.0 - obj_target) * (1.0 - obj_probs)
        object_loss = ((1.0 - pt) ** focal_gamma * bce).mean()

        grid_scale = torch.tensor(grid_shape, device=device, dtype=dtype)
        pred_offset = torch.tanh(det_pred[b, positive_idx, 1:4]) / grid_scale
        pred_center = (centers[positive_idx] + pred_offset).clamp(0.0, 1.0)

        reg_loss = F.smooth_l1_loss(pred_center, target_center)
        total = total + object_loss + reg_loss
        valid_count += 1

    if valid_count == 0:
        return det_pred.sum() * 0.0

    return total / valid_count

