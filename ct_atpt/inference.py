from __future__ import annotations

import itertools

import torch


def flip_combinations(axes: tuple[int, ...] = (2, 3, 4)) -> list[tuple[int, ...]]:
    """All flip/no-flip combinations over the given spatial axes (includes identity).

    For the 3 spatial axes of a [B, 1, Z, Y, X] volume this yields 8 views.
    Axis flips are label-preserving for nodule malignancy, so averaging over
    them is a valid test-time augmentation.
    """
    combos: list[tuple[int, ...]] = []
    for r in range(len(axes) + 1):
        combos.extend(itertools.combinations(axes, r))
    return combos


@torch.no_grad()
def tta_average_proba(
    model: torch.nn.Module,
    volume: torch.Tensor,
    axes: tuple[int, ...] = (2, 3, 4),
) -> torch.Tensor:
    """Average softmax probabilities over axis-flip TTA.

    Args:
        model:  callable returning (logits, aux).
        volume: [B, 1, Z, Y, X].
        axes:   spatial axes to flip over.

    Returns:
        Mean softmax probabilities [B, num_classes] over all flip views.
    """
    prob_sum: torch.Tensor | None = None
    combos = flip_combinations(axes)
    for combo in combos:
        view = torch.flip(volume, dims=list(combo)) if combo else volume
        logits, _ = model(view)
        probs = torch.softmax(logits.float(), dim=1)
        prob_sum = probs if prob_sum is None else prob_sum + probs
    return prob_sum / len(combos)
