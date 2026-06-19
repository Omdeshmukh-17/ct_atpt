from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _random_zoom_3d(
    volume: np.ndarray,
    rng: random.Random,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Isotropic random zoom via trilinear resampling, restored to original shape.

    Scales the whole volume by a single factor (mimics nodule-size / scanner-FOV
    variation), then center crops or pads back to the input shape so the tensor
    size is unchanged. Dependency-free (uses torch's interpolate).
    """
    scale = rng.uniform(*scale_range)
    if abs(scale - 1.0) < 1e-3:
        return volume

    original_shape = volume.shape
    t = torch.from_numpy(np.ascontiguousarray(volume))[None, None]  # [1, 1, Z, Y, X]
    zoomed = torch.nn.functional.interpolate(
        t, scale_factor=scale, mode="trilinear", align_corners=False, recompute_scale_factor=False
    )
    out = zoomed[0, 0].numpy().astype(np.float32)
    return center_pad_or_crop_zyx(out, original_shape)


def augment_volume(
    volume: np.ndarray,
    det_values: list[float],
    has_det: bool,
    rng: random.Random,
) -> tuple[np.ndarray, list[float]]:
    """3D augmentations for CT volumes: flips, 90° in-plane rotations, zoom, jitter.

    All transforms are label-preserving for nodule malignancy. Flips and 90°
    rotations are exact symmetries (no interpolation); zoom and intensity jitter
    mimic scanner / acquisition variability. The nodule is centered in the crop,
    so flips/rotations keep the (0.5, 0.5, 0.5) centroid invariant.

    Args:
        volume: float32 array [Z, Y, X] normalized to [0, 1].
        det_values: [center_z, center_y, center_x, radius] normalized coords.
        has_det: whether detection targets are valid.
        rng: seeded random.Random instance for reproducibility.

    Returns:
        Augmented volume and updated det_values.
    """
    det = list(det_values)

    # Random flip along each spatial axis independently (p=0.5 each)
    for axis, coord_idx in [(0, 0), (1, 1), (2, 2)]:
        if rng.random() < 0.5:
            volume = np.flip(volume, axis=axis).copy()
            if has_det:
                det[coord_idx] = 1.0 - det[coord_idx]

    # Random 90° rotation in the axial (Y, X) plane. A centered nodule's
    # centroid (0.5, 0.5) is invariant under these rotations, so det is unchanged.
    k = rng.randint(0, 3)
    if k:
        volume = np.rot90(volume, k=k, axes=(1, 2)).copy()

    # Random isotropic zoom (p=0.5) — restored to original shape.
    if rng.random() < 0.5:
        volume = _random_zoom_3d(volume, rng)

    # Gaussian intensity noise (sigma ~ U[0, 0.02])
    if rng.random() < 0.7:
        noise_sigma = rng.uniform(0.0, 0.02)
        noise_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        volume = (volume + noise_rng.normal(0.0, noise_sigma, volume.shape)).astype(np.float32)

    # Random HU brightness shift (±5% of range)
    if rng.random() < 0.5:
        shift = rng.uniform(-0.05, 0.05)
        volume = (volume + shift).astype(np.float32)

    volume = np.clip(volume, 0.0, 1.0)
    return volume, det


def _center_pad_or_crop_axis(volume: np.ndarray, target: int, axis: int) -> np.ndarray:
    size = volume.shape[axis]
    if size == target:
        return volume

    if size > target:
        start = (size - target) // 2
        stop = start + target
        slices = [slice(None)] * volume.ndim
        slices[axis] = slice(start, stop)
        return volume[tuple(slices)]

    pad_before = (target - size) // 2
    pad_after = target - size - pad_before
    pads = [(0, 0)] * volume.ndim
    pads[axis] = (pad_before, pad_after)
    return np.pad(volume, pads, mode="constant", constant_values=0.0)


def center_pad_or_crop_zyx(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    for axis, target in enumerate(target_shape):
        volume = _center_pad_or_crop_axis(volume, target, axis)
    return volume


class CTVolumeDataset(Dataset):
    """Loads normalised CT nodule crops saved as `.npy`.

    Manifest columns (produced by scripts/prepare_lidc.py):
    - volume_path : path to float32 .npy crop  [Z, Y, X]
    - label       : 0 = benign, 1 = malignant
                    (derived from mean LIDC-IDRI radiologist score: >3 → malignant)
    - center_z/y/x: nodule centre normalised to [0, 1] within the crop
    - radius      : nodule radius normalised by crop size

    Detection values are relative to the final volume shape after pad/crop.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        target_shape: tuple[int, int, int] = (128, 512, 512),
        augment: bool = False,
        soft_label_min: float = 1.0,
        soft_label_max: float = 5.0,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.root = self.manifest_csv.parent
        self.target_shape = target_shape
        self.augment = augment
        # Range used to map the LIDC mean_malignancy score (1..5) to a soft
        # target in [0, 1]. Consumed only when training with --soft-labels.
        self.soft_label_min = soft_label_min
        self.soft_label_max = soft_label_max
        self._epoch = 0
        self._rng = random.Random(42)

        with self.manifest_csv.open("r", encoding="utf-8", newline="") as f:
            self.rows: list[dict[str, str]] = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_csv}")

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the augmentation RNG so each epoch gets different random augmentations."""
        self._epoch = epoch
        self._rng = random.Random(epoch * 1000 + 42)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_volume_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        volume_path = self._resolve_volume_path(row["volume_path"])
        volume = np.load(volume_path).astype(np.float32)

        if volume.ndim != 3:
            raise ValueError(f"Expected volume shape (Z, Y, X), got {volume.shape} from {volume_path}")

        volume = center_pad_or_crop_zyx(volume, self.target_shape)
        volume = np.clip(volume, 0.0, 1.0)

        label = int(row["label"])

        # Soft target from the averaged radiologist malignancy score (1..5),
        # mapped to [0, 1]. Falls back to the hard label when the column is
        # missing/blank. Training uses this only under --soft-labels; metrics
        # always use the hard `label` so reported AUC stays honest.
        mean_mal = row.get("mean_malignancy", "")
        if mean_mal not in ("", None):
            span = max(self.soft_label_max - self.soft_label_min, 1e-6)
            soft_label = (float(mean_mal) - self.soft_label_min) / span
            soft_label = float(min(1.0, max(0.0, soft_label)))
        else:
            soft_label = float(label)

        det_values = []
        has_det = True
        for key in ("center_z", "center_y", "center_x", "radius"):
            value = row.get(key, "")
            if value == "":
                has_det = False
                det_values.append(0.0)
            else:
                det_values.append(float(value))

        if self.augment:
            # Use a per-sample RNG seeded by (epoch, index) for true randomness across epochs
            sample_rng = random.Random(self._epoch * 100003 + index)
            volume, det_values = augment_volume(volume, det_values, has_det, sample_rng)

        return {
            "volume": torch.from_numpy(volume).unsqueeze(0),  # [1, Z, Y, X]
            "label": torch.tensor(label, dtype=torch.long),
            "soft_label": torch.tensor(soft_label, dtype=torch.float32),
            "det_target": torch.tensor(det_values, dtype=torch.float32),
            "has_det": torch.tensor(has_det, dtype=torch.bool),
            "path": str(volume_path),
        }

