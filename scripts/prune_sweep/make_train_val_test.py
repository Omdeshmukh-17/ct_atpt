"""Build a strict, patient-level 70/15/15 train/val/test split from a full LIDC manifest.

Strict labels (matches scripts/make_folds.py):
  - benign    (0): mean_malignancy <= --benign-max   (default 2.0)
  - malignant (1): mean_malignancy >= --malignant-min (default 4.0)
  - dropped      : 2.0 < mean_malignancy < 4.0 (indeterminate)

Splitting is done in two stages, both patient-grouped and label-stratified via
StratifiedGroupKFold so no patient_id ever appears in more than one of the three
final sets:
  1. Split off ~15% of patients as the held-out test set (1 fold of a K-fold
     where K = round(1 / test_fraction)).
  2. From the remaining ~85%, split off val as val_fraction / (1 - test_fraction)
     of that remainder (again via StratifiedGroupKFold), leaving train.

Usage:
    python scripts/prune_sweep/make_train_val_test.py \
        --manifest /path/to/lidc_manifest.csv \
        --out-dir  scripts/prune_sweep/splits_tvt \
        --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a strict patient-level train/val/test split.")
    p.add_argument("--manifest", type=Path, required=True, help="Full manifest CSV from prepare_lidc.py")
    p.add_argument("--out-dir", type=Path, default=Path("scripts/prune_sweep/splits_tvt"),
                   help="Directory to write train.csv / val.csv / test.csv")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--benign-max", type=float, default=2.0, help="mean_malignancy <= this => benign")
    p.add_argument("--malignant-min", type=float, default=4.0, help="mean_malignancy >= this => malignant")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _stratified_group_split(df: pd.DataFrame, holdout_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (remainder, holdout) with holdout ~= holdout_frac of rows,
    grouped by patient_id and stratified by label. Uses one fold of a K-fold
    StratifiedGroupKFold where K = round(1/holdout_frac)."""
    n_splits = max(2, round(1.0 / holdout_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["patient_id"].values
    y = df["label"].values
    remainder_idx, holdout_idx = next(sgkf.split(df, y, groups))
    remainder = df.iloc[remainder_idx].reset_index(drop=True)
    holdout = df.iloc[holdout_idx].reset_index(drop=True)
    return remainder, holdout


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {frac_sum}")

    df = pd.read_csv(args.manifest)
    for col in ("mean_malignancy", "patient_id"):
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column '{col}'. Found: {list(df.columns)}")

    n_raw = len(df)
    mm = df["mean_malignancy"].astype(float)
    is_benign = mm <= args.benign_max
    is_malig = mm >= args.malignant_min
    df = df[is_benign | is_malig].copy()
    df["label"] = is_malig[is_benign | is_malig].astype(int).values
    df = df.reset_index(drop=True)

    print(f"Strict filter: kept {len(df)}/{n_raw} nodules "
          f"(benign<= {args.benign_max}, malignant>= {args.malignant_min})")
    print(f"  benign={int((df['label'] == 0).sum())}  malignant={int((df['label'] == 1).sum())}")
    print(f"  patients={df['patient_id'].nunique()}")

    # Stage 1: carve off test set.
    remainder_df, test_df = _stratified_group_split(df, args.test_frac, args.seed)

    # Stage 2: from the remainder, carve off val. val's share of the remainder
    # is val_frac / (train_frac + val_frac) so the final proportions match
    # train_frac/val_frac/test_frac of the original data.
    val_share_of_remainder = args.val_frac / (args.train_frac + args.val_frac)
    train_df, val_df = _stratified_group_split(remainder_df, val_share_of_remainder, args.seed)

    # Hard leakage checks across all three pairs.
    train_pt = set(train_df["patient_id"])
    val_pt = set(val_df["patient_id"])
    test_pt = set(test_df["patient_id"])
    assert not (train_pt & val_pt), f"Patient leakage train/val: {len(train_pt & val_pt)} ids"
    assert not (train_pt & test_pt), f"Patient leakage train/test: {len(train_pt & test_pt)} ids"
    assert not (val_pt & test_pt), f"Patient leakage val/test: {len(val_pt & test_pt)} ids"

    train_path = args.out_dir / "train.csv"
    val_path = args.out_dir / "val.csv"
    test_path = args.out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    def _summary(name: str, d: pd.DataFrame) -> str:
        return (f"{name}: n={len(d)} ({d['patient_id'].nunique()} patients, "
                f"{100 * d['label'].mean():.1f}% malignant)")

    print()
    print(_summary("train", train_df))
    print(_summary("val  ", val_df))
    print(_summary("test ", test_df))
    print(f"\nNo patient overlap across train/val/test — verified.")
    print(f"Wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()
