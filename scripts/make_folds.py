"""Build strict, patient-level 5-fold CV manifests from a full LIDC manifest.

Strict labels (paper Section IV / dataset_selection_criteria.md):
  - benign    (0): mean_malignancy <= --benign-max   (default 2.0)
  - malignant (1): mean_malignancy >= --malignant-min (default 4.0)
  - dropped      : 2.0 < mean_malignancy < 4.0 (indeterminate)

Folds use StratifiedGroupKFold on patient_id so (a) no patient appears in both
train and val of any fold, and (b) the malignant/benign ratio is balanced across
folds. Run training once per fold, then aggregate with scripts/aggregate_folds.py.

Usage:
    python scripts/make_folds.py \
        --manifest /content/drive/MyDrive/lidc_manifest.csv \
        --out-dir  /content/folds_strict \
        --n-splits 5 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make strict patient-level CV folds.")
    p.add_argument("--manifest", type=Path, required=True, help="Full manifest CSV from prepare_lidc.py")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to write fold manifests")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--benign-max", type=float, default=2.0, help="mean_malignancy <= this => benign")
    p.add_argument("--malignant-min", type=float, default=4.0, help="mean_malignancy >= this => malignant")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Strict filter: kept {len(df)}/{n_raw} nodules "
          f"(benign<= {args.benign_max}, malignant>= {args.malignant_min})")
    print(f"  benign={int((df['label'] == 0).sum())}  malignant={int((df['label'] == 1).sum())}")
    print(f"  patients={df['patient_id'].nunique()}")

    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    groups = df["patient_id"].values
    y = df["label"].values

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])
        assert not overlap, f"Fold {fold}: patient leakage ({len(overlap)} ids in both halves)"

        train_path = args.out_dir / f"train_fold{fold}.csv"
        val_path = args.out_dir / f"val_fold{fold}.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(
            f"fold {fold}: "
            f"train={len(train_df)} ({train_df['patient_id'].nunique()} pt, "
            f"{100 * train_df['label'].mean():.1f}% malignant) | "
            f"val={len(val_df)} ({val_df['patient_id'].nunique()} pt, "
            f"{100 * val_df['label'].mean():.1f}% malignant)"
        )

    print(f"\nWrote {args.n_splits} folds to {args.out_dir}")


if __name__ == "__main__":
    main()
