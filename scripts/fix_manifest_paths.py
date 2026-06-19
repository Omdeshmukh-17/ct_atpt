"""Fix manifest CSV paths from Colab/Drive to local Windows paths.

Usage:
    python scripts/fix_manifest_paths.py \
        --manifest data/train_manifest.csv \
        --old-prefix /content/drive/MyDrive/lidc_processed \
        --new-prefix data/lidc_processed

    python scripts/fix_manifest_paths.py \
        --manifest data/val_manifest.csv \
        --old-prefix /content/drive/MyDrive/lidc_processed \
        --new-prefix data/lidc_processed
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite volume_path in manifest CSVs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-prefix", type=str, required=True,
                        help="Path prefix to replace (e.g. /content/drive/MyDrive/lidc_processed)")
    parser.add_argument("--new-prefix", type=str, required=True,
                        help="Replacement prefix (e.g. data/lidc_processed)")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        fieldnames = list(reader.fieldnames)
        for row in reader:
            old = row["volume_path"]
            row["volume_path"] = old.replace(args.old_prefix, args.new_prefix)
            # Normalize slashes for the current OS
            row["volume_path"] = str(Path(row["volume_path"]))
            rows.append(row)

    with args.manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} paths in {args.manifest}")
    print(f"  '{args.old_prefix}' → '{args.new_prefix}'")
    if rows:
        print(f"  Example: {rows[0]['volume_path']}")


if __name__ == "__main__":
    main()
