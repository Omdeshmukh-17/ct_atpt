"""LIDC-IDRI preprocessing: pylidc -> nodule crops (.npy) + manifest.csv

Maximum-speed version using ProcessPoolExecutor.

Each worker process:
  1. Initializes pylidc fresh (its own SQLAlchemy session — no cross-process issues).
  2. Re-queries the scan by ID.
  3. Calls scan.to_volume() — uses pylidc's tested DICOM loader.
  4. Extracts crops, saves .npy files.

Works on both Colab (Drive-mounted DICOMs) and local machines (SSD).

Usage:
  python scripts/prepare_lidc.py \
    --out-dir  /content/drive/MyDrive/lidc_processed \
    --manifest /content/drive/MyDrive/lidc_manifest.csv \
    --crop-size 96 --min-radiologists 3 --min-diameter-mm 3.0 \
    --exclude-ambiguous --workers 8
"""
from __future__ import annotations

# ── Python 3.12+ / NumPy 1.24+ compatibility patches ─────────────────────────
import configparser
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser  # type: ignore[attr-defined]

import numpy as np
for _alias, _builtin in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side functions (must be at module level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _lung_window(vol: np.ndarray,
                 low: float = -1000.0,
                 high: float = 400.0) -> np.ndarray:
    return ((np.clip(vol, low, high) - low) / (high - low)).astype(np.float32)


def _extract_crop(volume: np.ndarray,
                  centroid_zyx: list[float],
                  crop_size: int) -> np.ndarray:
    half   = crop_size // 2
    pad    = half + 5
    padded = np.pad(volume, pad, constant_values=0.0)
    cz, cy, cx = [int(round(centroid_zyx[i])) + pad for i in range(3)]
    c = padded[cz-half:cz+half, cy-half:cy+half, cx-half:cx+half]
    if c.shape != (crop_size, crop_size, crop_size):
        out = np.zeros((crop_size, crop_size, crop_size), np.float32)
        sz = min(c.shape[0], crop_size)
        sy = min(c.shape[1], crop_size)
        sx = min(c.shape[2], crop_size)
        out[:sz, :sy, :sx] = c[:sz, :sy, :sx]
        return out
    return c.astype(np.float32)


def _process_scan(job: dict) -> tuple[list[dict], list[str]]:
    """
    Worker: initialise pylidc, re-query scan by id, load volume, crop nodules.
    Runs in a fresh process so its SQLAlchemy session is independent.
    """
    # Re-apply patches inside the worker process
    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser
    import numpy as np
    for _a, _b in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if not hasattr(np, _a):
            setattr(np, _a, _b)

    import pylidc as pl

    rows: list[dict] = []
    warns: list[str] = []

    try:
        scan = pl.query(pl.Scan).filter(pl.Scan.id == job["scan_id"]).first()
        if scan is None:
            return [], [f"{job['patient_id']}: scan_id {job['scan_id']} not found"]
        vol = scan.to_volume().astype(np.float32)
    except Exception as e:
        return [], [f"{job['patient_id']} failed to load: {e}"]

    vol_n = _lung_window(vol)
    sp    = float(np.mean([job["slice_thickness"], job["pixel_spacing"]]))
    out   = Path(job["out_dir"])
    size  = job["crop_size"]

    for nod in job["nodules"]:
        npy_path = out / nod["fname"]
        r_norm   = float(np.clip((nod["mean_radius_mm"] / sp) / size, 0.0, 1.0))
        row = {
            "volume_path":      str(npy_path.resolve()),
            "label":            nod["label"],
            "center_z":         0.5,
            "center_y":         0.5,
            "center_x":         0.5,
            "radius":           round(r_norm, 5),
            "patient_id":       job["patient_id"],
            "mean_malignancy":  nod["mean_malignancy"],
            "num_radiologists": nod["num_radiologists"],
        }
        if npy_path.exists():
            rows.append(row)   # already done, still record in manifest
            continue
        try:
            c = _extract_crop(vol_n, nod["centroid"], size)
            np.save(npy_path, c)
            rows.append(row)
        except Exception as e:
            warns.append(f"{nod['fname']}: {e}")

    return rows, warns


# ─────────────────────────────────────────────────────────────────────────────
# Main process: Phase 1 (collect jobs) + Phase 2 (process pool)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",           type=Path,  required=True)
    parser.add_argument("--manifest",          type=Path,  required=True)
    parser.add_argument("--crop-size",         type=int,   default=96)
    parser.add_argument("--min-radiologists",  type=int,   default=3)
    parser.add_argument("--min-diameter-mm",   type=float, default=3.0)
    parser.add_argument("--exclude-ambiguous", action="store_true")
    parser.add_argument("--max-scans",         type=int,   default=None)
    parser.add_argument("--workers",           type=int,   default=0,
                        help="Worker processes (0 = cpu_count - 1).")
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = max(1, (os.cpu_count() or 4) - 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    import pylidc as pl

    # ── Phase 1: collect annotations only (NO ORM objects passed to workers) ──
    print("Phase 1: collecting annotation data from pylidc...")
    t0 = time.time()
    scans = pl.query(pl.Scan).all()
    if args.max_scans:
        scans = scans[: args.max_scans]
    total_scans_db = len(scans)

    jobs: list[dict] = []
    skipped_few_rads  = 0
    skipped_small     = 0
    skipped_ambiguous = 0

    for i, scan in enumerate(scans):
        nodules = []
        for nod_idx, anns in enumerate(scan.cluster_annotations()):
            if len(anns) < args.min_radiologists:
                skipped_few_rads += 1
                continue
            dias = [a.diameter for a in anns if a.diameter > 0]
            if not dias or float(np.mean(dias)) < args.min_diameter_mm:
                skipped_small += 1
                continue
            ms = float(np.mean([a.malignancy for a in anns]))
            if args.exclude_ambiguous and abs(ms - 3.0) < 1e-6:
                skipped_ambiguous += 1
                continue
            centroid = np.array([a.centroid for a in anns]).mean(0).tolist()
            nodules.append({
                "fname":            f"{scan.patient_id}_nod{nod_idx:02d}.npy",
                "label":            1 if ms > 3.0 else 0,
                "centroid":         centroid,
                "mean_radius_mm":   float(np.mean(dias)) / 2.0,
                "mean_malignancy":  round(ms, 2),
                "num_radiologists": len(anns),
            })

        if nodules:
            jobs.append({
                "scan_id":         scan.id,           # used by worker to re-query
                "patient_id":      scan.patient_id,
                "slice_thickness": float(scan.slice_thickness),
                "pixel_spacing":   float(scan.pixel_spacing),
                "nodules":         nodules,
                "out_dir":         str(args.out_dir),
                "crop_size":       args.crop_size,
            })

        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{total_scans_db}  | usable: {len(jobs)}")

    total_scans   = len(jobs)
    total_nodules = sum(len(j["nodules"]) for j in jobs)
    print(f"\n  Scans with nodules : {total_scans}")
    print(f"  Nodules to extract : {total_nodules}")
    print(f"  Skipped (few rads) : {skipped_few_rads}")
    print(f"  Skipped (small)    : {skipped_small}")
    print(f"  Skipped (ambiguous): {skipped_ambiguous}")
    print(f"  Phase 1 time       : {time.time()-t0:.1f}s")

    if not jobs:
        print("\nNothing to extract.")
        return

    # ── Phase 2: parallel process pool ────────────────────────────────────────
    print(f"\nPhase 2: extracting crops with {args.workers} processes...")
    t1 = time.time()

    all_rows: list[dict] = []
    done_scans = 0

    # Surface only one global warning filter so per-scan errors don't spam
    warnings.filterwarnings("once")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_scan, job): job["patient_id"] for job in jobs}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                rows, warns = future.result()
            except Exception as e:
                rows, warns = [], [f"{pid}: {e}"]

            all_rows.extend(rows)
            done_scans += 1
            for w in warns:
                warnings.warn(w)

            elapsed = time.time() - t1
            rate    = done_scans / elapsed if elapsed > 0 else 0.0
            eta_s   = (total_scans - done_scans) / rate if rate > 0 else 0.0
            print(f"\r  [{done_scans}/{total_scans}] {100*done_scans/total_scans:5.1f}%"
                  f"  | nodules: {len(all_rows)}"
                  f"  | {rate:4.1f} scans/s"
                  f"  | ETA {int(eta_s//60):02d}:{int(eta_s%60):02d}",
                  end="", flush=True)

    print(f"\n\nPhase 2 time: {(time.time()-t1)/60:.1f} min")

    # ── write manifest ────────────────────────────────────────────────────────
    fields = ["volume_path", "label", "center_z", "center_y", "center_x",
              "radius", "patient_id", "mean_malignancy", "num_radiologists"]
    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(1 for r in all_rows if r["label"] == 1)
    neg = len(all_rows) - pos
    print(f"\n{'='*55}")
    print(f"Manifest saved : {args.manifest}")
    print(f"Total nodules  : {len(all_rows)}")
    print(f"  Malignant (1): {pos}  ({100*pos/max(1,len(all_rows)):.1f}%)")
    print(f"  Benign    (0): {neg}  ({100*neg/max(1,len(all_rows)):.1f}%)")
    print(f"Total time     : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
