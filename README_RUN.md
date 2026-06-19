# CT-ATPT — How to Run (college GPU / fresh machine)

3D ViT with adaptive token pruning for LIDC-IDRI lung-nodule malignancy
classification. This guide gets training running on a fresh Linux GPU box.

---

## 1. What you need to copy onto the machine

1. **This `research/` folder** (all the code).
2. **The CT crops** — the `.npy` volume files (the bulk of the data).
3. **The master manifest CSV** — must contain columns:
   `volume_path, label, mean_malignancy, patient_id, center_z, center_y, center_x, radius`.

**Layout matters.** `volume_path` in the manifest is resolved **relative to the
manifest file's own directory** (absolute paths are used as-is). Keep the manifest
and the `.npy` crops in the same relative arrangement they had on Colab. Quick check:

```bash
python - <<'PY'
import csv, pathlib
m = pathlib.Path("PATH/TO/manifest.csv")
row = next(csv.DictReader(m.open()))
vp = pathlib.Path(row["volume_path"])
vp = vp if vp.is_absolute() else (m.parent / vp)
print("resolves to:", vp, "| exists:", vp.exists())
PY
```
If `exists: False`, the paths don't match this machine — fix the layout or rewrite
`volume_path` before training.

---

## 2. Environment

Python 3.10–3.12. With a CUDA GPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# 1) install torch matched to the box's CUDA (cu121 shown; check `nvidia-smi`)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) the rest
pip install -r requirements.txt
```

Sanity check:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Pretrained ViT-B/16 weights:** `--pretrained` makes torchvision download
ImageNet ViT-B/16 (~330 MB) to `~/.cache/torch/hub/checkpoints/` on first run.
- **Box has internet:** nothing to do.
- **Box is offline:** on a machine *with* internet run
  `python -c "import torchvision; torchvision.models.vit_b_16(weights='IMAGENET1K_V1')"`,
  then copy `~/.cache/torch/hub/checkpoints/vit_b_16-*.pth` to the same path on the GPU box.

---

## 3. Build the CV folds (once)

Strict labels (benign mean≤2, malignant≥4, ambiguous dropped) + patient-level
`StratifiedGroupKFold(5)`. Seed 42 reproduces the paper splits.

```bash
python scripts/make_folds.py \
  --manifest /path/to/manifest.csv \
  --out-dir  ./folds_strict \
  --n-splits 5 --seed 42
```
Produces `folds_strict/train_fold{0..4}.csv` and `val_fold{0..4}.csv`.

---

## 4. Train — soft-pruning, all 5 folds

Current best recipe (Phase 2, soft pruning). `$f` loops over folds 0–4:

```bash
for f in 0 1 2 3 4; do
  echo "===== Fold $f ====="
  python scripts/train_ct_atpt_ddp.py \
    --train-manifest ./folds_strict/train_fold${f}.csv \
    --val-manifest   ./folds_strict/val_fold${f}.csv \
    --output-dir     ./runs/soft_cv/fold${f} \
    --pretrained --pruning-mode soft \
    --pruning-warmup-epochs 10 --prune-ramp-epochs 15 --prune-target-keep 0.5 \
    --sparsity-weight 1.5 --soft-lambda-init -0.5 --gate-sharpness 4 \
    --depth 96 --height 96 --width 96 --patch-z 8 --patch-y 16 --patch-x 16 \
    --embed-dim 768 --transformer-depth 12 --heads 12 \
    --epochs 45 --warmup-epochs 5 --lr 1e-4 --weight-decay 0.05 \
    --batch-size 64 --grad-accum 1 --grad-clip 1.0 \
    --cls-loss ce --label-smoothing 0.05 --drop-path 0.1 \
    --tta --det-weight 0 --amp bf16 --best-metric roc_auc
done
```

**Baseline (no pruning)** — for comparison; swap `--pruning-mode soft …` and the
soft-only flags for just `--pruning-mode none`. Everything else stays the same.

### Tuning knobs
- **VRAM:** `--batch-size 64` suits an A100 (40 GB). Smaller card → drop to 32/16
  and optionally `--grad-accum 2` to keep the effective batch.
- **Pruning strength:** raise `--sparsity-weight` (→3–4) to push keep-ratio toward
  0.5; lower it / raise `--prune-target-keep` if AUC drops too much.

### What a healthy run looks like (per-step log)
- Epochs 0–9 (warmup): `keep=[1.0×12]`, `spar=0`, `abg≈[0.333,0.333,0.333]`.
- Epoch 10+: `keep` drops toward target, `spar>0`, `abg` drifts, `fallbacks=0`.
- `val_auc` should sit in ~0.85–0.92. A brief dip right at epoch 10 (prune onset)
  that recovers is normal; a flatline at ~0.5 that never recovers is a failure — stop.

---

## 5. Aggregate the 5-fold result

```bash
python scripts/aggregate_folds.py --runs-root ./runs/soft_cv
```
Prints per-fold and mean±std ROC-AUC etc. Point `--runs-root` at the folder that
*directly* contains the `fold*/best_checkpoint.pt` dirs (don't mix soft + baseline
runs in one folder, or it averages them together).

Evaluate a single checkpoint:
```bash
python scripts/eval_checkpoint.py \
  --checkpoint ./runs/soft_cv/fold0/best_checkpoint.pt \
  --val-manifest ./folds_strict/val_fold0.csv --tta --amp bf16
```

---

## 6. Reference numbers
- **Phase 1 baseline (no pruning):** ROC-AUC **0.870 ± 0.064** (strict 5-fold patient CV).
- Soft pruning must match this while actually dropping tokens. See `PROGRESS_LOG.md`
  for the full experiment history and rationale.

---

## Troubleshooting
- `FileNotFoundError: …/train_fold0.csv` → run step 3, or fix `--train-manifest` path.
- `FileNotFoundError` on a `.npy` → manifest `volume_path` doesn't resolve here (see §1).
- `torch.cuda.is_available() == False` → wrong torch build for the box's CUDA; reinstall.
- `weights_only`/`UnpicklingError` loading checkpoints → already handled in our scripts
  (`weights_only=False`); needs PyTorch ≥2.6 compatible code (this repo is).
- OOM → lower `--batch-size`, add `--grad-accum`.
```
