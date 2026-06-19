# CT-ATPT — Development & Experiment Log

A running record of what we tried, why, what we changed, and what happened. Newest entries appended at the bottom. Maintained across sessions.

**Project:** CT-ATPT — adaptive token-pruning 3D ViT for LIDC-IDRI lung-nodule malignancy classification.
**Execution:** code on Windows mount (`C:\Users\Aries\Documents\Codex\2026-05-15\research`), training on Google Colab A100. Claude writes code; user runs on Colab (re-zip + re-upload `research.zip` after each code change).

---

## Phase 0 — Starting point (before this effort)

- Best prior result: **ROC-AUC ≈ 0.8428** on a single strict-split (90 val samples), 768/12/12 ViT trained **from scratch** with focal loss.
- Known issues: ViTs "grok" (≈55-epoch flat plateau before learning from scratch); focal loss suppressed gradients and prevented learning on noisier splits; the token-pruning module collapsed to its fallback floor (`abg` frozen at `[0.333,0.333,0.333]`, keep-ratios pinned) and, being mask-based, produced **no real compute savings**.

---

## Phase 1 — Maximize backbone AUC (pruning set aside)

**Decision:** optimize a strong **pretrained ViT-B/16 backbone with pruning OFF**, treat pruning as a separate efficiency contribution to fix later. Headline number from **strict labels + 5-fold patient-level CV**.

**Rationale:** pruning was *hurting* AUC and isn't a real speedup yet; pretraining is the biggest lever (eliminates the grokking plateau — reached good AUC by epoch ~4 vs ~55 from scratch). AUC is a ranking metric, so threshold/calibration tweaks can't move it — only backbone/data/stability can.

### Code changes (all additive, toggleable; defaults preserve old behavior)
| File | Change |
|---|---|
| `ct_atpt/model.py` | Added `DropPath` (stochastic depth), `drop_path_rate` config, linear per-block schedule. |
| `ct_atpt/inference.py` (new) | `tta_average_proba` — flip TTA (8 axis-flip views, averaged softmax). |
| `scripts/train_ct_atpt_ddp.py` | Weight **EMA** (`--ema-decay`, best ckpt taken across raw/EMA), **TTA** at val (`--tta`), **label smoothing** (`--label-smoothing`), `--drop-path`. |
| `scripts/eval_checkpoint.py` | `--tta`. |
| `scripts/make_folds.py` (new) | Strict relabel (benign mean≤2 / malignant≥4, drop ambiguous) + `StratifiedGroupKFold(5)` on `patient_id`. |
| `scripts/aggregate_folds.py` (new) | Mean±std of headline metrics across fold `best_checkpoint.pt`. |
| (later) all `torch.load` | `weights_only=False` — PyTorch ≥2.6 default broke loading our checkpoints (they store an `args` dict with `pathlib.Path`). |

### Recipe
Pretrained 768/12/12, patch (8,16,16) → 432 tokens, 96³ crops, `--pruning-mode none`, CE + label-smoothing 0.05, drop-path 0.1, EMA 0.999, TTA, `--det-weight 0` (detection head is degenerate — centroid always 0.5), lr 1e-4, batch 16, 45 epochs.

### Result — **ROC-AUC = 0.870 ± 0.064** (5-fold patient CV)
| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| ROC-AUC | 0.924 | 0.787 | 0.814 | 0.914 | 0.910 |

PR-AUC 0.866 ± 0.064 · balanced-acc 0.796 ± 0.077 · best-threshold bal-acc 0.833 ± 0.056.

**Observations:**
- Big, trustworthy jump over the old single-split 0.84; now in the published LIDC range (CNNs ~0.85–0.92, ViTs ~0.87–0.93).
- **EMA never won** — `best_checkpoint` was `raw` in all 5 folds. `--ema-decay 0.999` is too slow over ~1.1k steps; the EMA never escapes its init blend. → drop or lower it.
- Sensitivity at the fixed 0.5 threshold is unstable across folds (0.75 ± 0.18) because the threshold isn't calibrated per fold; rely on AUC / threshold-tuned bal-acc.
- Training fully memorizes: train `cls` loss flatlines at the label-smoothing floor (0.117) within ~15 epochs → regularization headroom exists.

**This is the baseline the pruning must match.**

---

## Phase 2 — Make the adaptive pruning actually work (the paper's core idea)

### Why it always collapsed — two real blockers (not just hyperparameters)
1. **Threshold could only ever keep ≤50%.** `λ = softplus(λ_raw) ≥ 0`, so `τ = μ + λσ ≥ μ`. With `λ_raw=1.0` it tried to keep ~10% → tripped the min-keep fallback every step. No "gentle" setting was reachable.
2. **Scoring weights got no gradient.** The keep decision (`scores > τ` + top-k fallback) is non-differentiable, so the classification loss never taught `α/β/γ` or `λ` anything → `abg` frozen at `[0.333,0.333,0.333]` forever. Not "learned end-to-end."

### Fix — new differentiable `pruning_mode="soft"` (leaves `adaptive`/`none` untouched)
- Separate **unconstrained** `soft_lambda_raw` (can be negative → `τ` below mean → keep >50%).
- **Soft gate** `g = sigmoid(κ(S − τ))` multiplies token magnitudes — no hard removal, no fallback → fully differentiable, so `abg` and `λ` get gradient from the classification loss.
- **Keep-ratio curriculum:** target keep ramps 1.0 → `prune_target_keep` over `prune_ramp_epochs` after warmup; a sparsity penalty `relu(mean_gate − ρ)` drives it.
- Inference note: this is magnitude-gating (v1), not true token-dropping yet — real FLOP/latency savings come in a later step.
- New args: `--pruning-mode soft`, `--prune-target-keep`, `--prune-ramp-epochs`, `--sparsity-weight` (+ later `--soft-lambda-init`, `--gate-sharpness`).

### Soft-pruning v1 smoke test (fold 0) — `λ_init=-2.0`, `κ=10`, `sparsity-weight=0.5`
**Verdict: infrastructure works, pruning too weak.**
- ✅ `abg` MOVED `[0.333,0.333,0.333] → [0.332,0.341,0.326]` (β up, γ down) — weights finally learning.
- ✅ `fallbacks=0` throughout — no collapse.
- ✅ AUC held ~0.90 (peak 0.918 @ ep14) ≈ baseline 0.92 — pruning didn't hurt accuracy.
- ❌ Barely pruned: `keep ≈ 0.98` at end (target 0.5); `λ` crept −2.000 → −1.972.

**Root cause: gate saturation.** `λ_init=-2` pins gates at ~1.0; a saturated sigmoid has ~0 derivative, so the sparsity/cls gradients can't push gates (or `λ`) down. The signal vanishes exactly where it's needed.

### Modifications for v2 (pending run)
- Exposed `--gate-sharpness` (κ) and `--soft-lambda-init` as knobs (were hardcoded).
- New settings: **`κ=4`** (softer, gradient-carrying gates), **`λ_init=-0.5`** (gates start ~70% open, responsive), **`sparsity-weight=1.5`** (push harder to target).
- **`batch-size 64`** — A100 was massively underused at batch 16; bigger batch also smooths the batch-averaged sparsity signal.

**Success criteria for v2:** `keep` descends toward ~0.5 by the end (deep layers first), `λ` climbs meaningfully from −0.5, `abg` keeps drifting, AUC stays ~0.88–0.92.

### Soft-pruning v2 smoke test (fold 0) — `λ_init=-0.5`, `κ=4`, `sparsity-weight=1.5`, `batch=64` — **WORKING**
- ✅ Real pruning: `keep` drops from 1.0 (warmup) to **~0.65** at prune onset and holds (~35% pruned, vs v1's ~2%). `fallbacks=0`, `abg` drifted to `[0.331,0.335,0.334]`.
- ✅ AUC recovered to **~0.85** by the end (≈ the 0.87 no-prune baseline).
- ⚠️ Epoch-10 **shock**: AUC dipped 0.87→0.55 when pruning engaged, then recovered over ~15 epochs (gates jump open to ~65% instantly rather than ramping).
- ⚠️ Plateaued at keep≈0.65, not the 0.5 target; `λ` barely moved (−0.5→−0.498) → keep level is set by `λ_init`/`κ`, not driven by the schedule. The sparsity penalty (weight 1.5) is too weak to push `λ` up.

**Next dial:** raise `--sparsity-weight` to ~3–4 to reach keep≈0.5; optionally soften the onset. Then run full strict 5-fold and compare CV mean vs the 0.870 baseline.

---

## Phase 3 — Push backbone AUC further (overfitting is the bottleneck)

**Motivation:** train `cls` loss flatlines within ~15 epochs (Phase 1 finding) → the
model memorizes a small dataset. AUC is a ranking metric, so gains must come from
label quality, regularization/augmentation, and ensembling — not thresholds. Three
additive, toggleable levers (defaults preserve Phase 1 behavior):

### Code changes
| File | Change |
|---|---|
| `ct_atpt/data.py` | **Stronger augmentation** baked into `augment_volume`: random 90° in-plane (Y,X) rotations + isotropic random zoom (trilinear, restored to crop shape), on top of existing flips/noise/brightness. All label-preserving (centroid invariant). |
| `ct_atpt/data.py` | **Soft labels**: dataset now emits `soft_label` = `clamp((mean_malignancy − 1)/4, 0, 1)` (falls back to hard label if column absent). |
| `scripts/train_ct_atpt_ddp.py` | `--soft-labels` (soft-target CE from mean_malignancy; folds in label smoothing), `--mixup-alpha` (Beta MixUp on volume + target, applied pre-forward). Shared soft-target CE path; focal bypassed when either is on. Val metrics still use the **hard** label → honest AUC. |
| `scripts/ensemble_eval.py` (new) | Average softmax over N checkpoints on a common manifest (multi-seed on one split, or the 5 fold models on a shared held-out **test** set). Reports per-member AUC + ensemble AUC + Δ. |

### Rationale per lever
- **Soft labels** — LIDC labels are thresholded averages of 1–5 radiologist scores; hard `>3` injects the most noise exactly at the benign/malignant boundary. Training on the continuous score is a well-established LIDC AUC booster.
- **Stronger aug + MixUp** — directly attacks the 15-epoch memorization; rotations/zoom enlarge the effective dataset, MixUp smooths decision boundaries.
- **Ensembling** — averaging independent models is a reliable CV-AUC gain and a clean paper headline.

**Status:** implemented, not yet run. Suggested first run: re-run the Phase 1 recipe (pruning OFF, pretrained, CE+LS 0.05, drop-path 0.1, TTA) **+ `--soft-labels` + `--mixup-alpha 0.2` + `--augment`**, batch ≥ 16, across the 5 strict folds; then `ensemble_eval.py` over the fold checkpoints on a held-out test manifest. Compare CV mean vs the **0.870** baseline.

---

## Open items / backlog
- [ ] Run v2 soft-pruning smoke test (fold 0); confirm keep actually ramps to target with AUC intact.
- [ ] Full strict 5-fold soft-pruning run; compare CV mean vs the 0.870 backbone baseline.
- [ ] **True token dropping** (not magnitude masking) + measure real speedup / peak-memory vs no-pruning — required to back the paper's efficiency claim.
- [ ] Ablations for paper Table II: each of A/R/E off, fixed vs learned threshold, recycling on/off.
- [ ] Reconcile paper draft: it currently states focal loss (project lesson: CE) and a 384/6/6 config (runs use 768/12/12); fill `[RESULT]` placeholders with CV numbers.
