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

### Soft-pruning v3 — quantile-driven threshold (code fix)
**Diagnosis of the v2 stall (both symptoms, one cause):** the threshold `τ = μ + λσ`
depended on the weak `relu(mean_keep − ρ)` penalty to push `λ`, and that penalty lost
every tug-of-war with the classification loss. So (a) keep-ratio parked at the
`λ_init`/`κ` operating point (~0.65) instead of reaching the 0.5 target, and (b)
because pruning never really bit, `abg` got no meaningful gradient and stayed ~0.333.

**Fix (`ct_atpt/model.py::_prune_soft`):** drive the threshold from the retention
schedule directly. Each step, cut at the **`(1 − ρ)` per-sample quantile** of the
scores (`τ = quantile(scores.detach(), 1−ρ)`), so ~ρ of the soft gates open *by
construction*, independent of `λ`. `ρ` still ramps 1.0 → `prune_target_keep`. The
gate stays a soft sigmoid, so `abg` still receives gradient through `scores`.
`soft_lambda_raw` is kept as a bounded learnable offset (`tanh(λ)·0.5·σ`) so it stays
in the graph without breaking the guarantee. The old sparsity term became a mild
two-sided `(mean_keep − ρ)²` regulariser (quantile already enforces the ratio).
Cites: retention-rate scheduling in EViT / DynamicViT.

**Bonus:** at prune onset (frac=0, ρ=1.0) the quantile keeps ~100%, so the v2
epoch-10 AUC shock disappears — pruning eases in smoothly.

**Validation (pure-python sim, no GPU):** realised keep-ratio tracks target ρ almost
exactly — target 0.50 → 0.500, 0.75 → 0.750, 1.0 → 0.998 (both κ=4 and κ=8). Compiles.

### Soft-pruning v3 smoke test (fold 0) — **CATASTROPHIC COLLAPSE, reverted**
Ran with quantile threshold, target_keep=0.5. Warmup (ep 0–9, no pruning) trained
fine to AUC 0.81. **The instant pruning turned on (ep 10) it collapsed to AUC ~0.49
and never recovered** — even at ep 10 where keep=0.998 (essentially nothing pruned).
cls loss jumped 0.51→0.76 and stuck at ln2 (~0.69, random); model predicted a single
class for the rest of the run. abg barely moved.

**Root cause:** the detached quantile threshold scale-shocks the pretrained backbone.
At ρ=1.0 the cut sits at the *min* score, so the soft gate multiplies EVERY token by
`sigmoid(κ(score−min)) ∈ [0.5, 0.83]` — a sudden ~30% activation drop at onset — and,
because tau is detached (not the self-centring μ+λσ), the entropy loss can corrupt the
attention/energy features to force decisive gates. v2's smooth μ+λσ form avoided both.

### Soft-pruning v4 — lambda SCHEDULE on the v2 threshold (current)
**Fix:** revert to v2's stable `τ = μ + λσ` (smooth, self-centring — never collapsed),
but *schedule λ* from `lam_start=-2.0` (keep ~0.98, gentle onset) to
`lam_end = ndtri(1 − target_keep)` (keep ~target) over the ramp. Small learnable
residual `tanh(soft_lambda_raw)·0.25`. Sparsity reverted to one-sided
`relu(mean_keep − target)`. This walks keep down smoothly instead of shock-cutting.
**Sim:** keep 0.98→0.50 smoothly; onset mean-gate 0.75 (gentler than the stable v2's
0.58) → no onset shock. Compiles.
**Plan:** smoke-test at `--prune-target-keep 0.6` first (≈40% prune, safely within the
v2-proven regime) to confirm stability + AUC hold, then decide 0.5 vs 0.6 for 5-fold.
**Status:** awaiting v4 smoke-test.

### Code-review fixes (pre-smoke-test sweep)
1. **`scripts/eval_checkpoint.py` + `scripts/ensemble_eval.py` — eval used wrong soft-pruning config.**
   Both rebuilt `CTATPTConfig` from saved args but dropped `gate_sharpness`,
   `prune_target_keep`, `prune_ramp_epochs`, `soft_lambda_init`, `drop_path`.
   `gate_sharpness` is a plain float (NOT in the state_dict) and
   `prune_target_keep` sets the eval-time λ schedule — so any soft-mode
   checkpoint trained with non-default κ/keep would silently evaluate with the
   wrong gates (wrong AUC). Now passed through.
2. **`ct_atpt/data.py` — augmentation corrupted det targets.** 90° rotations and
   random zoom moved the nodule but left `det_values` untouched (flips already
   updated them). Now: rotation maps `(cy,cx) → (1−cx,cy)` per 90°; zoom maps
   `c → 0.5+(c−0.5)·s` and `r → r·s`. Rotation also guarded to square Y/X crops
   (`np.rot90` would change tensor shape otherwise). Numerically validated
   (voxel-marker sim: rot90 exact, zoom within 1-voxel NN rounding).
### Adaptive-mode v5 — staged, budgeted, straight-through (rewrite)
**Diagnosed from the stuck logs** (`AUC 0.50`, `keep=[0.502 ×12]`, `abg=[0.333…]`,
fallback ~30%): the adaptive mode had two structural bugs.
1. `τ = μ + softplus(λ)σ ≥ μ` ⇒ every block keeps ≤ ~half of its REMAINING
   tokens; applied after all 12 blocks the cut compounds (0.5 → 0.25 → …) and
   slams into the `max(64, 25%·T)=1024` floor within two blocks ⇒ the fallback
   hard-top-k fires for every sample at every block. Hard top-k passes **no
   gradient** to `importance_logits`/`λ` ⇒ abg frozen at softmax-of-zeros
   uniform, keep pinned at the floor — "adaptive" degenerated to fixed-ratio.
2. Onset: 100% → ~25–30% tokens in one epoch, incl. cuts at blocks 1–2 where
   attention is uninformative ⇒ pretrained backbone collapses to one class
   (AUC 0.5). (Soft mode's `keep=0.502` flatline is the same disease: λ→0 pins
   frac(scores>μ+λσ) at exactly 0.5 for every sample/block, content-free.)

**v5 fix (`ct_atpt/model.py`, adaptive mode):**
- **Staged pruning (EViT-style):** prune only at blocks {D/4, D/2, 3D/4}
  (= {3,6,9} for depth 12); other blocks just update the rollout.
- **Scheduled cumulative budget:** ρ(epoch) ramps 1.0 → `prune_target_keep`
  after warmup; stage j clamps the keep count to ±10% of the ABSOLUTE track
  `ρ^(j/S)·T`. Anchoring on the cumulative track (not per-stage relative)
  provably prevents band-drift compounding (sim: relative band drifted to
  0.26 at ρ=0.5; cumulative tracks 0.45–0.55 for any λ).
- **Per-sample adaptivity kept:** count above `τ = μ + λσ` decides each
  sample's keep within the band; λ now **unconstrained** (init −1.5 ⇒ onset
  keep ~0.9, no shock). Sim on heavy-tailed (lognormal) scores: keep-fraction
  std 0.09, corr +0.40 with score concentration ⇒ genuinely content-adaptive
  (Gaussian scores are the degenerate case).
- **Straight-through gate:** kept tokens stay at FULL strength in forward
  (hard dropping = token dropout, no activation-scale shock on the pretrained
  backbone) while backward routes gradient through the soft gate into scores
  ⇒ abg/λ finally learn. Recycling into CLS unchanged.
- `--prune-target-keep` / `--prune-ramp-epochs` now drive adaptive mode too;
  `aux["lambda"]` logs raw λ (unconstrained).
**Expected log signature when healthy:** keep ≈ 1.0 at warmup end, walking to
[0.45, 0.55] over the ramp; keep varies across samples; abg drifts off 0.333;
fallback counts only band-clamps (nonzero early is fine, should shrink as λ
learns). **Status:** awaiting v5 smoke-test (fold 0, `--pruning-mode adaptive`).

### v5 smoke test (fold 0, target_keep 0.6) — **STABLE, best AUC 0.9003**
No collapse at pruning onset (ep10: 0.786→0.759, immediate recovery). Keep
tracked the cumulative ramp exactly (1.0 → [0.925, 0.782, 0.66] at blocks
3/6/9, frozen from ep25). Best val AUC **0.9003** (ep16) with 34% of tokens
pruned — above the 0.870 pruning-off baseline (caveat: n_val=90, plateau
~0.86). abg moved off uniform for the first time: [0.340, 0.343, 0.316].
**Remaining gap:** λ stayed ≈ init (−1.48) ⇒ every sample clamped at the band
CEILING (0.66 = 1.1·0.6, fallbacks≈188/192) ⇒ pruning was scheduled, not yet
per-sample adaptive — the clamp, not τ, decided the count, so the cls loss
gave λ no gradient incentive.

### v5.1 — budget-centring sparsity loss (adaptive mode)
`_prune_after_block` now emits `sparsity = mean_b (soft_keep_frac_b −
cum_target/n_active_b)²` on the PRE-clamp soft gate fraction. Pulls τ onto the
budget track ⇒ clamp goes quiet ⇒ per-sample counts decided by τ within the
band (adaptivity can express). Uses the existing `--sparsity-weight` plumbing
(default 0.5). **Watch on rerun:** λ should move (≈ −1.0 for 84%/stage),
fallbacks should shrink, keep should VARY across samples.

### v5.1 smoke test — TWO findings, neither fatal
1. **The script had NO seed control.** This run's warmup (identical code path,
   pruning off, spar not in loss) hit only AUC 0.58 by ep9 vs 0.82 in the v5
   run — pure init/shuffle luck. All cross-run comparisons so far carry this
   noise. **Fix: `--seed` (default 42)** now seeds python/numpy/torch/cuda and
   the DistributedSampler. Run-to-run comparisons must pin it from now on.
2. **Learnable λ lost the tug-of-war again:** drifted −1.500→−1.545 (wrong
   direction; cls loss prefers keeping everything; quadratic sparsity on a
   0.08 gap × weight 0.5 is negligible). Fallbacks stayed ~180 ⇒ still
   clamp-driven, zero adaptivity.

### v5.2 — schedule λ in adaptive mode (soft-v4 recipe)
`λ = ndtri(1 − ρ(ep)^(1/S)) + 0.25·tanh(lambda_raw)`, residual init 0 ⇒ τ
starts exactly on the budget schedule, inside the band by construction ⇒
clamp goes quiet without any loss-vs-loss arm wrestling. Sparsity loss kept
as a mild centring regulariser. `aux["lambda"]` reports the effective value.
`lambda_raw` re-init −1.5 → 0.0 (it is now a bounded residual, not the level).
**Watch:** lambda print should walk ≈ −3.7 → −1.0 over the ramp; fallbacks
low; keep varying across batches. Rerun with `--seed 42` and compare v5.2 vs
a pruning-off run at the SAME seed before judging AUC.

### v5.2 smoke test — **PASSED all criteria** (fold 0, keep 0.6, seed 42)
λ walked −3.72 → −1.02 exactly on schedule; keep ramped to [0.86, 0.75, 0.63]
(inside band, varying across batches); fallbacks ~50–60/192 (guardrail, not
driver); abg → [0.343, 0.323, 0.334]; **best AUC 0.8681**, plateau ~0.85, no
onset collapse. Adaptive pruning mechanism confirmed working end-to-end.

### 5-fold CV results (seed 42, keep 0.6) — **HEADLINE NUMBERS**
**Adaptive v5.2:** ROC-AUC **0.8370 ± 0.0529** (folds: 0.868, 0.915, 0.790,
0.803, 0.809), PR-AUC 0.822 ± 0.104, best-bal-acc 0.797 ± 0.055 — with ~37%
of tokens pruned.
**Baseline (pruning off, same seed):** folds 0–1 = 0.843, 0.904 — adaptive
BEAT the identical-seed baseline on both completed folds (token-dropout
regularization effect). Baseline fold2 `best_checkpoint.pt` was truncated
(EOFError, run died mid-save or Drive sync) → rerun fold2; folds 3–4 to
verify. `aggregate_folds.py` now skips corrupt checkpoints with a warning
instead of crashing.
**Next:** complete baseline folds 2–4 → paper Table I; then true token
dropping for the efficiency claim; then ablations (Table II).

3. **`ct_atpt/model.py` — λ logging + ndtri edge case.** Effective scheduled λ
   factored into `_soft_effective_lambda()`; `aux["lambda"]` now reports it
   (was raw `soft_lambda_raw`, which would have read ≈−2.0 all run during the
   v4 smoke test). `ndtri` argument clamped so `target_keep=1.0` can't produce
   −inf → NaN gates. No behavioral change to training at default settings.

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
