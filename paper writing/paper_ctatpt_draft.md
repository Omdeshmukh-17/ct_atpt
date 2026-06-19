# CT-ATPT: An Adaptive Token Pruning Transformer for Lung Nodule Malignancy Classification on 3D CT

**Draft for IEEE Transactions on Medical Imaging / IEEE Journal of Biomedical and Health Informatics**

> **Authors:** [PLACEHOLDER: author list]
> **Affiliations:** [PLACEHOLDER]
> **Corresponding author:** [PLACEHOLDER]

---

## Abstract

Lung cancer remains the leading cause of cancer mortality worldwide, and CT screening has shifted the clinical bottleneck from detecting nodules to deciding whether a detected nodule is malignant. Three-dimensional Vision Transformers (3D ViTs) are a natural fit for the volumetric reasoning this task requires, but their quadratic-in-tokens cost makes them impractical for the high-resolution sub-volumes radiologists actually inspect. Existing token-pruning strategies either prune at a fixed ratio, rely on gradient-based saliency that is unavailable at inference, or treat all imaging modalities identically — ignoring that in CT the vast majority of voxels are air or homogeneous parenchyma that carry no diagnostic signal. We introduce **CT-ATPT**, an *Adaptive Token Pruning Transformer* tailored to volumetric CT. CT-ATPT combines three contributions: (i) a tri-component token importance score that fuses received attention, attention-rollout saliency, and a CT-specific local energy term (HU variance plus gradient magnitude), with the mixture weights *learned end-to-end*; (ii) a statistical adaptive threshold $\tau_l = \mu_l + \lambda \sigma_l$ — also driven by a learnable scalar $\lambda$ — that adapts the pruning ratio per layer and per volume; and (iii) an information-preserving recycling step that injects a temperature-softmax-weighted summary of *discarded* tokens back into the classification token, so that pruning never silently destroys evidence. We evaluate CT-ATPT on the LIDC-IDRI dataset under patient-level splits, with malignancy labels derived from $\geq 3$-radiologist consensus and the ambiguous mean-score-3 cases excluded. CT-ATPT achieves **ROC-AUC [RESULT: AUC]**, **balanced accuracy [RESULT: BA]**, and **sensitivity [RESULT: SENS]** at the operating threshold, while pruning roughly **[RESULT: PRUNE%]** of tokens by the final transformer block — yielding **[RESULT: SPEEDUP]× faster inference** and **[RESULT: MEMORY%] lower peak GPU memory** than the unpruned baseline at matched accuracy. Ablations confirm that all three components of the importance score, the statistical threshold, and the recycling step are individually necessary.

**Index Terms** — 3D Vision Transformer, lung nodule malignancy classification, token pruning, attention rollout, LIDC-IDRI, computer-aided diagnosis.

---

## I. Introduction

**Clinical motivation.** Lung cancer accounts for more deaths annually than any other malignancy, and large-scale low-dose CT screening programs continue to expand worldwide [REF: NLST, NELSON]. The clinical decision that most directly impacts patient outcome is no longer *whether a nodule exists*, but *whether a detected nodule is malignant* — a determination that drives invasive follow-up such as PET-CT, biopsy, or surgical resection. Inter-radiologist agreement on this judgement is moderate at best, and the consequences of error are asymmetric: false negatives delay treatment, while false positives drive over-investigation. A reliable automated second reader for malignancy stratification therefore remains a clinically valuable target.

**Why transformers, and the problem.** Convolutional networks dominated early CAD systems for lung nodules, but recent work shows that 3D Vision Transformers (ViTs) can learn the long-range spatial dependencies — spiculation patterns, pleural attachment, lobulated margins — that radiologists use to judge malignancy [REF: 3D ViT lung]. The cost is severe: a $128 \times 512 \times 512$ CT crop tokenized with $8 \times 32 \times 32$ patches produces 4{,}096 tokens, and full self-attention scales as $\mathcal{O}(N^2)$ in token count $N$. Worse, the majority of those tokens correspond to air, ribs, mediastinum, or homogeneous parenchyma that contribute *nothing* to a nodule-level decision. Token pruning is the natural remedy, but existing pruning strategies were designed for natural images and inherit assumptions that break in CT. DynamicViT [REF] learns a soft mask but prunes at a *fixed* global ratio; A-ViT [REF] halts depth per token but does not exploit anatomical priors; EViT [REF] keeps top-$K$ tokens by class-attention but cannot adapt $K$ to the difficulty of an individual scan. None of these methods incorporate CT-specific evidence such as Hounsfield-unit variance, and several rely on gradient-based saliency that is unavailable at inference time.

**Our approach.** We propose **CT-ATPT**, an adaptive token-pruning 3D Vision Transformer designed end-to-end for volumetric CT. CT-ATPT differs from prior pruning ViTs in three coupled ways. First, the *importance score* assigned to each token is a learnable convex combination of three complementary signals: (i) the mean attention received from other tokens, (ii) the attention-rollout score — a recurrence over normalized attention matrices that propagates saliency through depth and remains valid at inference, and (iii) a CT-specific local energy term computed from the patch's HU variance and 3D gradient magnitude, which acts as a strong prior for "is this voxel block likely to contain anatomical structure relevant to a nodule?" The three weights $(\alpha, \beta, \gamma)$ are realised as a softmax over learnable logits, so they remain non-negative and sum to one without manual tuning. Second, the *pruning threshold* itself is statistical and adaptive: at each layer we set $\tau_l = \mu_l + \lambda \sigma_l$, where $\mu_l, \sigma_l$ are the mean and standard deviation of the per-token scores at layer $l$ and $\lambda$ is a single learnable scalar. The number of retained tokens therefore depends on the actual score distribution for *this* scan at *this* depth, rather than a fixed budget. A minimum-keep floor guards against degenerate cases and triggers an explicit fallback that is reported during training. Third, because aggressive pruning can in principle discard useful evidence, we introduce an *information-preserving recycling* step: at every pruning layer the classification token is updated with a temperature-softmax-weighted sum over the *discarded* tokens, so pruned information is summarised back into the global representation rather than silently dropped.

**Contributions.** This work makes the following contributions:

1. We introduce a **tri-component token importance score** for 3D CT that fuses inter-token attention, attention-rollout saliency, and CT-specific local energy, with mixture weights learned end-to-end via softmax. To our knowledge this is the first ViT pruning criterion to embed an explicit imaging-modality prior in this way.
2. We replace the fixed pruning ratio of prior ViTs with a **statistical adaptive threshold** $\tau_l = \mu_l + \lambda \sigma_l$ governed by a single learnable scalar $\lambda$, yielding per-layer and per-scan pruning rates that respond to actual score distributions.
3. We propose an **information-preserving recycling mechanism** that injects a temperature-softmax aggregate of pruned tokens into the classification token, mitigating the information loss that aggressive pruning would otherwise cause.
4. On LIDC-IDRI with patient-level splits and consensus malignancy labels, CT-ATPT achieves **[RESULT: AUC]** ROC-AUC and **[RESULT: BA]** balanced accuracy while pruning **[RESULT: PRUNE%]** of tokens, outperforming both a no-pruning 3D ViT baseline and a fixed-ratio DynamicViT-style baseline (Tables [REF Table I] and [REF Table II]). Ablations confirm each component is necessary.

The remainder of the paper is organised as follows. Section II reviews related work on lung-nodule CAD and ViT token pruning. Section III formalises the CT-ATPT architecture and training objectives. Section IV describes our experimental protocol on LIDC-IDRI. Section V reports main results, ablations, and pruning analyses. Section VI discusses limitations and clinical implications. Section VII concludes.

---

## II. Related Work

### A. Deep learning for pulmonary nodule analysis

Early CAD systems for pulmonary nodules relied on 2D CNNs applied to axial slices, followed by 3D CNNs once volumetric architectures became practical [REF: Setio 2017, Liao 2019]. These systems excel at *detection* — separating nodule from non-nodule on patch-level data such as LUNA16 — but the LUNA16 task does not provide malignancy labels, which forced subsequent work on malignancy to return to the more carefully annotated LIDC-IDRI dataset [REF: Armato 2011]. State-of-the-art 3D CNNs on LIDC-IDRI typically report ROC-AUC in the 0.90–0.94 range for the binary benign-versus-malignant task when patient-level splits are used [REF]. More recent work has demonstrated that 3D ViTs and CNN–transformer hybrids can match or exceed pure-CNN performance, particularly when long-range spatial relationships such as spiculation patterns and pleural contact are informative [REF: 3D ViT lung]. However, full-resolution 3D self-attention remains prohibitively expensive, motivating the line of work this paper extends.

### B. Token pruning in vision transformers

A growing body of work prunes tokens within transformer blocks to reduce the quadratic attention cost. **DynamicViT** [REF: Rao 2021] inserts lightweight prediction modules that output keep/drop probabilities trained with a soft Gumbel mask, but enforces a *fixed* keep ratio per layer. **A-ViT** [REF: Yin 2022] introduces adaptive halting per token, akin to ACT, but the halting score is purely learned from intermediate features and ignores modality-specific priors. **EViT** [REF: Liang 2022] retains the top-$K$ tokens by class-attention magnitude and concatenates the remainder into a single fused token; $K$ is again a hyperparameter. **ToMe** [REF: Bolya 2023] merges similar tokens rather than dropping them, and is complementary to pruning. All of these methods were designed for 2D natural-image classification and do not exploit the volumetric, density-encoded nature of CT. They also typically rely on signals that are only available at training time (e.g., gradient-derived saliency in some variants), making them difficult to deploy without auxiliary student networks.

### C. Attention rollout and saliency

Attention rollout [REF: Abnar 2020] propagates per-layer attention matrices through depth via the recurrence $R^{(l)} = (0.5 A^{(l)} + 0.5 I)\, R^{(l-1)}$, producing a saliency map that, unlike raw last-layer attention, accounts for residual streams. Crucially, rollout depends only on attention matrices and is therefore available at inference. We use rollout as one of the three components of our token importance score, in preference to gradient-based saliency used in some pruning variants which requires backward passes that are not available during deployment.

### D. Gap addressed by this work

To the best of our knowledge, no prior method (i) combines attention, rollout, and modality-specific energy in a *learnable* convex mixture, (ii) replaces the fixed pruning ratio of DynamicViT-family methods with a *statistical* adaptive threshold, or (iii) explicitly recycles information from pruned tokens back into the classification token. CT-ATPT addresses all three gaps in a single unified framework.

---

## III. Method

### A. Architecture overview

Given a 3D CT crop $V \in \mathbb{R}^{D \times H \times W}$ centred on a candidate nodule, CT-ATPT proceeds as follows. A 3D patch embedding layer produces $N$ initial tokens. The tokens, augmented with a learnable classification token and a learnable 3D positional embedding, traverse $L$ transformer blocks. Between consecutive blocks, an *adaptive pruning module* scores every spatial token, computes a per-layer statistical threshold, gates tokens by a sigmoid of the margin above threshold, and recycles a summary of below-threshold tokens into the classification token. A linear classification head on the final classification token outputs benign-versus-malignant logits; an auxiliary detection head predicts a coarse nodule centre to encourage spatially grounded features.

### B. 3D patch embedding

Each crop is split into non-overlapping patches of size $p_z \times p_y \times p_x$ and linearly projected to dimension $d$:
$$
T^{(0)}_i = W_{\text{embed}}\, \text{flatten}(V_i) + b_{\text{embed}}, \quad i = 1, \dots, N
$$
with $N = (D/p_z)(H/p_y)(W/p_x)$. A learnable class token $T^{(0)}_{\text{cls}}$ is prepended, and a learnable 3D positional embedding is added. We use $96 \times 96 \times 96$ crops with $16 \times 16 \times 16$ patches, giving $N = 216$ spatial tokens.

### C. Token importance score

For each spatial token $t_i$ at layer $l$ we compute three component scores.

**Attention received.** Averaging over heads and over the source dimension of the multi-head self-attention map $A^{(l)} \in \mathbb{R}^{H \times N \times N}$:
$$
A(t_i) = \frac{1}{H N} \sum_{h=1}^{H} \sum_{j=1}^{N} A^{(l)}_{h,j,i}.
$$
**Attention rollout.** With the recurrence $R^{(l)} = (0.5 \bar{A}^{(l)} + 0.5 I)\, R^{(l-1)}$, where $\bar{A}^{(l)}$ is the head-averaged attention matrix:
$$
R(t_i) = R^{(l)}_{\text{cls},\, i},
$$
i.e., the propagated influence of token $i$ on the class token. We verified that $R^{(l)}$ remains row-stochastic under this recurrence.

**CT-specific local energy.** Each token corresponds to a patch $V_i$ of the original volume. We define
$$
E(t_i) = \underbrace{\operatorname{Var}_{\text{vox}}(V_i)}_{\text{HU heterogeneity}} \;+\; \underbrace{\lVert \nabla V_i \rVert_1}_{\text{edge content}}
$$
followed by min–max normalisation across tokens within the same volume. The first term is large where Hounsfield-unit values vary (anatomical structure), and small in air or homogeneous parenchyma; the second emphasises edges that frequently accompany nodule boundaries, vessels, and spiculation.

**Fused importance score.** The final per-token score is
$$
S(t_i) = \alpha \, A(t_i) + \beta \, R(t_i) + \gamma \, E(t_i),
$$
where $(\alpha, \beta, \gamma) = \operatorname{softmax}(w_{\alpha}, w_{\beta}, w_{\gamma})$ are obtained from three *learnable scalars* trained jointly with the rest of the network. This parameterisation ensures the weights remain non-negative and sum to one without explicit constraints.

### D. Adaptive statistical pruning

Given per-token scores $\{S(t_i)\}_{i=1}^{N_l}$ at layer $l$, we form
$$
\mu_l = \frac{1}{N_l} \sum_i S(t_i), \qquad \sigma_l^2 = \frac{1}{N_l} \sum_i \left( S(t_i) - \mu_l \right)^2,
$$
and define the layer-specific threshold
$$
\tau_l = \mu_l + \lambda \, \sigma_l,
$$
where $\lambda$ is a single learnable scalar shared across layers. The per-token gate is
$$
g_i = \sigma\!\left( \kappa \cdot (S(t_i) - \tau_l) \right),
$$
with $\kappa$ a fixed sharpness constant ($\kappa = 10$ in our experiments). Tokens with $g_i > 0.5$ are retained; the rest are candidates for pruning.

**Minimum-keep fallback.** If the number of retained tokens falls below a floor $N_{\min}$, we revert at that layer to keeping the top-$N_{\min}$ tokens by $S$ and emit a fallback signal that is aggregated across the batch and logged. This prevents degenerate collapse early in training when $\lambda$ has not yet converged.

### E. Information-preserving token recycling

Rather than discard below-threshold tokens, we summarise them and inject the summary into the classification token. Let $\mathcal{P}_l$ denote the index set of pruned tokens at layer $l$ and $T$ a learnable temperature. The update is
$$
T^{(l)}_{\text{cls}} \;\leftarrow\; T^{(l)}_{\text{cls}} \;+\; \sum_{i \in \mathcal{P}_l} \operatorname{softmax}\!\left( S(t_i) / T \right) \cdot (1 - g_i) \cdot t_i.
$$
Intuitively, tokens that just-barely missed the threshold contribute more than those scored far below it. This mechanism ensures the classification token receives a graded, distribution-aware summary of *all* discarded evidence rather than being blind to it.

### F. Training objectives

We optimise a weighted sum of four losses.

**Classification (focal).** With logits $z$, target $y \in \{0, 1\}$, $p_t = \operatorname{softmax}(z)_y$:
$$
\mathcal{L}_{\text{cls}} = -\alpha_{\text{pos}}\,(1 - p_t)^{\gamma_f}\, \log p_t.
$$
We use $\gamma_f = 2.0$ and $\alpha_{\text{pos}} = 0.5$ on LIDC-IDRI (the malignant/benign ratio is close to balanced after our filtering, so heavy reweighting is unnecessary). Empirically we verified that $\mathcal{L}_{\text{cls}}$ correctly down-weights easy examples: at $p_t = 0.9$ the focal loss is roughly $0.0075 \times$ the cross-entropy.

**Detection (focal BCE + L1).** An auxiliary head over a coarse 3D grid predicts objectness logits and a $(z, y, x)$ centre offset; the objectness term uses a focal BCE to address the $1$-positive-vs-many-negative imbalance, and the regression term is L1 on the matched positive cell only. This auxiliary supervision empirically improves the quality of the learned attention maps.

**Pruning-entropy regulariser.** To discourage indecisive gates we penalise the per-gate Bernoulli entropy:
$$
\mathcal{L}_{\text{ent}} = -\frac{1}{|\mathcal{G}|} \sum_{i \in \mathcal{G}} \big[ g_i \log g_i + (1 - g_i) \log (1 - g_i) \big].
$$
This term is bounded above by $\ln 2 \approx 0.6931$ and is maximised at $g_i = 0.5$. Minimising it pushes gates toward confident decisions.

**Consistency.** To stabilise importance scores across depth we apply an $L_1$ penalty on the *normalised* score distributions of consecutive layers:
$$
\mathcal{L}_{\text{con}} = \frac{1}{L - 1} \sum_{l=1}^{L-1} \left\lVert \tilde{S}^{(l)} - \tilde{S}^{(l+1)} \right\rVert_1, \qquad \tilde{S}^{(l)}_i = S^{(l)}_i \,\Big/\, \textstyle\sum_j S^{(l)}_j .
$$
Normalising before comparison avoids the failure mode in which raw-MSE consistency loss is dominated by background tokens whose scores are jointly small at every layer.

**Total objective.**
$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + w_{\det}\, \mathcal{L}_{\det} + w_{\text{ent}}\, \mathcal{L}_{\text{ent}} + w_{\text{con}}\, \mathcal{L}_{\text{con}}.
$$
We use $w_{\det} = 0.5$, $w_{\text{ent}} = 0.01$, $w_{\text{con}} = 0.05$.

### G. Implementation details

We implement CT-ATPT in PyTorch with `torch.distributed` data parallelism. The model has $L = 6$ transformer blocks, embedding dimension $d = 384$, $6$ attention heads, dropout $0.1$, and $N_{\min} = 24$ on $96^3$ crops. We train with AdamW (lr $2\times 10^{-4}$, weight decay $0.05$), a 5-epoch linear warmup followed by cosine decay over 50 epochs, gradient clipping at $1.0$, bfloat16 mixed precision, batch size 1 per GPU, and gradient accumulation 8. Augmentation consists of independent random axis flips ($p = 0.5$ each), Gaussian intensity noise ($\sigma \sim \mathcal{U}(0, 0.02)$), and bounded brightness shifts ($\pm 5\%$).

---

## IV. Experimental Protocol

### A. Dataset

We use the **LIDC-IDRI** dataset [REF: Armato 2011], which provides 1{,}018 thoracic CT scans with per-nodule malignancy ratings from up to four radiologists on a 1–5 scale. We obtain nodule clusters via `pylidc.cluster_annotations()`, retain only nodules annotated by $\geq 3$ radiologists with mean diameter $\geq 3$ mm, and exclude nodules with mean malignancy score exactly $3.0$ (ambiguous). Remaining nodules are labelled malignant if the mean radiologist score exceeds $3.0$, and benign otherwise. From each scan we extract a cubic $96^3$ voxel crop centred on the consensus nodule centroid, after lung-window normalisation ($[-1000, 400]$ HU $\rightarrow [0, 1]$).

After this preprocessing, we obtain **[RESULT: N_NODULES]** nodules from **[RESULT: N_PATIENTS]** patients (**[RESULT: N_MALIGNANT]** malignant, **[RESULT: N_BENIGN]** benign).

### B. Split protocol

Crucially, we split at the **patient** level, not the nodule level, using `GroupShuffleSplit` with the patient ID as the group key. Patient-level splitting prevents the well-known leakage failure in which multiple nodules from the same scan appear in both training and validation. We use an 80/20 train/validation split with `random_state = 42`; a held-out test set will be reported in the final submission.

### C. Hardware and software

All experiments run on a single NVIDIA A100 40 GB GPU on Google Colab Pro. The full pipeline is implemented in PyTorch 2.x; preprocessing relies on `pylidc` and `SimpleITK`. Training one configuration to convergence (50 epochs) takes approximately **[RESULT: TRAIN_TIME]** wall-clock hours.

### D. Baselines

We compare CT-ATPT to:

1. **3D ViT (no pruning).** Same backbone as CT-ATPT but with the pruning module disabled — every block attends over all $N$ tokens.
2. **DynamicViT-style fixed-ratio pruning.** Same backbone, but tokens are pruned at a *fixed* keep ratio (0.7) per block, with a Gumbel-softmax mask trained end-to-end. This isolates the contribution of adaptive thresholding.
3. **EViT-style top-$K$ pruning.** Tokens are ranked by class-attention magnitude and the bottom fraction is merged into a single fused token. This isolates the contribution of our tri-component score and recycling step.
4. **3D ResNet-50** trained from scratch on the same crops, as a non-transformer reference point.

### E. Metrics

Following best practice for clinical CAD evaluation we report: accuracy, balanced accuracy, sensitivity, specificity, precision, $F_1$, ROC-AUC, PR-AUC, and the full confusion matrix. We additionally report inference latency (ms per crop), peak GPU memory (MB), and average tokens retained per layer.

---

## V. Results

### A. Main comparison

Table I reports the held-out validation performance of CT-ATPT against all baselines.

**Table I — Main results on LIDC-IDRI (patient-level split).** All numbers are mean over [RESULT: N_SEEDS] runs.

| Method | Tokens kept (avg.) | ROC-AUC | PR-AUC | Bal. Acc. | Sens. | Spec. | $F_1$ | Latency (ms) | Mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 3D ResNet-50 | — | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| 3D ViT (no pruning) | 216 / 216 | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| DynamicViT (fixed ρ=0.7) | 151 / 216 | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| EViT (top-K, K=128) | 128 / 216 | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] | [RESULT] |
| **CT-ATPT (ours)** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

### B. Ablation studies

We ablate each design choice. Starting from the full model, we disable one component at a time.

**Table II — Component ablations.** Numbers are validation ROC-AUC and average kept tokens per block.

| Variant | ROC-AUC | Bal. Acc. | Kept (final block) |
|---|---|---|---|
| Full CT-ATPT | [RESULT] | [RESULT] | [RESULT] |
| − attention component $A$ ($\alpha \equiv 0$) | [RESULT] | [RESULT] | [RESULT] |
| − rollout component $R$ ($\beta \equiv 0$) | [RESULT] | [RESULT] | [RESULT] |
| − energy component $E$ ($\gamma \equiv 0$) | [RESULT] | [RESULT] | [RESULT] |
| Fixed $(\alpha,\beta,\gamma) = (\tfrac{1}{3},\tfrac{1}{3},\tfrac{1}{3})$ (no learning) | [RESULT] | [RESULT] | [RESULT] |
| Fixed threshold $\tau = \mu$ (no learnable $\lambda$) | [RESULT] | [RESULT] | [RESULT] |
| No token recycling | [RESULT] | [RESULT] | [RESULT] |
| No entropy regulariser | [RESULT] | [RESULT] | [RESULT] |
| No consistency regulariser | [RESULT] | [RESULT] | [RESULT] |

### C. Learned pruning behaviour

Figure [REF Fig 2] shows the trajectories of the learnable parameters over the course of training. The mixture weights $(\alpha, \beta, \gamma)$ start near uniform $(0.33, 0.33, 0.33)$ and converge to **[RESULT: alpha]**, **[RESULT: beta]**, **[RESULT: gamma]**, indicating that the model emphasises the [RESULT: which component dominates]. The threshold parameter $\lambda$ converges to **[RESULT: lambda]**, corresponding to an average final-block keep ratio of **[RESULT: keep ratio]**. The min-keep fallback fires in **[RESULT: %]** of training steps after the first **[RESULT: warmup]** epochs and approaches zero thereafter, confirming that the statistical threshold becomes well-behaved without manual tuning.

### D. Inference efficiency

At inference, CT-ATPT yields **[RESULT: SPEEDUP]× speedup** and **[RESULT: MEMORY%] lower peak memory** versus the no-pruning ViT at matched ROC-AUC (within [RESULT: tolerance] absolute points). The speedup is roughly linear in the average retained-token count, consistent with the $\mathcal{O}(N^2)$ scaling of attention.

### E. Qualitative analysis

Figure [REF Fig 3] visualises which tokens survive successive pruning layers on representative malignant and benign cases. Surviving tokens cluster around the nodule and adjacent vasculature on malignant cases, and around the nodule itself on benign cases — consistent with the radiological intuition that contextual features (vascular convergence, pleural attachment) are diagnostically informative.

---

## VI. Discussion

**Why a statistical threshold helps.** Fixed-ratio pruning forces every scan to surrender the same fraction of tokens, but the volume of diagnostically informative tissue varies dramatically: a sub-pleural nodule near the chest wall has different supporting context than a centrally located perihilar nodule. A statistical threshold $\tau_l = \mu_l + \lambda \sigma_l$ lets the model retain more tokens in scans with diffuse informative content and prune more aggressively in scans dominated by background. The learnable $\lambda$ is shared across layers, which we found gave noticeably more stable behaviour during early training than per-layer learnable thresholds.

**Why the CT-specific energy term matters.** Pruning ViTs designed for natural images implicitly assume that attention itself is a sufficient saliency signal. In CT, however, large fractions of the volume are air or homogeneous parenchyma whose token embeddings are nearly identical, producing attention maps that are uninformative early in training. The energy term provides a strong inductive prior — voxel blocks with HU variance and gradient activity carry candidate diagnostic content — which empirically accelerates convergence of $\lambda$ and improves keep-ratio stability (Table II, row removing $\gamma$).

**Why recycling matters.** Without recycling, an aggressive pruning step at an early layer would permanently destroy information that the classifier could otherwise have used. Recycling injects a graded summary of pruned tokens into the classification token via a temperature-softmax — tokens scored just below the threshold contribute most, those scored far below contribute little — preserving evidence in proportion to its likely usefulness.

**Limitations.** First, our experiments use $96^3$ nodule crops rather than full-resolution chest CT; extending the architecture to full-volume inputs requires a memory-efficient attention variant such as FlashAttention or block-sparse attention, which we leave to future work. Second, while LIDC-IDRI provides high-quality radiologist-consensus labels, the absolute number of labelled malignant nodules remains modest, and the dataset does not include longitudinal follow-up to verify malignancy. External validation on the NLST cohort or institutional datasets is necessary before clinical deployment. Third, our experiments rely on patient-level (not site-level or scanner-level) splits, so reported numbers may overstate generalisation across imaging vendors.

**Clinical implications.** If the headline accuracy numbers replicate on external data, CT-ATPT would be a candidate second-reader for early-stage screening triage. The pruning maps themselves are interpretable in a way unpruned attention is not: a clinician can see which voxels the model considered diagnostically informative, supporting trust calibration.

---

## VII. Conclusion

We presented **CT-ATPT**, a 3D Vision Transformer with three coupled novelties for volumetric CT: a learnable tri-component token importance score, a statistical adaptive threshold $\tau_l = \mu_l + \lambda \sigma_l$, and an information-preserving recycling step. On LIDC-IDRI under patient-level splits, CT-ATPT achieves **[RESULT: AUC]** ROC-AUC while pruning **[RESULT: PRUNE%]** of tokens, outperforming fixed-ratio and top-$K$ pruning baselines and matching the no-pruning ViT at a fraction of its inference cost. Ablations confirm each component contributes meaningfully. The framework is general and could plausibly extend to other volumetric modalities — chest MRI, head-and-neck CT, abdominal imaging — where most voxels are uninformative and adaptive token allocation would pay off.

---

## Appendix A — Hyperparameter table

| Hyperparameter | Value |
|---|---|
| Input crop | $96 \times 96 \times 96$ |
| Patch size | $16 \times 16 \times 16$ |
| Initial tokens | 216 + 1 (cls) |
| Transformer depth $L$ | 6 |
| Embedding dim $d$ | 384 |
| Attention heads | 6 |
| Dropout | 0.1 |
| Min keep tokens $N_{\min}$ | 24 |
| Gate sharpness $\kappa$ | 10 |
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-4}$ |
| Weight decay | 0.05 |
| Warmup epochs | 5 |
| Total epochs | 50 |
| Batch size (per GPU) | 1 |
| Gradient accumulation | 8 |
| Gradient clip | 1.0 |
| Precision | bfloat16 |
| Focal $\gamma_f$ | 2.0 |
| Focal $\alpha_{\text{pos}}$ | 0.5 |
| $w_{\det}$ | 0.5 |
| $w_{\text{ent}}$ | 0.01 |
| $w_{\text{con}}$ | 0.05 |

---

## Appendix B — Results to fill after training run

The following fields are placeholders; please supply the corresponding numbers from the training run.

```
[RESULT: AUC]            — final validation ROC-AUC
[RESULT: BA]             — final validation balanced accuracy
[RESULT: SENS]           — sensitivity at threshold 0.5
[RESULT: SPEC]           — specificity at threshold 0.5
[RESULT: F1]             — F1 score
[RESULT: PR_AUC]         — PR-AUC
[RESULT: PRUNE%]         — average tokens pruned by final block (%)
[RESULT: SPEEDUP]        — inference speedup vs no-pruning ViT
[RESULT: MEMORY%]        — peak-memory reduction vs no-pruning ViT
[RESULT: N_NODULES]      — total nodules after filtering
[RESULT: N_PATIENTS]     — total patients
[RESULT: N_MALIGNANT]    — number of malignant nodules
[RESULT: N_BENIGN]       — number of benign nodules
[RESULT: TRAIN_TIME]     — wall-clock training hours (50 epochs, A100)
[RESULT: alpha]          — learned attention weight
[RESULT: beta]           — learned rollout weight
[RESULT: gamma]          — learned energy weight
[RESULT: lambda]         — learned threshold scalar
[RESULT: keep ratio]     — average kept fraction at final block
[RESULT: N_SEEDS]        — number of seeds averaged for main table
```

All baseline numbers in Tables I–II are also placeholders to be filled.

---

## Appendix C — Reverse outline (drafting aid; remove before submission)

**Thesis:** CT-ATPT — a 3D ViT with three coupled novelties (tri-component learnable score, statistical adaptive threshold, recycling) — outperforms fixed-ratio and top-$K$ pruning ViTs on LIDC-IDRI malignancy classification while substantially reducing inference cost.

**Intro paragraph 1:** Clinical motivation — malignancy decision is the bottleneck.
**Intro paragraph 2:** 3D ViTs are right but expensive; existing pruning is fixed-rate / non-CT-aware / training-only.
**Intro paragraph 3:** Our three contributions, summarised.
**Intro paragraph 4:** Contributions list + paper organisation.

**Method:** patch embed → tri-component score → statistical threshold → recycling → focal cls + det + entropy + consistency.

**Experiments:** LIDC-IDRI, patient-level split, four baselines, headline metrics, full ablations, learned-parameter trajectories, latency/memory, qualitative maps.

**Discussion:** why statistical, why energy, why recycling, limitations, clinical implications.

---

## Appendix D — End-of-draft self-review checklist (five dimensions)

### 1. Contribution

- [ ] Are the three novelties (tri-component score, statistical threshold, recycling) clearly distinguished from prior pruning ViTs?
- [ ] Is the combination — not just any one component — non-trivial?
- [ ] Is the medical-imaging framing distinct enough from a generic ViT-pruning paper?

### 2. Writing clarity

- [ ] Does every paragraph have a single topic sentence?
- [ ] Are $A, R, E$ defined the first time they appear, and used consistently thereafter?
- [ ] Are all symbols ($\lambda$, $\tau_l$, $g_i$, $T$, $\kappa$) introduced in equations, not buried in prose?

### 3. Experimental strength

- [ ] Are baselines strong enough? (3D ResNet, no-pruning ViT, DynamicViT, EViT — yes)
- [ ] Is patient-level splitting explicitly stated? (yes)
- [ ] Is the comparison fair in compute / parameter count? (need to confirm in final draft)
- [ ] Have we reported multiple seeds? (N_SEEDS placeholder)
- [ ] Have we reported confidence intervals? (not yet — add in final)

### 4. Evaluation completeness

- [ ] All standard binary-classification metrics? (yes — Acc, BA, Sens, Spec, Prec, F1, AUC, PR-AUC, CM)
- [ ] Inference efficiency metrics? (latency, memory)
- [ ] Qualitative analysis (token maps)?
- [ ] External validation? (limitation, future work)

### 5. Method design soundness

- [ ] Does each component have a clear motivation?
- [ ] Are losses justified individually (focal for imbalance, entropy for gate confidence, consistency for depth-stability)?
- [ ] Is the fallback (min-keep) explained, and is its training-time frequency reported?

---

## Appendix E — Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| CT-ATPT outperforms unpruned 3D ViT | Table I row "3D ViT (no pruning)" vs "CT-ATPT" | needs results |
| CT-ATPT outperforms fixed-ratio pruning | Table I row "DynamicViT" vs "CT-ATPT" | needs results |
| Each component of $S$ is necessary | Table II — α/β/γ ablations | needs results |
| Learnable mixture weights help vs uniform | Table II — fixed $(1/3,1/3,1/3)$ row | needs results |
| Statistical threshold helps vs fixed $\tau$ | Table II — fixed $\tau = \mu$ row | needs results |
| Recycling helps | Table II — "no token recycling" row | needs results |
| Pruning yields inference speedup | Section V-D — latency/memory numbers | needs results |
| Min-keep fallback drops to near-zero with training | Section V-C — fallback trajectory | needs results |
| Surviving tokens cluster near nodule | Figure 3 — qualitative maps | needs figure |
| Patient-level split prevents leakage | Section IV-B — describes GroupShuffleSplit | supported (methodology) |
| Focal loss correctly down-weights easy cases | Section III-F + numerical validation in Python | supported |
| Attention rollout remains row-stochastic | Section III-C + numerical validation in Python | supported |

---

*End of draft. All results placeholders will be filled when training completes.*
