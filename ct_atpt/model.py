from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def _dtype_eps(dtype: torch.dtype) -> float:
    try:
        return max(float(torch.finfo(dtype).eps), 1e-6)
    except TypeError:
        return 1e-6


def _batch_normalize_01(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Min-max normalize each sample in a batch independently. x: [B, N]"""
    eps = _dtype_eps(x.dtype) if eps is None else eps
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


def compute_patch_energy(volume: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """CT-specific local energy: patch variance + local gradient energy.

    Input shape:  [B, 1, Z, Y, X]
    Output shape: [B, num_patches]  — normalized per sample to [0, 1]
    """
    pz, py, px = patch_size

    with torch.no_grad():
        patches = volume.unfold(2, pz, pz).unfold(3, py, py).unfold(4, px, px)
        variance = patches.var(dim=(-1, -2, -3), unbiased=False).squeeze(1)

        grad_z = F.pad(torch.abs(volume[:, :, 1:] - volume[:, :, :-1]),   (0, 0, 0, 0, 1, 0))
        grad_y = F.pad(torch.abs(volume[:, :, :, 1:] - volume[:, :, :, :-1]), (0, 0, 1, 0, 0, 0))
        grad_x = F.pad(torch.abs(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]), (1, 0, 0, 0, 0, 0))
        grad   = grad_z + grad_y + grad_x

        grad_patches = grad.unfold(2, pz, pz).unfold(3, py, py).unfold(4, px, px)
        grad_energy  = grad_patches.mean(dim=(-1, -2, -3)).squeeze(1)

        energy = (variance + grad_energy).flatten(1)   # [B, num_patches]
        return _batch_normalize_01(energy)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero whole residual branches per sample."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x / keep_prob * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float,
                 drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                [B, T+1, D]
            key_padding_mask: [B, T+1] bool — True positions are ignored by attention.
                              Position 0 (CLS) should always be False.
        """
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_input, attn_input, attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x, attn_weights   # attn_weights: [B, heads, T+1, T+1]


@dataclass(frozen=True)
class CTATPTConfig:
    input_shape:      tuple[int, int, int] = (128, 512, 512)
    patch_size:       tuple[int, int, int] = (8, 32, 32)
    embed_dim:        int   = 768
    depth:            int   = 6
    num_heads:        int   = 12
    mlp_ratio:        float = 4.0
    dropout:          float = 0.1
    num_classes:      int   = 2
    drop_path_rate:   float = 0.0
    min_keep_tokens:  int   = 64
    min_keep_ratio:   float = 0.25
    max_prune_fraction_per_block: float = 0.5
    pruning_mode:     str   = "adaptive"
    scale_kept_tokens: bool = False
    pruning_warmup_epochs: int = 10
    # "soft" mode only: gentle keep-ratio curriculum (differentiable, learnable).
    prune_target_keep: float = 0.5
    prune_ramp_epochs: int   = 15
    soft_lambda_init:  float = -2.0   # less negative => gates start responsive (not saturated)
    gate_sharpness:    float = 10.0   # kappa; lower => softer gates, more gradient flow


class CTATPT(nn.Module):
    """CT Adaptive Token Pruning Transformer.

    Batched implementation: supports any batch size.

    Pruning is done via gating + attention masking:
      - All token positions stay in the tensor throughout (fixed shape).
      - Pruned positions are zeroed and masked from attention.
      - Adaptive mode: hard token dropping at a few stage blocks (depth/4,
        depth/2, 3*depth/4), per-sample keep count from tau = mu + lam*sigma
        clamped to a band around a scheduled budget; kept tokens stay at full
        strength and a straight-through gate carries gradient into the scores.
      - Soft mode: every kept position is scaled by a learned gate in (0, 1).
      - Pruned token information is recycled into the CLS token before zeroing.

    This design allows standard batched operations (no variable-length sequences)
    while the adaptive threshold still varies per sample within a batch.
    """

    def __init__(self, config: CTATPTConfig = CTATPTConfig()) -> None:
        super().__init__()
        self.config = config

        pz, py, px = config.patch_size
        z,  y,  x  = config.input_shape
        if z % pz or y % py or x % px:
            raise ValueError("input_shape must be divisible by patch_size in every dimension")

        self.grid_shape = (z // pz, y // py, x // px)
        self.num_tokens = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, config.embed_dim))

        self.det_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 4),
        )

        # Linearly increasing stochastic-depth rate across blocks (0 -> drop_path_rate).
        dpr = [
            config.drop_path_rate * i / max(1, config.depth - 1)
            for i in range(config.depth)
        ]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(config.depth)
        ])

        self.importance_logits    = nn.Parameter(torch.zeros(3))        # alpha, beta, gamma
        # "adaptive" mode: learnable RESIDUAL on the scheduled lambda,
        # bounded via tanh(.)*0.25. Init 0 => tau starts exactly on the
        # budget schedule; the model fine-tunes the operating point from there.
        self.lambda_raw           = nn.Parameter(torch.tensor(0.0))
        # "soft" mode threshold scalar: UNconstrained so tau can sit below the
        # mean (keep >50%). Init negative => starts keeping more tokens (gentle).
        self.soft_lambda_raw      = nn.Parameter(torch.tensor(float(config.soft_lambda_init)))
        self.temperature_raw      = nn.Parameter(torch.tensor(0.0))
        # Gate sharpness κ (paper Section III-D, eq. for g_i). Lower => softer
        # gates that stay in the responsive (non-saturated) range during training.
        self.gate_sharpness       = float(config.gate_sharpness)

        # CLS token + Global Average Pool of active patch tokens → classifier
        # Using both gives the classifier direct access to all patch features
        # rather than relying solely on learned attention routing to CLS.
        cls_input_dim = config.embed_dim * 2  # [CLS ; GAP(patches)]
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_input_dim),
            nn.Linear(cls_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Adaptive mode prunes at a few stage blocks (EViT-style), not after
        # every block: per-block compounding (tau >= mean kept <=50% of the
        # remainder 12 times over) forced every sample into the min-keep
        # fallback, freezing abg/lambda and collapsing AUC to 0.5.
        d = config.depth
        self._prune_stage_list: list[int] = sorted(
            {d // 4, d // 2, (3 * d) // 4} if d >= 4 else {d - 1}
        )
        self._prune_stage_set: set[int] = set(self._prune_stage_list)

        self._current_epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch for pruning warmup scheduling."""
        self._current_epoch = epoch

    def _soft_effective_lambda(self) -> torch.Tensor:
        """Effective soft-mode threshold scalar: scheduled + learnable residual.

        Walks lambda from lam_start (tau well below mean => keep ~all, gentle
        onset) to lam_end (=> keep ~target) over the post-warmup ramp.
        lam_end comes from keep = P(score > mu + lambda*sigma) under a ~normal
        score distribution => lambda = ndtri(1 - keep).
        """
        warm = self.config.pruning_warmup_epochs
        ramp = max(1, self.config.prune_ramp_epochs)
        frac = min(1.0, max(0.0, (self._current_epoch - warm) / ramp))
        lam_start = -2.0                                               # keep ~0.98 at onset
        keep = min(max(float(self.config.prune_target_keep), 1e-4), 1.0 - 1e-4)
        lam_end = float(torch.special.ndtri(torch.tensor(1.0 - keep)))
        lam_sched = lam_start + frac * (lam_end - lam_start)
        return lam_sched + torch.tanh(self.soft_lambda_raw) * 0.25     # small learnable residual

    def _adaptive_effective_lambda(self) -> torch.Tensor:
        """Effective adaptive-mode threshold scalar: scheduled + learnable residual.

        lam_sched targets the per-stage keep fraction rho(epoch)^(1/num_stages)
        via keep = P(score > mu + lam*sigma) => lam = ndtri(1 - keep).
        """
        warm = self.config.pruning_warmup_epochs
        ramp = max(1, self.config.prune_ramp_epochs)
        frac = min(1.0, max(0.0, (self._current_epoch - warm) / ramp))
        rho  = 1.0 - frac * (1.0 - float(self.config.prune_target_keep))
        num_stages = max(1, len(self._prune_stage_list))
        stage_frac = min(max(rho ** (1.0 / num_stages), 1e-4), 1.0 - 1e-4)
        lam_sched  = float(torch.special.ndtri(torch.tensor(1.0 - stage_frac)))
        return lam_sched + torch.tanh(self.lambda_raw) * 0.25

    # ------------------------------------------------------------------
    # Per-block pruning (batched)
    # ------------------------------------------------------------------

    def _prune_after_block(
        self,
        x:            torch.Tensor,   # [B, T+1, D]
        attn:         torch.Tensor,   # [B, heads, T+1, T+1]
        rollout:      torch.Tensor,   # [B, T+1, T+1]
        patch_energy: torch.Tensor,   # [B, T]
        active_mask:  torch.Tensor,   # [B, T] bool — True = still active
        block_idx:    int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B = x.shape[0]
        T = self.num_tokens
        active_f = active_mask.float()   # [B, T]

        # Skip pruning when disabled or during warmup (backbone learns features first)
        in_warmup = (
            self.config.pruning_mode in ("adaptive", "soft")
            and self._current_epoch < self.config.pruning_warmup_epochs
        )
        if self.config.pruning_mode == "none" or in_warmup:
            zero = x.sum() * 0.0
            stats = {
                "entropy": zero,
                "score_map": active_f,
                "keep_ratio": active_f.sum(dim=1) / T,
                "tau": torch.zeros(B, device=x.device),
                "lambda": self.lambda_raw.detach(),
                "alpha_beta_gamma": torch.softmax(self.importance_logits, dim=0).detach(),
                "fallback": torch.zeros(B, dtype=torch.bool, device=x.device),
                "sparsity": zero,
            }
            return x, active_mask, rollout, stats

        if self.config.pruning_mode == "soft":
            return self._prune_soft(x, attn, rollout, patch_energy, active_mask)

        # ── Update attention rollout ──────────────────────────────────
        attn_mean = attn.mean(dim=1)   # [B, T+1, T+1]

        # Mask out pruned rows/cols before rollout update so inactive
        # tokens don't pollute propagation paths.
        active_full = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=x.device), active_mask], dim=1
        )  # [B, T+1]
        mask2d = (active_full.unsqueeze(2) & active_full.unsqueeze(1)).float()
        attn_mean = torch.nan_to_num(attn_mean * mask2d, nan=0.0, posinf=0.0, neginf=0.0)

        eye     = torch.eye(T + 1, device=x.device, dtype=attn_mean.dtype).unsqueeze(0)
        attn_aug = (attn_mean + eye) * 0.5
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp_min(_dtype_eps(attn_aug.dtype))
        rollout = torch.bmm(attn_aug, rollout)   # [B, T+1, T+1]

        # Rollout is updated at every block (so stage blocks see the full
        # propagation history), but tokens are only pruned at stage blocks.
        if block_idx not in self._prune_stage_set:
            zero = x.sum() * 0.0
            stats = {
                "entropy": zero,
                "score_map": active_f,
                "keep_ratio": active_f.sum(dim=1) / T,
                "tau": torch.zeros(B, device=x.device),
                "lambda": self.lambda_raw.detach(),
                "alpha_beta_gamma": torch.softmax(self.importance_logits, dim=0).detach(),
                "fallback": torch.zeros(B, dtype=torch.bool, device=x.device),
                "sparsity": zero,
            }
            return x, active_mask, rollout, stats

        # ── Importance scores ─────────────────────────────────────────
        # How much each patch token is attended to (average over all queries)
        attention_received = attn_mean[:, :, 1:].mean(dim=1)   # [B, T]
        # CLS-to-patch rollout: how much final CLS "sees" each patch
        rollout_score = rollout[:, 0, 1:]                       # [B, T]
        energy_score  = patch_energy.to(dtype=x.dtype)          # [B, T]

        # Normalize per sample, then zero inactive positions
        attention_received = _batch_normalize_01(attention_received) * active_f
        rollout_score      = _batch_normalize_01(rollout_score)      * active_f
        energy_score       = energy_score                            * active_f

        weights = torch.softmax(self.importance_logits, dim=0)
        scores  = (
            weights[0] * attention_received
            + weights[1] * rollout_score
            + weights[2] * energy_score
        ).clamp(0.0, 1.0) * active_f   # [B, T]

        # ── Scheduled network-level keep budget rho(epoch) ────────────
        # rho ramps 1.0 -> prune_target_keep after warmup. Each stage clamps
        # the per-sample keep count to a band around the CUMULATIVE budget
        # track rho^(j/S) of the ORIGINAL T (j = stage rank, S = #stages) —
        # anchoring on the absolute track means per-stage clamping cannot
        # compound into budget drift (a relative band drifts by band^S).
        warm = self.config.pruning_warmup_epochs
        ramp = max(1, self.config.prune_ramp_epochs)
        frac = min(1.0, max(0.0, (self._current_epoch - warm) / ramp))
        rho  = 1.0 - frac * (1.0 - float(self.config.prune_target_keep))
        num_stages = max(1, len(self._prune_stage_list))
        stage_rank = self._prune_stage_list.index(block_idx) + 1
        cum_target = (rho ** (stage_rank / num_stages)) * T

        # ── Per-sample adaptive threshold ─────────────────────────────
        n_active = active_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_s   = (scores * active_f).sum(dim=1, keepdim=True) / n_active
        var_s    = ((scores - mean_s).pow(2) * active_f).sum(dim=1, keepdim=True) / n_active
        std_s    = var_s.sqrt()

        # Lambda is SCHEDULED to the stage budget with a small learnable
        # residual (same recipe as soft-mode v4). A purely learnable lambda
        # lost the tug-of-war against the classification loss (v5.1 run:
        # lambda drifted AWAY from the budget, every sample stayed clamped at
        # the band ceiling). Scheduling puts tau inside the band by
        # construction; the residual and the score distribution's shape
        # decide the per-sample counts within it.
        lam  = self._adaptive_effective_lambda()
        tau  = mean_s + lam * std_s                               # [B, 1]

        gate = torch.sigmoid(self.gate_sharpness * (scores - tau)) * active_f  # [B, T]

        # ── Recycle pruned info into CLS ──────────────────────────────
        token_x       = x[:, 1:, :]                                    # [B, T, D]
        recycle_temp  = F.softplus(self.temperature_raw) + 1e-4
        recycle_logits = scores / recycle_temp
        # Only tokens being pruned (below threshold) contribute to recycling
        recycle_w  = torch.softmax(recycle_logits, dim=1) * (1.0 - gate) * active_f
        recycle_w  = recycle_w / recycle_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        recycled   = torch.bmm(recycle_w.unsqueeze(1), token_x).squeeze(1)  # [B, D]
        cls_x      = x[:, :1, :] + recycled.unsqueeze(1)               # [B, 1, D]

        # ── Keep / prune decision: adaptive count within a budget band ──
        # tau decides how many tokens THIS sample keeps (per-sample adaptivity:
        # on heavy-tailed real score distributions the fraction above mu+lam*
        # sigma varies with the sample's score concentration), and the count is
        # clamped to +/-10% of the cumulative budget track so the schedule is
        # honoured and the hard floors are preserved.
        keep_mask = (scores > tau) & active_mask                        # [B, T]

        cfg_min_keep = min(
            max(self.config.min_keep_tokens, math.ceil(T * self.config.min_keep_ratio), 1),
            T,
        )
        fallback = torch.zeros(B, dtype=torch.bool, device=x.device)
        for b in range(B):
            n_active_b = int(active_mask[b].sum().item())
            floor_b = max(
                cfg_min_keep,
                math.ceil(n_active_b * (1.0 - self.config.max_prune_fraction_per_block)),
                math.ceil(cum_target * 0.9),
            )
            ceil_b = max(floor_b, math.floor(min(float(n_active_b), cum_target * 1.1)))
            floor_b = min(floor_b, n_active_b)
            k_b = int(keep_mask[b].sum().item())
            k_clamped = min(max(k_b, floor_b), ceil_b, n_active_b)
            if k_clamped != k_b:
                s = scores[b].masked_fill(~active_mask[b], -1.0)
                top_idx = torch.topk(s, k=k_clamped, largest=True).indices
                row = torch.zeros(T, dtype=torch.bool, device=x.device)
                row[top_idx] = True
                keep_mask[b] = row
                fallback[b] = True

        # Kept tokens stay at FULL strength in the forward pass (hard dropping
        # is plain token dropout to a pretrained ViT — no activation-scale
        # shock), while a straight-through gate routes gradient into scores so
        # alpha/beta/gamma and lambda actually learn.
        keep_f = keep_mask.to(dtype=token_x.dtype).unsqueeze(-1)
        if self.config.scale_kept_tokens:
            kept_tokens = token_x * gate.unsqueeze(-1) * keep_f
        else:
            g = gate.unsqueeze(-1)
            kept_tokens = token_x * (keep_f + g - g.detach())
        x = torch.cat([cls_x, kept_tokens], dim=1)                     # [B, T+1, D]

        # ── Budget-centring sparsity loss (differentiable, drives lambda) ──
        # The hard count is clamped to the band, so the classification loss
        # alone gives lambda no reason to move (smoke test: lambda stayed at
        # init, every sample pinned at the band ceiling => zero adaptivity).
        # Penalise the PRE-clamp soft keep fraction's distance from the
        # cumulative track: tau learns to sit inside the band, the clamp goes
        # quiet, and per-sample variation can express.
        soft_keep_frac = (gate * active_f).sum(dim=1) / n_active.squeeze(1)   # [B]
        target_frac    = (cum_target / n_active.squeeze(1)).clamp(max=1.0)    # [B]
        sparsity       = (soft_keep_frac - target_frac).pow(2).mean()

        # ── Auxiliary outputs ─────────────────────────────────────────
        score_map = scores * keep_mask.float()                          # [B, T]

        # Entropy regularization (encourages decisive gating)
        eps = _dtype_eps(gate.dtype)
        g_ent   = gate.masked_fill(~active_mask, 0.5).clamp(eps, 1.0 - eps)
        ent_per = -(g_ent * g_ent.log() + (1 - g_ent) * (1 - g_ent).log())
        entropy = (ent_per * active_f).sum() / active_f.sum().clamp_min(1.0)

        stats = {
            "entropy":         entropy,
            "score_map":       score_map,                                  # [B, T]
            "keep_ratio":      keep_mask.float().sum(dim=1) / T,           # [B]
            "tau":             tau.squeeze(1).detach(),                    # [B]
            "lambda":          lam.detach(),
            "alpha_beta_gamma": weights.detach(),
            "fallback":        fallback,                                    # [B] bool
            "sparsity":        sparsity,
        }
        return x, keep_mask, rollout, stats

    # ------------------------------------------------------------------
    # Soft, differentiable pruning (curriculum) — pruning_mode == "soft"
    # ------------------------------------------------------------------

    def _prune_soft(
        self,
        x:            torch.Tensor,   # [B, T+1, D]
        attn:         torch.Tensor,   # [B, heads, T+1, T+1]
        rollout:      torch.Tensor,   # [B, T+1, T+1]
        patch_energy: torch.Tensor,   # [B, T]
        active_mask:  torch.Tensor,   # [B, T] bool — all True in soft mode
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Differentiable soft pruning with a gentle keep-ratio curriculum.

        Unlike `_prune_after_block`, no token is hard-removed and there is no
        fallback: every patch token is multiplied by a soft gate g in (0,1).
        Because the gate is differentiable, the importance weights (alpha/beta/
        gamma) and the threshold scalar (soft_lambda) receive gradient from the
        classification loss. A scheduled sparsity penalty ramps the average keep
        fraction from 1.0 down to `prune_target_keep`, so pruning ramps in gently.
        """
        B = x.shape[0]
        T = self.num_tokens

        # ── Attention rollout (all tokens active here) ───────────────
        attn_mean = attn.mean(dim=1)                                   # [B, T+1, T+1]
        eye = torch.eye(T + 1, device=x.device, dtype=attn_mean.dtype).unsqueeze(0)
        attn_aug = (attn_mean + eye) * 0.5
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp_min(_dtype_eps(attn_aug.dtype))
        rollout = torch.bmm(attn_aug, rollout)

        # ── Tri-component importance score (differentiable in abg) ────
        attention_received = _batch_normalize_01(attn_mean[:, :, 1:].mean(dim=1))  # [B, T]
        rollout_score      = _batch_normalize_01(rollout[:, 0, 1:])                # [B, T]
        energy_score       = patch_energy.to(dtype=x.dtype)                        # [B, T]

        weights = torch.softmax(self.importance_logits, dim=0)
        scores = (
            weights[0] * attention_received
            + weights[1] * rollout_score
            + weights[2] * energy_score
        ).clamp(0.0, 1.0)                                              # [B, T]

        # ── Statistical threshold tau = mu + lambda*sigma, lambda SCHEDULED ──
        # Keep v2's smooth, self-centring (mu + lambda*sigma) threshold: the
        # quantile variant collapsed the pretrained backbone because it
        # scale-shocked every activation the instant pruning turned on (at the
        # min-score cut, the soft gate multiplies *all* tokens by ~0.5-0.8).
        # Instead we *schedule lambda* from lam_start (tau well below mean =>
        # keep ~all, gentle onset) to lam_end (=> keep ~target), walking the
        # keep-ratio down smoothly with no onset shock. A small learnable
        # residual lets the model fine-tune the operating point.
        lam = self._soft_effective_lambda()

        mean_s = scores.mean(dim=1, keepdim=True)
        std_s  = scores.std(dim=1, keepdim=True, unbiased=False)
        tau    = mean_s + lam * std_s                                  # [B, 1]
        gate   = torch.sigmoid(self.gate_sharpness * (scores - tau))   # [B, T] in (0, 1)

        # ── Recycle below-threshold info into CLS (differentiable) ────
        token_x       = x[:, 1:, :]                                    # [B, T, D]
        recycle_temp  = F.softplus(self.temperature_raw) + 1e-4
        recycle_w     = torch.softmax(scores / recycle_temp, dim=1) * (1.0 - gate)
        recycle_w     = recycle_w / recycle_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        recycled      = torch.bmm(recycle_w.unsqueeze(1), token_x).squeeze(1)   # [B, D]
        cls_x         = x[:, :1, :] + recycled.unsqueeze(1)            # [B, 1, D]

        # ── Soft-suppress patch tokens by the gate (no hard removal) ──
        gated_tokens = token_x * gate.unsqueeze(-1)
        x = torch.cat([cls_x, gated_tokens], dim=1)                    # [B, T+1, D]

        # ── Retention-ratio regulariser (mild; lambda schedule does the work) ──
        # One-sided: only penalise keeping MORE than the final target, so it
        # nudges in the same direction as the schedule and never fights it.
        mean_keep = gate.mean()
        sparsity  = torch.relu(mean_keep - float(self.config.prune_target_keep))

        # Entropy regulariser (push gates toward decisive 0/1)
        eps = _dtype_eps(gate.dtype)
        g   = gate.clamp(eps, 1.0 - eps)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log()).mean()

        stats = {
            "entropy":          entropy,
            "score_map":        scores,                                # [B, T]
            "keep_ratio":       (gate > 0.5).float().sum(dim=1) / T,   # [B]
            "tau":              tau.squeeze(1).detach(),
            "lambda":           lam.detach(),
            "alpha_beta_gamma": weights.detach(),
            "fallback":         torch.zeros(B, dtype=torch.bool, device=x.device),
            "sparsity":         sparsity,
        }
        return x, active_mask, rollout, stats

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        volume: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            volume: [B, 1, Z, Y, X] float32, HU-windowed and normalised to [0, 1].

        Returns:
            logits: [B, num_classes]
            aux:    dict of losses and diagnostics.
        """
        B = volume.shape[0]
        patch_energy = compute_patch_energy(volume, self.config.patch_size)  # [B, T]

        # Patch embedding
        x_grid      = self.patch_embed(volume)                    # [B, D, gz, gy, gx]
        patch_tokens = x_grid.flatten(2).transpose(1, 2)          # [B, T, D]
        det_pred     = self.det_head(patch_tokens)                 # [B, T, 4]

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, patch_tokens], dim=1)               # [B, T+1, D]
        x   = x + self.pos_embed[:, : x.shape[1], :]

        # Tracking state
        active_mask = torch.ones(B, self.num_tokens, dtype=torch.bool, device=volume.device)
        rollout = (
            torch.eye(self.num_tokens + 1, device=volume.device, dtype=x.dtype)
            .unsqueeze(0).expand(B, -1, -1).clone()
        )  # [B, T+1, T+1]

        entropy_terms  = []
        score_maps     = []
        keep_ratios    = []
        sparsity_terms = []
        fallback_total = torch.zeros(1, device=volume.device)

        for block_idx, block in enumerate(self.blocks):
            # key_padding_mask: True = ignore. CLS (pos 0) is never masked.
            kpm = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=volume.device),
                ~active_mask,
            ], dim=1)  # [B, T+1]

            x, attn = block(x, key_padding_mask=kpm)

            # Prevent numerical drift in zeroed positions after attention.
            # Use a new tensor instead of an in-place slice write so autograd can
            # still use the pre-mask activations during backward.
            token_x = x[:, 1:, :] * active_mask.to(dtype=x.dtype).unsqueeze(-1)
            x = torch.cat([x[:, :1, :], token_x], dim=1)

            x, active_mask, rollout, stats = self._prune_after_block(
                x=x,
                attn=attn,
                rollout=rollout,
                patch_energy=patch_energy,
                active_mask=active_mask,
                block_idx=block_idx,
            )
            entropy_terms.append(stats["entropy"])
            score_maps.append(stats["score_map"])
            keep_ratios.append(stats["keep_ratio"].mean())          # scalar per layer
            sparsity_terms.append(stats["sparsity"])
            fallback_total = fallback_total + stats["fallback"].float().sum()

        # Classification: CLS token + Global Average Pool of active patch tokens
        cls_out = x[:, 0, :]                                      # [B, D]
        patch_out = x[:, 1:, :]                                    # [B, T, D]
        # Weight the mean by active_mask so pruned tokens don't contribute
        active_weights = active_mask.float().unsqueeze(-1)         # [B, T, 1]
        gap = (patch_out * active_weights).sum(dim=1) / active_weights.sum(dim=1).clamp_min(1.0)  # [B, D]
        cls_input = torch.cat([cls_out, gap], dim=1)               # [B, 2*D]
        logits = self.cls_head(cls_input)                          # [B, num_classes]

        # Consistency loss: adjacent-layer score distributions should agree
        if len(score_maps) > 1:
            def _norm_map(s: torch.Tensor) -> torch.Tensor:
                return s / s.sum(dim=1, keepdim=True).clamp_min(1e-6)

            consistency = torch.stack([
                (_norm_map(score_maps[i]) - _norm_map(score_maps[i + 1]))
                .abs().sum(dim=1).mean()
                for i in range(len(score_maps) - 1)
            ]).mean()
        else:
            consistency = logits.sum() * 0.0

        if self.config.pruning_mode == "soft":
            lambda_report = self._soft_effective_lambda().detach()
        else:
            lambda_report = self._adaptive_effective_lambda().detach()
        aux = {
            "det_pred":         det_pred,
            "grid_shape":       self.grid_shape,
            "entropy_loss":     torch.stack(entropy_terms).mean(),
            "consistency_loss": consistency,
            "sparsity_loss":    torch.stack(sparsity_terms).mean(),
            "keep_ratios":      torch.stack(keep_ratios),           # [depth]

            "fallback_count":   fallback_total,
            "lambda":           lambda_report,
            "alpha_beta_gamma": torch.softmax(self.importance_logits, dim=0).detach(),
        }
        return logits, aux
