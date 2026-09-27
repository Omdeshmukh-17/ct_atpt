"""Lion: EvoLved Sign Momentum optimizer (Chen et al., "Symbolic Discovery of
Optimization Algorithms", 2023).

Lion tracks a single momentum buffer and steps by the *sign* of a momentum/
gradient interpolation, rather than by a magnitude-normalized moving average
like Adam/AdamW. This makes every parameter's per-step update the same size
(lr, up to sign), which tends to need a smaller learning rate and a larger
decoupled weight decay than AdamW to match its effective step size.

  Interpolate: c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
  Update:      theta_t = theta_{t-1} - lr * (sign(c_t) + weight_decay * theta_{t-1})
  Momentum:    m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

Decoupled weight decay is folded into the update above (AdamW-style: it scales
the parameter directly, not the gradient).

Hyperparameter defaults (per the paper's guidance for Lion vs AdamW):
  lr=1e-4 (Lion typically wants 3-10x smaller lr than AdamW),
  betas=(0.9, 0.99), weight_decay=1e-2 (Lion typically wants 3-10x larger
  weight_decay than AdamW to compensate for the smaller effective lr).
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]

                # Decoupled weight decay, applied directly to the parameter
                # (AdamW-style), before the sign-momentum step.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Update direction interpolates momentum and the fresh
                # gradient, then only its SIGN is used for the step — every
                # parameter moves by exactly `lr` (or 0) per update.
                update = momentum.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # Momentum itself uses a slower-moving beta2 average of the
                # raw gradient (decoupled from the beta1 interpolation above).
                momentum.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
