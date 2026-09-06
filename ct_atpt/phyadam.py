"""PhyAdam: Physics-inspired adaptive optimizer using Newtonian mechanics.

Instead of Adam's exponential moving averages, PhyAdam models optimization
as a particle moving through the loss landscape under Newtonian dynamics:

  Force:        F_t = -grad(L(theta))                    (negative gradient)
  Variance:     v_t = beta2*v_{t-1} + (1-beta2)*g_t^2     (gradient variance EMA)
  Adaptive mass: M_t = M0 + alpha*sqrt(v_hat_t + eps)     (heavier where variance is high)
  Acceleration: A_t = (F_t - mu*V_{t-1}) / M_t            (Newton's 2nd law + friction)
  Velocity:     V_t = beta*V_{t-1} + (1-beta)*A_t         (inertia-based velocity)
  Update:       theta_{t+1} = theta_t + lr*V_hat_t        (step by bias-corrected velocity)

Bias correction (like Adam) is applied to both v_t and V_t.
Decoupled weight decay (like AdamW) is applied BEFORE the velocity update.

Hyperparameter defaults:
  lr=1e-4, beta=0.9, beta2=0.999, base_mass=1.0, mass_scale=0.1,
  friction=0.1, weight_decay=0.0, eps=1e-8
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class PhyAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        beta2: float = 0.999,
        base_mass: float = 1.0,
        mass_scale: float = 0.1,
        friction: float = 0.1,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if base_mass <= 0.0:
            raise ValueError(f"Invalid base_mass: {base_mass}")

        defaults = dict(
            lr=lr,
            beta=beta,
            beta2=beta2,
            base_mass=base_mass,
            mass_scale=mass_scale,
            friction=friction,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            beta2 = group["beta2"]
            base_mass = group["base_mass"]
            mass_scale = group["mass_scale"]
            friction = group["friction"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PhyAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["velocity"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p)

                velocity = state["velocity"]
                variance = state["variance"]
                state["step"] += 1
                step = state["step"]

                # Gradient-variance EMA, bias-corrected (as in Adam's v_t).
                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction2 = 1.0 - beta2 ** step
                variance_hat = variance / bias_correction2

                # Adaptive mass: heavier (harder to accelerate) where gradient
                # variance is high, i.e. noisier directions get damped.
                mass = base_mass + mass_scale * torch.sqrt(variance_hat + eps)

                # Newton's second law with friction opposing the previous
                # velocity: F = -grad (downhill force), A = (F - mu*V)/M.
                force = -grad
                acceleration = (force - friction * velocity) / mass

                # Inertia-based velocity update, bias-corrected (as in Adam's m_t).
                velocity.mul_(beta).add_(acceleration, alpha=1.0 - beta)
                bias_correction1 = 1.0 - beta ** step
                velocity_hat = velocity / bias_correction1

                # Decoupled weight decay (AdamW-style), applied before the
                # velocity-based update so both optimizers regularize identically.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # theta += lr * V_hat: V_hat points downhill (same sign
                # convention as Adam's m_hat/-grad), so this steps downhill.
                p.add_(velocity_hat, alpha=lr)

        return loss
