"""SAM: Sharpness-Aware Minimization (Foret et al., "Sharpness-Aware
Minimization for Efficiently Improving Generalization", 2021).

SAM is not a standalone update rule like AdamW/PhyAdam/Lion — it's a wrapper
around a base optimizer that seeks parameters lying in flat loss-landscape
neighborhoods (which tend to generalize better) by taking an extra ascent
step before the real descent step:

  1. first_step:  climb to the worst point in an epsilon-ball of radius rho
                  around theta, in the direction of the current gradient:
                    e_w = rho * grad / ||grad||_2
                    theta <- theta + e_w
  2. (caller re-computes loss/gradients at the perturbed theta)
  3. second_step: undo the climb, then let the base optimizer (here AdamW)
                  take its normal step using the gradient computed AT the
                  perturbed point:
                    theta <- theta - e_w
                    base_optimizer.step()

This requires TWO forward+backward passes per training step (one at theta to
find e_w, one at theta+e_w to get the sharpness-aware gradient) — the caller
(the training loop) is responsible for that, calling first_step() then
second_step() instead of a single optimizer.step().

Default hyperparameters: rho=0.05 (neighborhood size); base optimizer is
AdamW with lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999).
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer=torch.optim.AdamW,
        rho: float = 0.05,
        adaptive: bool = False,
        **base_optimizer_kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer_kwargs)
        super().__init__(params, defaults)

        # The base optimizer owns the actual parameter groups (so per-group
        # lr/weight_decay overrides, e.g. the backbone vs. pruning-scalar
        # split, pass through untouched); SAM shares the same groups so the
        # LR scheduler can step() either object's .param_groups interchangeably.
        self.base_optimizer = base_optimizer(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # "adaptive" (ASAM) scales the perturbation by |theta|; plain
                # SAM (default here) uses the raw gradient direction.
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                # Undo the ascent step before the base optimizer's real
                # update, so the parameter update is relative to theta, not
                # theta + e_w.
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise RuntimeError(
            "SAM requires two forward/backward passes per training step — "
            "call first_step()/second_step() explicitly instead of step() "
            "(see the SAM branch in scripts/train_ct_atpt_ddp.py's training loop)."
        )

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)

    def state_dict(self):
        # The e_w perturbation state is transient (recomputed every step);
        # only the base optimizer's persistent state (Adam moments, step
        # counts) needs to survive a checkpoint save/resume.
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
