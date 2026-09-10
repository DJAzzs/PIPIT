"""Liquid Time Kernel (LTK): per-token dynamic depth via RK4 integration of an ODE.

Instead of a fixed FFN, each token runs the same learnable vector field
f(h) = dh/dt for `tau` steps. The step count is decided per-token from input
entropy: more uncertain/important tokens get deeper inference (adaptive compute).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import OSCConfig, entropy_of_logits


class LTK(nn.Module):
    def __init__(self, cfg: OSCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        hsize = 4 * d
        # learnable vector field f : R^D -> R^D (dh/dt)
        self.f1 = nn.Linear(d, hsize)
        self.act = nn.GELU()
        self.f2 = nn.Linear(hsize, d)
        # light projection used to estimate token "entropy" for depth control.
        self.ent_proj = nn.Linear(d, 4)

    def _f(self, h: torch.Tensor) -> torch.Tensor:
        return self.f2(self.act(self.f1(h)))

    @staticmethod
    def rk4_step(h0, dt, f):
        k1 = f(h0)
        k2 = f(h0 + 0.5 * dt * k1)
        k3 = f(h0 + 0.5 * dt * k2)
        k4 = f(h0 + dt * k3)
        return h0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _tau(self, x: torch.Tensor):
        """Per-token integer step count in [1, T_max]."""
        logits = self.ent_proj(x)                       # (B,L,E)
        entr = entropy_of_logits(logits)                # (B,L), >=0  (class dim collapsed)
        # normalize: use a target peak ~ ltk_alpha; clamp to [0,1]
        scale = max(self.cfg.ltk_alpha, 1e-6)
        t_norm = (entr / scale).clamp(0.0, 1.0)         # (B,L)
        steps = torch.round(t_norm * (self.cfg.ltk_t_max - 1) + 1
                            ).long().clamp(min=1, max=self.cfg.ltk_t_max)
        return steps

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        tau = self._tau(x)                               # (B,L)
        h = x.clone()
        dt = float(self.cfg.ltk_dt)
        for i in range(1, self.cfg.ltk_t_max + 1):
            active = (tau >= i).to(h.device)             # bool (B,L): token still needs steps
            if not bool(active.any()):
                break
            nxt = self.rk4_step(h, dt, self._f)
            h = torch.where(active[..., None], nxt, h)   # broadcast (B,L,1) over D
        return h

    def depth_report(self, x: torch.Tensor):
        """Return avg & per-token steps (for profiling / balancing)."""
        tau = self._tau(x)
        return tau.float().mean(), tau
