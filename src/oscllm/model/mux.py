"""Time-Frequency Coupler (MUX): adaptive gated fusion of CFRA (freq) & LTK (time).

A light MLP computes two weights that blend the frequency-domain branch and the
time-domain branch into a single output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OSCConfig


class MUX(nn.Module):
    def __init__(self, cfg: OSCConfig):
        super().__init__()
        d = cfg.d_model_hidden if hasattr(cfg, 'd_model_hidden') else cfg.d_model
        hsize = 4 * max(1, getattr(cfg, "mux_hidden_mult", 2))
        self.net = nn.Sequential(
            nn.Linear(d, hsize),
            nn.GELU(),
            nn.Linear(hsize, 2),          # two unbounded logits -> softmax
        )

    def forward(self, x: torch.Tensor,
                cfra_out: torch.Tensor, ltk_out: torch.Tensor) -> torch.Tensor:
        g = self.net(x)                              # (B,L,2)
        w = F.softmax(g, dim=-1)                     # (g_freq, g_time)
        gf = w[..., 0:1]
        gt = w[..., 1:2]
        return gf * cfra_out + gt * ltk_out
