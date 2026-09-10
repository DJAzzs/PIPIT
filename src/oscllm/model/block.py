"""OSCLLM block: SFR -> (parallel CFRA & LTK) -> MUX fuse, with gated residual.

  input x
    |
  SFR   -> gate (per-token 0/1), membrane V updated for next layer
    |
      if not fired: out = residual(x)
      else:
         n_a = LN(CFRA(x))     # frequency branch
         n_b = LN(LTK(x))      # time/dynamic-depth branch
         fused = MUX(norm(x), cfra_out, ltk_out)   [weighted blend]
         block_out = gate * fused
    |
  output = x + block_out

Normalizations follow a pre-norm pattern for training stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import OSCConfig
from .sfr import SFR
from .cfra import CFRA
from .ltk import LTK
from .mux import MUX


class PreNorm(nn.Module):
    def __init__(self, cfg: OSCConfig, dim=None):
        super().__init__()
        d = dim or cfg.d_model
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x)


class Block(nn.Module):
    def __init__(self, cfg: OSCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.sfr = SFR(cfg)
        # two parallel branches each behind its own layernorm (pre-norm style)
        self.norm_cfra = PreNorm(cfg, d)
        self.cfra = CFRA(cfg)

        self.norm_ltk = PreNorm(cfg, d)
        self.ltk = LTK(cfg)

        self.mux_in_norm = PreNorm(cfg, d)
        self.mux = MUX(cfg)

    def forward(self, x: torch.Tensor, membrane=None):
        # SFR returns (hard STE {0/1}, membrane, differentiable P(fire)) in BOTH
        # train & eval; the hard value is 0/1 so skip-masking stays clean.
        gate_hard, _Vm, prob_mem = self.sfr(x)
        fired_mask = (gate_hard > 0)

        # expose differentiable signal for sparsity aux loss + monitor metrics
        self.last_gate = gate_hard            # {0/1}-ish value (STE gradient path ok)
        # keep the differentiable firing probability for the sparse aux loss;
        # in eval we don't need it so store None to avoid accidental .backward.
        self.last_prob = prob_mem if self.training else None
        self.fired_mean = float(fired_mask.float().mean())

        b_freq_raw = self.cfra(self.norm_cfra(x))
        b_time_raw = self.ltk(self.norm_ltk(x))

        fused = self.mux(self.mux_in_norm(x), b_freq_raw, b_time_raw)   # (B,L,D)

        gate4d = fired_mask.float().unsqueeze(-1).detach()
        block_out = gate4d * fused
        out = x + block_out

        return out
