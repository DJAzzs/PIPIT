"""Complex-Frequency Resonant Attention (CFRA) — implemented as a spectral filter.

Deliberately does NOT build an n*n QK matrix. Instead:
  1. project input into heads,
  2. RFFT along the sequence dim  -> frequency domain,
  3. apply a learnable per-head elementwise gate to each frequency bin
     (a resonance band centered on that head's eigenfrequency),
  4. IRFFT back to time, then renormalize.

Cost O(n log n). Expressiveness from number of heads + learned band centers
plus the output mixing projection.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import OSCConfig


class CFRA(nn.Module):
    def __init__(self, cfg: OSCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        h = cfg.n_head
        assert d % h == 0, "d_model must be divisible by n_head"
        self.h = h
        self.head_dim = dh = d // h

        # input projection (shared across heads)
        self.proj_in = nn.Linear(d, d)

        init_eigs = torch.linspace(0.02, 0.98, steps=h)     # normalized freqs [0,1]
        if cfg.eig_freq_mode == "learned":
            self.eigen = nn.Parameter(init_eigs.clone())
        else:
            self.register_buffer("eigen", init_eigs)

        self.proj_out = nn.Linear(d, d)
        # normalization to keep freq-domain magnitudes bounded (real part LN)
        from .base import real_variance_preserving_norm as _rln
        self.rnorm = _rln

    def _gate(self, L: int):
        nobin = L // 2 + 1
        freq = torch.linspace(0.0, 1.0, steps=nobin,
                              device=self.eigen.device).view(-1, 1)   # (bin,1)
        eigc = self.eigen.view(1, -1).clamp(1e-3, 1 - 1e-3)           # (1,h)
        dist2 = (freq - eigc)**2                                       # (bin,h)
        width = max(self.cfg.cfra_lowpass_power, 0.0)
        gate = torch.exp(-width * dist2.double())                        # (bin,h)
        return gate.float()

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        v = self.proj_in(x).view(B, L, self.h, self.head_dim)\
                             .permute(0, 2, 1, 3)      # (B,H,L,DH)

        gate_bh = self._gate(L)                        # (bin,h)
        gain = gate_bh.t().unsqueeze(0).unsqueeze(-1)  # (1,h,1,1)? no => (1,h,bin,1)
        gain_in_head_axis = gate_bh.t()                # (h, bin)
        g = gain_in_head_axis.unsqueeze(0).unsqueeze(-1)   # (1,h,nobin,DH?) wrong dims

        V_hat = torch.fft.rfft(v.float(), dim=2)       # (B,H,L/2+1,DH)

        # gate shape must broadcast with (B, H, bin, DH):  =>(1,H,bin,1)
        gain4d = gain_in_head_axis[None, :, :, None]   # (1,H,nobin,1)
        V_gated = V_hat * gain4d

        v_out = torch.fft.irfft(V_gated, n=L, dim=2)   # (B,H,L,DH)
        out = self.rnorm(v_out.float()).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.proj_out(out.to(x.dtype))
