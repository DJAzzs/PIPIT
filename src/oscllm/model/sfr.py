"""Spiking-Firing Router (SFR): dynamic per-token bypass with membrane potential.

Forward uses a hard threshold; backward flows through `sigmoid(slope*margin)`
so params get real gradients. The block adds an auxiliary sparsity loss against
the *continuous* gate probability to make the firing rate converge, avoiding the
classic STE saturation dead-zone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OSCConfig


def ste_forward(margin: torch.Tensor):
    """Hard decision forward; sigmoid(4*margin) backward.

    Forward value is exactly 0/1 (safe to skip compute). Backward gradient is the
    derivative of a logistic — nonzero across the real line, so even saturated
    gates receive signal for learning sparsity.
    """
    fwd = (margin > 0.0).to(margin.dtype)
    return margin + (fwd - torch.sigmoid(4.0 * margin)).detach() * 0 \
        + fwd


class SFR(nn.Module):
    """Membrane-gated router.

      drive   = proj(x)                       # (B,L,1) raw per-token drive
      Vmem    = beta*prev + (1-beta)*drive    # leaky membrane across depth/layers
      score   = LayerNorm over tokens axis    # zero-centered -> sparse fires controllable
      margin  = theta_bias - score            # decision variable
      gate(0/1), prob(~sigmoid(-margin/τ)) returned.

    Zero-centering the drive (via LN across token positions) is essential: a raw
    softplus/V signal stays strictly >0, making low firing rates unreachable.
    The single learnable scalar `theta_bias` then linearly steers P(fire).
    """

    def __init__(self, cfg: OSCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.proj = nn.Linear(d, 1)
        self.slope = getattr(cfg, "sf_slope", 4.0)          # temperature of gate
        # threshold initialized so roughly half the LN-normalized drives fire.
        self.theta_bias = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor, membrane=None):
        drive = self.proj(x)                                             # (B,L,1)
        prev = membrane if membrane is not None else \
            x.new_zeros((), dtype=x.dtype)
        Vmem = self.cfg.sf_beta * _scalar_as(prev, drive) + \
               (1 - self.cfg.sf_beta) * drive                            # (B,L,1)

        score = F.layer_norm(Vmem.squeeze(-1), x.shape[1:2]).unsqueeze(-1)  # (B,L,1)
        margin = self.theta_bias - score                                # fire if theta > score

        gate_ste = ste_forward(margin).squeeze(-1)                       # (B,L) {0,1}+grad
        prob = torch.sigmoid(self.slope * margin.to(torch.float32)).to(x.dtype).squeeze(-1)

        return gate_ste, Vmem.squeeze(-1), prob


def _scalar_as(s, ref):
    if isinstance(s, torch.Tensor) and s.ndim > 0:
        s = s.mean().reshape(())
    return torch.as_tensor(s, device=ref.device, dtype=ref.dtype)


def firing_rate(gate: torch.Tensor):
    """Mean active fraction across batch*seq. 0<=rate<=1."""
    return gate.float().mean()
