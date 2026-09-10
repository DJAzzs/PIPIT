"""OSCLLM LM head / wrapper.

Stack of Blocks + token/position embedding → final norm → next-token logits.
Each Block owns its own SFR membrane (stateful across a forward call).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import OSCConfig, real_variance_preserving_norm as rnorm
from .block import Block


class OSCLLM(nn.Module):
    """Pretraining-ready wrapper: embedding + N blocks -> final norm -> LM head.

    Returns raw logits tensor (B,L,V) by default; optional dict with a
    `metrics` view when return_metrics=True.
    """

    def __init__(self, cfg: OSCConfig):
        super().__init__()
        self.cfg = cfg
        vocab_size = int(getattr(cfg, "vocab_size", 32768))
        d = cfg.d_model

        self.tok_emb = nn.Embedding(vocab_size, d)
        max_pos = getattr(cfg, "max_position", 4096)
        self.pos_emb = nn.Parameter(torch.randn(1, max_pos, d) * (d ** -0.5))
        self.dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = rnorm

        if getattr(cfg, "tie_emb", True):
            self.lm_head = None                      # weight tied to tok_emb
        else:
            self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, input_ids: torch.Tensor,
                return_metrics=False):
        B, L = input_ids.shape
        h = self.tok_emb(input_ids)                          # (B,L,D)
        if getattr(self.cfg, "use_pos", True) and \
           L <= self.pos_emb.size(1):
            h = h + self.pos_emb[:, :L].to(h.dtype)
        h = self.dropout(h)

        use_ckpt = bool(getattr(self.cfg, "grad_checkpointing", False)) \
            and self.training
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out

        x_norm = self.final_norm(h.float())
        if self.lm_head is not None:
            logits = self.lm_head(x_norm.to(h.dtype))
        else:                                          # tied weights
            logits = torch.matmul(x_norm.to(h.dtype), self.tok_emb.weight.t())

        metrics = dict(firing=[getattr(b, "fired_mean", 0.0)
                               for b in self.blocks])
        if return_metrics:
            return {"logits": logits, **metrics}
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
