"""JSON/YAML config -> OSCConfig loader + model-size helpers.

Allows `configs/*.json` to set any OSCConfig field plus extra keys ignored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .model.base import OSCConfig


_KNOWN = {f.name for f in OSCConfig.__dataclass_fields__.values()}


def coerce_cfg(obj) -> OSCConfig:
    """Build an OSCConfig from a dataclass instance / namespace dict / mapping,
    ignoring unknown keys and list/dict-valued config extras."""
    if isinstance(obj, OSCConfig):
        return obj
    d = vars(obj) if not isinstance(obj, (dict,)) else obj
    kw = {}
    for k, v in (d or {}).items():
        if k in _KNOWN and not isinstance(v, (list, dict)):
            kw[k] = v
    # apply only fields present on dataclass; extras dropped.
    return OSCConfig(**{k: v for k, v in kw.items()
                        if k in {f.name for f in OSCConfig.__dataclass_fields__.values()}})


def load_config(path: str | os.PathLike) -> OSCConfig:
    p = Path(path)
    raw: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return config_from_dict(raw)


def config_from_dict(d: Dict[str, Any]) -> OSCConfig:
    """Build an OSCConfig from a dict. Unknown keys are kept on the instance so
    pretraining-only fields (lr, batch_size,...) don't break model construction."""
    known = {f.name for f in OSCConfig.__dataclass_fields__.values()}
    cfg_kwargs = {}
    extra = {}

    # allow nested "model" dict like HF configs
    src = d.get("model", {}).copy()
    base_d = d.copy(); base_d.pop("model", None)
    for k, v in {**base_d, **src}.items():
        if isinstance(v, (dict, list)):
            continue
        (cfg_kwargs if k in known else extra)[k] = v

    cfg = OSCConfig(**{k: v for k, v in cfg_kwargs.items()})
    # attach extras as attributes for trainer use
    for k, v in {**d, **(extra)}.items():
        setattr(cfg, k, v)
    return cfg


def model_param_count(d_model, n_layer, vocab_size, tie=True, n_head=None,
                      ffn_mult=4, head_dim=32):
    """Rough analytic param estimate (blocks + embeddings), useful for sizing."""
    h = n_head or max(2, d_model // head_dim)
    dh = d_model // h
    # per-block approx: CFRA proj in+out (2*d^2) ; LTK f1/f2 (~2*4d^2) ;
    # SFR tiny; MUX ~ small -> use ~6.5 * d^2 as block weight proxy.
    per_block = 6.0 * d_model * d_model
    blocks = n_layer * per_block
    emb = (vocab_size * d_model if not tie else vocab_size * d_model)   # tok_emb (+ lm_head tied)
    embeddings = emb + d_model            # pos embedding ~ trivial
    return int(blocks + embeddings)


def suggest_config_for_params(target: float, *, vocab_size=32768, n_layer_min=4,
                              n_layer_max=64):
    """Search (d_model,n_layer) closest to target total params. Returns dict."""
    best = None; best_err = 1e18
    # sweep width*depth roughly keeping ratio from prototype (~2M@128/6layer)
    for layers in range(n_layer_min, n_layer_max + 3):
        # pick d so layer-FFN dominates while emb stays reasonable
        est_d = target / (layers * 8.0)            # crude; refine below by scan
        lo, hi = max(64, int((est_d*0.5)//16)*16), int((est_d*2//16)*16)+16
        for d in range(max(lo,72), min(hi+1, 2048), 32):
            n_est = model_param_count(d, layers, vocab_size=vocab_size)
            err = abs(n_est - target) / target
            if err < best_err:
                best_err, best = err, (d, layers)
    d, L = best
    return dict(d_model=d, n_layer=L,
                param_ref=int(model_param_count(d, L, vocab_size=vocab_size)),
                exact_est=model_param_count(d, L, vocab_size=vocab_size))
