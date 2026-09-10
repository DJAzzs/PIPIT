"""OSCLLM shared base: config + lightweight helpers.

Keep this dependency-light so the prototype runs on minimal installs.
"""
from dataclasses import asdict, field, dataclass
import torch


@dataclass
class OSCConfig:
    d_model: int = 512
    n_layer: int = 4
    n_head: int = 8
    # --- SFR ---
    sf_beta: float = 0.9          # membrane leak coefficient
    sf_theta_init: float = 1.0    # initial firing threshold (learnable via bias)
    target_firing: float = 0.30   # desired firing rate for aux loss
    fire_warmup_steps: int = 200  # steps to force full activation before sparse-loss
    sf_aux_scale: float = 3e-2    # strength of the sparsity auxiliary loss
    # --- CFRA ---
    cfra_lowpass_power: float = 0.8   # soft gate exponent gamma on freq distance
    eig_freq_mode: str = "learned"     # 'learned' per head or 'fixed'
    # --- LTK ---
    ltk_t_max: int = 5             # max RK4 iterations
    ltk_alpha: float = 1.0         # entropy->tau mapping slope
    ltk_dt: float = 0.05           # fixed RK4 step size
    # --- MUX ---
    mux_hidden_mult: int = 2       # mlp hidden multiplier
    dropout: float = 0.0
    eps_norm: float = 1e-5
    # --- sequence / embedding (added for pretraining) ---
    vocab_size: int = 32768        # must match tokenizer
    max_position: int = 4096       # learned pos embedding limit (short-context aid)
    tie_emb: bool = True           # tie input embedding & LM head


def param_ratio(shape):
    n = 1
    for s in shape:
        n *= s
    return n


# --- norm helpers (used by CFRA on complex freq-domain) ---

def real_variance_preserving_norm(x, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm applied separately to the last dim of a real-valued tensor."""
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def entropy_of_logits(logits: torch.Tensor, dim: int = -1):
    """Softmax-entropy given raw logits. Used by LTK to pick iteration depth.

    Returns tensor shaped like logits without the last dimension.
    """
    m = torch.log_softmax(logits, dim=dim)
    p = torch.exp(m)
    h = -(p * m).sum(dim=dim)          # natural-log entropy; >= 0
    return torch.clamp(h, min=0.0)


def make_pos_ids(l: int):
    return torch.arange(0, l).unsqueeze(0)


__all__ = ["OSCConfig", "param_ratio", "real_variance_preserving_norm",
           "entropy_of_logits", "make_pos_ids"]
