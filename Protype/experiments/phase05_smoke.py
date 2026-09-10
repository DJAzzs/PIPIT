"""Phase 0.5 prototype feasibility check for three risky ideas:

 (A) SFR firing gate trains to near target sparsity without collapsing.
 (B) The full oscillatory block learns a next-token task with stable grads
     (no NaN / exploding norm).
 (C) LTK adaptive depth runs RK4 successfully.

Run from Protype/:  PYTHONPATH=src python experiments/phase05_smoke.py [--cuda]
(If src isn't importable, set it relative: PYTHONPATH=$(realpath ../src))
"""
from __future__ import annotations

import argparse
import sys
import os
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    # ensure the reusable src dir is importable
    root = os.path.dirname(os.path.abspath(__file__))        # .../Protype/experiments
    proj = os.path.join(root, "..", "src")
    if proj not in sys.path:
        sys.path.insert(0, proj)

    from oscllm.model.base import OSCConfig

    device = args.device or ("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"[phase05] device={device} d_model={args.d_model} seq_len={args.seq_len}")
    torch.manual_seed(0)

    n_head = max(args.d_model // 32, 2)
    cfg = OSCConfig(
        d_model=args.d_model,
        n_layer=3,
        n_head=n_head,
        target_firing=0.15,
        fire_warmup_steps=150,
        sf_aux_scale=2e-2,
        ltk_t_max=5,
        ltk_dt=0.01,               # small dt keeps RK4 stable at D=128
        cfra_lowpass_power=2.0,
    )

    from oscllm.model import OSCLLM

    model = OSCLLM(cfg, vocab_size=args.vocab).to(device)
    print(f"[phase05] params={model.count_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def sample(x):
        return torch.randint(2, args.vocab, (x, args.seq_len), device=device)

    # -------- training loop ---------
    model.train()
    step = 0
    import time
    t0 = time.time()

    for step in range(args.steps):
        xin = sample(args.bsz)
        opt.zero_grad(set_to_none=True)

        out = model(xin)
        logits = out["logits"] if isinstance(out, dict) else \
            (out[0] if isinstance(out, tuple) else out)      # (B,L,V)
        ce = F.cross_entropy(logits.transpose(1, 2).contiguous(), xin)

        # adaptive sparsity aux loss from each block's *continuous* firing prob.
        use_sparse = step >= cfg.fire_warmup_steps
        probs, fired_fracs = [], []
        for blk in model.blocks:
            p = getattr(blk, "last_prob", None)
            if p is not None:
                probs.append(p.float().mean())               # differentiable mean prob
            else:
                probs.append(torch.as_tensor(float("nan"), device=device))
            fired_fracs.append(float(getattr(blk, "fired_mean", 0.0)))

        frac_prob = sum(probs) / len(model.blocks)
        frac_meas = sum(fired_fracs) / (len(fired_fracs) or 1)

        aux_scale = cfg.sf_aux_scale if use_sparse else 0.0
        target_t = torch.as_tensor(cfg.target_firing, device=device)
        aux_loss = (frac_prob - target_t).pow(2) * aux_scale

        loss = ce + aux_loss
        loss.backward()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % 100 == 0 or not bool(torch.isfinite(loss)):
            print(f"step {step:5d}  ce={ce.item():.3f}  aux={aux_loss.item():.4f}"
                  f"  fire_frac={frac_meas*100:.1f}%"
                  f"  gnorm={gnorm:.2e}")
        if not torch.isfinite(loss):
            print("[phase05] !! NaN detected -> experiment failed (check dt / lr)")
            return {"ok": False}

    dur = time.time() - t0
    final_ce = ce.item()
    frac_meas_total = sum(float(getattr(b, "fired_mean", 0.0)) for b in model.blocks) \
        / len(model.blocks)
    print(f"[phase05] DONE in {dur:.1f}s  final_loss={final_ce:.3f} "
          f"fire_mean={frac_meas_total:.2%}")

    # -------- (C) LTK adaptive depth check ---------
    from oscllm.model import LTK
    ltk = LTK(cfg).to(device)
    xdum = torch.randn(args.bsz, args.seq_len, cfg.d_model, device=device)
    with torch.no_grad():
        avg_tau, _tau = ltk.depth_report(xdum)
    print(f"[phase05] (C) LTK adaptive depth: mean_steps={avg_tau.item():.2f} "
          f"over Tmax={cfg.ltk_t_max}, varying="
          f"{_tau.max().item() > _tau.min().item()}")

    return {"ok": torch.isfinite(torch.as_tensor(final_ce)).item()}


if __name__ == "__main__":
    main()
