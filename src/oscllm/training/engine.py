from __future__ import annotations

import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.comm import get_world_size

from oscllm.training.data import make_loader


def _get_model_attr(model, attr: str):
    """Get attribute from model, handling both wrapped (DeepSpeed) and unwrapped."""
    if hasattr(model, "module"):
        return getattr(model.module, attr, None)
    return getattr(model, attr, None)


class Engine:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        data_config: Dict[str, Any],
        corpus_path: str = None,
        log_dir: str = "/tmp/oscllm_train",
        resume_from_checkpoint: Optional[str] = None,
        fp8_mode: bool = False,
    ):
        self.model = model
        self.config = config
        self.data_config = data_config
        self.corpus_path = corpus_path  # path to corpus.bin
        self.log_dir = Path(log_dir)
        self.resume_ckpt_path = resume_from_checkpoint
        self.global_step = 0
        self.oom_count = 0
        self.max_oom_restarts = 3
        self.fp8_mode = fp8_mode  # enable FP8 via PyTorch amp autocast
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))

    def _parse_config(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        train_cfg = self.config.get("_pretrain", {})
        data_cfg = self.config.get("_data", {})
        return train_cfg, data_cfg

    def setup_deepspeed(self):
        train_cfg, _ = self._parse_config()
        
        if self.fp8_mode:
            bf16_enabled = False
        else:
            bf16_enabled = True

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": int(train_cfg.get("grad_accum_steps", 4)),
            "steps_per_print": 10,
            "wall_clock_breakdown": False,
            "bf16": {"enabled": bf16_enabled},
            "fp16": {"enabled": not bf16_enabled},   # if fp8, enable fp16 wrapper
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "gradient_clipping": float(train_cfg.get("max_grad_norm", 2.0)),
            "prescale_gradients": False,
            "communication_data_type": "bfloat16" if bf16_enabled else "fp16",
        }
        model_params = list(self.model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(
            config=ds_config,
            model=self.model,
            model_parameters=model_params,
        )
        return engine

    def create_dataloader(self, batch_size: int):
        train_cfg, data_cfg = self._parse_config()
        #优先使用 Engine 的 corpus_path，其次用 config 中的 dataset_dir
        if self.corpus_path:
            bin_path = self.corpus_path
        else:
            bin_path = str(Path(data_cfg.get("dataset_dir", "/tmp/dataset")) / "corpus.bin")
        seq_len = int(train_cfg.get("seq_len", 4096))
        micro_batch_train_tokens = batch_size * seq_len
        loader = make_loader(bin_path=bin_path, seq_len=seq_len,
                            micro_batch_train_tokens=micro_batch_train_tokens,
                            num_workers=16)
        return loader

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, L, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
        shift_labels = targets[:, 1:].contiguous().view(-1)
        ce_loss = F.cross_entropy(shift_logits, shift_labels)
        aux_scale = self.config.get("model", {}).get("sf_aux_scale", 2e-2)
        firing_target = self.config.get("model", {}).get("target_firing", 0.30)
        aux_losses = []
        
        blocks = _get_model_attr(self.model, "blocks")
        if blocks is not None:
            for blk in blocks:
                if hasattr(blk, "last_prob") and blk.last_prob is not None:
                    prob = blk.last_prob
                    aux_loss = torch.mean((prob - firing_target) ** 2)
                    aux_losses.append(aux_loss)
        
        if aux_losses:
            aux_loss = torch.stack(aux_losses).mean()
            total_loss = ce_loss + aux_scale * aux_loss
        else:
            aux_loss = torch.tensor(0.0, device=ce_loss.device)
            total_loss = ce_loss
        
        metrics = {"ce_loss": ce_loss.detach().item(),
                   "aux_loss": aux_loss.detach().item() if hasattr(aux_loss, 'detach') else 0.0,
                   "total_loss": total_loss.detach().item()}
        return total_loss, metrics

    def compute_tflops(self, batch_size: int, seq_len: int) -> float:
        model_params = sum(p.numel() for p in self.model.parameters())
        
        tok_emb_weight = _get_model_attr(self.model, "tok_emb")
        if isinstance(tok_emb_weight, torch.nn.Module):
            model_params -= tok_emb_weight.weight.numel()
        
        pos_emb = _get_model_attr(self.model, "pos_emb")
        if isinstance(pos_emb, torch.nn.Parameter):
            model_params -= pos_emb.numel()
        
        d = self.config.get("model", {}).get("d_model", 256)
        n_layer = self.config.get("model", {}).get("n_layer", 8)
        forward_flops = 2 * batch_size * seq_len * d * d * n_layer / 1e12
        total_flops = 3 * forward_flops
        return total_flops

    def _measure_resources(self) -> Dict[str, float]:
        metrics = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(f"cuda:{i}")
                gpu_mem_used = torch.cuda.memory_allocated(f"cuda:{i}") / 1024**3
                gpu_mem_total = props.total_memory / 1024**3
                metrics[f"gpu{i}_mem_gb"] = gpu_mem_used
                metrics[f"gpu{i}_mem_pct"] = gpu_mem_used / gpu_mem_total * 100
                
                try:
                    import subprocess
                    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
                                            "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=5)
                    lines = result.stdout.strip().split("\n")
                    if i < len(lines):
                        temp, util = lines[i].split(", ")
                        metrics[f"gpu{i}_temp"] = float(temp)
                        metrics[f"gpu{i}_util_pct"] = float(util)
                except Exception:
                    pass
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            metrics["cpu_mem_gb"] = mem.used / 1024**3
            metrics["cpu_mem_pct"] = mem.percent
            metrics["cpu_util_pct"] = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
        
        return metrics

    def train(self, initial_batch_size: int | None = None,
              max_steps: int | None = None, eval_interval: int = 500,
              save_interval: int = 1000, monitor_interval: int = 50):
        train_cfg, _ = self._parse_config()
        
        if initial_batch_size is None:
            initial_batch_size = int(train_cfg.get("micro_batch_train_tokens", 524288) //
                                    train_cfg.get("seq_len", 2048))
        
        if max_steps is None:
            max_steps = int(train_cfg.get("total_steps", 25000))
        
        batch_size = initial_batch_size
        
        while self.oom_count <= self.max_oom_restarts:
            try:
                return self._train_loop(batch_size=batch_size, max_steps=max_steps,
                                       eval_interval=eval_interval, save_interval=save_interval,
                                       monitor_interval=monitor_interval)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.oom_count += 1
                    print(f"[Engine] OOM detected, reducing batch size from {batch_size} to {batch_size // 2}")
                    batch_size = max(1, batch_size // 2)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if self.oom_count > self.max_oom_restarts:
                        raise RuntimeError(f"OOM after {self.max_oom_restarts} restarts") from e
                else:
                    raise

    def _train_loop(self, batch_size: int, max_steps: int, eval_interval: int,
                   save_interval: int, monitor_interval: int):
        train_cfg, data_cfg = self._parse_config()
        
        engine = self.setup_deepspeed()
        
        world_size = get_world_size() if dist.is_initialized() else 1
        real_batch_size = batch_size * world_size
        
        loader = self.create_dataloader(real_batch_size)
        loader_iter = iter(loader)
        
        seq_len = int(train_cfg.get("seq_len", 2048))
        base_lr = float(train_cfg.get("lr", 3e-4))
        warmup_steps = int(train_cfg.get("warmup_steps", 500))
        min_lr = float(train_cfg.get("min_lr", 1e-5))
        
        start_step = self.global_step
        
        for step in range(start_step, max_steps):
            self.global_step = step
            
            if step < warmup_steps:
                lr = base_lr * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
            for param_group in engine.optimizer.param_groups:
                param_group["lr"] = lr
            
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            
            input_ids = batch["input_ids"].to(engine.device)
            targets = input_ids.clone()
            
            # autocast for mixed precision (BF16 by default, FP8 reserved for future)
            ctx_mgr = torch.cuda.amp.autocast(dtype=torch.bfloat16) \
                if torch.cuda.is_available() else nullcontext()
            with ctx_mgr:
                logits = engine(input_ids)
            loss, metrics = self.compute_loss(logits, targets)

            engine.backward(loss)
            engine.step()
            
            tflops = self.compute_tflops(real_batch_size, seq_len)
            elapsed = time.time() - self.start_time
            tps = real_batch_size * seq_len / (elapsed / max(1, step - start_step + 1)) if step > start_step else 0
            
            if step % monitor_interval == 0:
                resources = self._measure_resources()
                
                print(f"[Step {step}] loss={metrics['total_loss']:.4f} "
                      f"ce={metrics['ce_loss']:.4f} aux={metrics['aux_loss']:.4f} "
                      f"lr={lr:.2e} tflops={tflops:.1f} tps={tps:.0f}")
                
                self.writer.add_scalar("train/loss", metrics["total_loss"], step)
                self.writer.add_scalar("train/ce_loss", metrics["ce_loss"], step)
                self.writer.add_scalar("train/aux_loss", metrics["aux_loss"], step)
                self.writer.add_scalar("train/lr", lr, step)
                self.writer.add_scalar("train/tflops", tflops, step)
                self.writer.add_scalar("train/tps", tps, step)
                
                for k, v in resources.items():
                    if "gpu" in k:
                        self.writer.add_scalar(f"resources/{k}", v, step)
            
            if step > 0 and step % save_interval == 0:
                ckpt_dir = self.log_dir / f"checkpoint-{step}"
                engine.save_checkpoint(ckpt_dir)
                print(f"[Engine] Saved checkpoint to {ckpt_dir}")
        
        return {"final_step": max_steps, "log_dir": str(self.log_dir)}


# ------------------------------------------------------------------- aux loss --
def sparse_aux_loss(engine):
    """CE-scale-relative SFR sparsity loss."""
    import torch
    oscfg = engine.oscfg if hasattr(engine, 'oscfg') else getattr(engine, "osgcfg", None)
    target = float(getattr(oscfg, "target_firing", 0.30))
    scale = float(getattr(oscfg, "sf_aux_scale", 2e-2))

    probs = []
    for blk in engine.ds_engine.module.blocks:
        p = getattr(blk, "last_prob", None)
        if p is not None:
            probs.append(p.float().mean())
    if not probs:
        return torch.zeros((), device=next(engine.ds_engine.parameters()).device)

    mean_p = torch.stack(probs).mean()
    aux = (mean_p - target) ** 2 * scale
    engine.last_sparse_mean = float(mean_p.detach()) if hasattr(engine, "last_sparse_mean") else \
        setattr(engine, "last_sparse_mean", float(mean_p.detach())) or float(mean_p.detach())
    return aux


def set_grad_ckpt_on(engine, flag=True):
    """Enable/disable checkpoint for blocks."""
    engine._grad_ckpt = bool(flag)
