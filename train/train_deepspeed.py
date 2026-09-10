from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is on path for deepspeed subprocesses
this_file = Path(__file__).resolve()
sys.path.insert(0, str(this_file.parent.parent))

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oscllm.model.base import OSCConfig
from oscllm.model.model import OSCLLM
from oscllm.training.engine import Engine


def setup_distributed():
    try:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    except Exception:
        return 0


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="OSCLLM DeepSpeed Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    parser.add_argument("--log_dir", type=str, default="/tmp/oscllm_train",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps")
    parser.add_argument("--initial_batch_size", type=int, default=None,
                        help="Initial batch size (will auto-reduce on OOM)")
    parser.add_argument("--monitor_interval", type=int, default=50,
                        help="Steps between monitoring logs")
    parser.add_argument("--fp8", action="store_true",
                        help="Enable FP8 mixed precision via PyTorch amp (Blackwell native)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank passed by DeepSpeed launcher")
    parser.add_argument("--bin", type=str, required=True,
                        help="Path to pre-tokenized corpus.bin")
    args = parser.parse_args()

    local_rank = setup_distributed()
    
    config = load_config(args.config)
    model_cfg = OSCConfig(**config.get("model", {}))
    
    if torch.cuda.is_available():
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "0"
    
    model = OSCLLM(model_cfg)
    model.train()
    
    engine = Engine(
        model=model,
        config=config,
        data_config=config.get("_data", {}),
        corpus_path=args.bin,
        log_dir=args.log_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp8_mode=args.fp8,
    )
    
    result = engine.train(
        initial_batch_size=args.initial_batch_size,
        max_steps=args.max_steps,
        monitor_interval=args.monitor_interval,
    )
    
    if local_rank == 0:
        print(f"Training completed: {result}")


if __name__ == "__main__":
    main()
