#!/usr/bin/env python3
"""Update oscllm_10m.json vocab_size to match Qwen's 151936."""

import json
from pathlib import Path

CONFIG_PATH = Path("/home/dja/桌面/OSC-LLM/configs/oscllm_10m.json")
VOCAB_SIZE = 151936

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

config["model"]["vocab_size"] = VOCAB_SIZE

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"Updated vocab_size to {VOCAB_SIZE}")
