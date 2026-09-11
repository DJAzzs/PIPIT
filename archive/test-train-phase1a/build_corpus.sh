#!/bin/bash
set -e

PROJECT_ROOT="/home/dja/桌面/OSC-LLM"
TOKENIZER_DIR="$PROJECT_ROOT/tokenizer_qwen"
DATA_DIR="/home/dja/桌面/Test Train OSC-LLM/data"
CORPUS_DIR="$PROJECT_ROOT/dataset"

mkdir -p "$CORPUS_DIR"

POETRY_DIR="$DATA_DIR/poetry"
ANCIENT_DIR="$DATA_DIR/ancient_texts"

if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER_DIR/tokenizer.json"
    exit 1
fi

source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || true

python3 << 'EOFPYTHON'
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/dja/桌面/OSC-LLM")

from oscllm.training.data import load_tokenizer, build_corpus_bin

tokenizer_path = "/home/dja/桌面/OSC-LLM/tokenizer_qwen/tokenizer.json"
print(f"Loading tokenizer from {tokenizer_path}...")
tok = load_tokenizer(tokenizer_path)
vocab_size = tok.get_vocab_size()
print(f"Tokenizer vocab size: {vocab_size}")

sources = []
poetry_dir = Path("/home/dja/桌面/Test Train OSC-LLM/data/poetry")
ancient_dir = Path("/home/dja/桌面/Test Train OSC-LLM/data/ancient_texts")

if poetry_dir.exists() and list(poetry_dir.glob("*.txt")):
    sources.append(str(poetry_dir))
    print(f"Adding poetry dir: {poetry_dir}")
if ancient_dir.exists() and list(ancient_dir.glob("*.txt")):
    sources.append(str(ancient_dir))
    print(f"Adding ancient_texts dir: {ancient_dir}")

if not sources:
    print("WARNING: No source data found. Creating minimal corpus for testing.")
    test_txt = "/home/dja/桌面/Test Train OSC-LLM/test_fallback.txt"
    with open(test_txt, "w") as f:
        f.write("天地玄黄宇宙洪荒\n日月盈昃辰宿列张\n寒来暑往秋收冬藏\n")
    sources = [test_txt]

output_prefix = Path("/home/dja/桌面/OSC-LLM/dataset/corpus")
total_tokens = build_corpus_bin(tok, sources, output_prefix)
print(f"Total tokens: {total_tokens:,}")
EOFPYTHON

echo "Corpus build complete!"
ls -la "$CORPUS_DIR/"
