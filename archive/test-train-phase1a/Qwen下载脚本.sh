#!/bin/bash
set -e

TOKENIZER_DIR="/home/dja/桌面/OSC-LLM/tokenizer_qwen"
RETRY=3
TIMEOUT=60

mkdir -p "$TOKENIZER_DIR"

for i in $(seq 1 $RETRY); do
    if curl -L --max-time $TIMEOUT -o "$TOKENIZER_DIR/config.json" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/config.json" 2>/dev/null; then
        break
    fi
    sleep 3
done

for i in $(seq 1 $RETRY); do
    if curl -L --max-time $TIMEOUT -o "$TOKENIZER_DIR/mergeable_ranks.txt" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/mergeable_ranks.txt" 2>/dev/null; then
        break
    fi
    sleep 3
done

for i in $(seq 1 $RETRY); do
    if curl -L --max-time $TIMEOUT -o "$TOKENIZER_DIR/tokenizer_config.json" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer_config.json" 2>/dev/null; then
        break
    fi
    sleep 3
done

for i in $(seq 1 $RETRY); do
    if curl -L --max-time $TIMEOUT -o "$TOKENIZER_DIR/tokenizer.json" \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json" 2>/dev/null; then
        break
    fi
    sleep 3
done

ls -la "$TOKENIZER_DIR/"

VOCAB_SIZE=$(grep -o '"vocab_size"[[:space:]]*:[[:space:]]*[0-9]*' "$TOKENIZER_DIR/config.json" | grep -o '[0-9]*$')
echo "Vocab size: $VOCAB_SIZE"
