#!/bin/bash
set -e

POETRY_DIR="/home/dja/桌面/Test Train OSC-LLM/data/poetry"
ZIP_FILE="/tmp/chinese-poetry.zip"
RETRY=3
TIMEOUT=120

mkdir -p "$POETRY_DIR"

echo "Downloading Faraday's Chinese Poetry (Fork)..."
for i in $(seq 1 $RETRY); do
    echo "  Attempt $i..."
    if curl -L --max-time $TIMEOUT -o "$ZIP_FILE" \
        "https://github.com/faradays-studio/chinese-poetry/archive/refs/heads/main.zip" 2>/dev/null; then
        break
    fi
    sleep 5
done

if [ ! -f "$ZIP_FILE" ]; then
    echo "ERROR: Failed to download poetry zip"
    exit 1
fi

echo "Unzipping..."
unzip -q "$ZIP_FILE" -d /tmp/
rm "$ZIP_FILE"

POETRY_SRC="/tmp/chinese-poetry-main"

if [ -d "$POETRY_SRC" ]; then
    echo "Copying poetry files to $POETRY_DIR..."
    
    find "$POETRY_SRC" -name "*.txt" -type f | while read -r file; do
        filename=$(basename "$file")
        cp "$file" "$POETRY_DIR/$filename"
    done
    
    rm -rf "$POETRY_SRC"
fi

echo "Poetry data downloaded to: $POETRY_DIR"
ls -la "$POETRY_DIR/" | head -20
