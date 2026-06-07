#!/bin/bash
set -e

DATASET="ModalityDance/PhysTool-Bench"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../data"

echo "Downloading PhysTool-Bench dataset from Hugging Face..."

pip install -qU huggingface_hub

hf download $DATASET \
  --repo-type dataset \
  --local-dir $TARGET_DIR

echo "✅ Download completed to $TARGET_DIR"