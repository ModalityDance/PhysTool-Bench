#!/bin/bash
set -e

DATASET="ModalityDance/PhysTool-Bench"
TARGET_DIR="../data"

echo "Downloading PhysTool-Bench dataset from Hugging Face..."

pip install -qU huggingface_hub

hf download $DATASET \
  --repo-type dataset \
  --local-dir $TARGET_DIR \

echo "✅ Download completed to $TARGET_DIR"