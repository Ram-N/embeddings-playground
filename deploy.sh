#!/bin/bash
set -e

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_DIR="/home/ram/projects/Lab/embeddings-playground"

MSG="${1:-Update}"

echo "==> Pushing to GitHub..."
cd "$SOURCE_DIR"
git add -A
git commit -m "$MSG" || echo "Nothing new to commit."
git push

echo "==> Pushing to Hugging Face Spaces..."
cp "$SOURCE_DIR/app.py" "$HF_DIR/"
cp "$SOURCE_DIR/requirements.txt" "$HF_DIR/"
cd "$HF_DIR"
git add app.py requirements.txt
git commit -m "$MSG" || echo "Nothing new to commit."
git push

echo "==> Done. Space will rebuild at https://huggingface.co/spaces/Ram-N/embeddings-playground"
