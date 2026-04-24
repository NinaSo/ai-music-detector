#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_kaggle.sh /ABS/PATH/to/data/raw
# Requires:
#   - kaggle CLI installed
#   - ~/.kaggle/kaggle.json configured

OUT_DIR="${1:-$(pwd)/data/raw}"
mkdir -p "$OUT_DIR"

echo "Downloading FakeMusicCaps..."
kaggle datasets download -d awsaf49/fmc-dataset -p "$OUT_DIR/fmc" --unzip

echo "Downloading FMA small+medium..."
kaggle datasets download -d imsparsh/fma-free-music-archive-small-medium -p "$OUT_DIR/fma" --unzip

echo "Done. Files downloaded under: $OUT_DIR"
