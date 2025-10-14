#!/usr/bin/env bash
set -euo pipefail

# Download the latest community dataset for QLib
DATA_URL="https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
TARGET_DIR="$HOME/.qlib/qlib_data/cn_data"
TMP_TAR="qlib_bin.tar.gz"

mkdir -p "$TARGET_DIR"

if command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_TAR" "$DATA_URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L "$DATA_URL" -o "$TMP_TAR"
else
  echo "Error: neither wget nor curl is available." >&2
  exit 1
fi

tar -zxvf "$TMP_TAR" -C "$TARGET_DIR" --strip-components=1
rm -f "$TMP_TAR"

echo "QLib data downloaded to $TARGET_DIR"
