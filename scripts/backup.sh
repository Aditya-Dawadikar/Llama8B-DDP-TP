#!/usr/bin/env bash
# Trigger periodic S3 backup of a directory.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source_dir> <s3_bucket> <interval_seconds> [prefix]"
  exit 1
fi
SOURCE_DIR="$1"
BUCKET="$2"
INTERVAL="$3"
PREFIX="${4:-}"

python utils/s3_backup.py \
  --source_dir "$SOURCE_DIR" \
  --bucket "$BUCKET" \
  --prefix "$PREFIX" \
  --interval "$INTERVAL"