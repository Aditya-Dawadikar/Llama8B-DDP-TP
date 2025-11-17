"""Periodic backup of checkpoints to Amazon S3.

This script uploads the contents of a local directory to an S3 bucket on a
recurring interval.  It is intended to run alongside training to ensure that
checkpoints are stored off‑machine in case of failures.  The S3 client uses
credentials from the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_SESSION_TOKEN, etc.).

Example usage:

    python utils/s3_backup.py \
        --source_dir runs/llama3_lora_sst2 \
        --bucket my‑bucket \
        --prefix backups/llama_sst2 \
        --interval 3600

"""

import argparse
import os
import time
import threading
from typing import Optional

import boto3


def upload_directory(source_dir: str, bucket: str, prefix: str = "") -> None:
    """Upload all files under `source_dir` to `bucket` with the given prefix."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(source_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # Construct the key relative to source_dir
            rel_path = os.path.relpath(local_path, source_dir)
            key = os.path.join(prefix, rel_path).replace(os.sep, "/")
            try:
                s3.upload_file(local_path, bucket, key)
                print(f"[s3_backup] Uploaded {local_path} to s3://{bucket}/{key}")
            except Exception as e:
                print(f"[s3_backup] Failed to upload {local_path}: {e}")


def periodic_backup(source_dir: str, bucket: str, prefix: str, interval: int) -> None:
    """Continuously back up `source_dir` to S3 every `interval` seconds."""
    while True:
        upload_directory(source_dir, bucket, prefix)
        time.sleep(interval)


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = argparse.ArgumentParser(description="Periodic S3 backup of a directory")
    parser.add_argument("--source_dir", type=str, required=True, help="Local directory to back up")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix under which to store files")
    parser.add_argument("--interval", type=int, default=3600, help="Backup interval in seconds")
    parsed = parser.parse_args(args=args)

    print(f"[s3_backup] Starting periodic backup from {parsed.source_dir} to s3://{parsed.bucket}/{parsed.prefix} every {parsed.interval}s")
    periodic_backup(parsed.source_dir, parsed.bucket, parsed.prefix, parsed.interval)


if __name__ == "__main__":
    main()
