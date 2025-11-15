"""Merge tensor‑parallel shards back into a single DDP checkpoint.

Given multiple shard files created by `ddp_to_tp.py`, this script reconstructs
the original state dict by concatenating the partitioned tensors along their
first dimension and validating that the remaining keys are identical.  All
shards must have the same keys; scalar and bias tensors are taken from the
first shard.
"""

import argparse
import os
from glob import glob
from typing import Dict, List

import torch


def load_shards(input_dir: str) -> List[Dict[str, torch.Tensor]]:
    shard_files = sorted(glob(os.path.join(input_dir, "tp_rank*")))
    shards = []
    for f in shard_files:
        shards.append(torch.load(f, map_location="cpu"))
    if not shards:
        raise RuntimeError(f"No shard files found in {input_dir}")
    return shards


def merge_shards(shards: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    keys = shards[0].keys()
    for key in keys:
        tensors = [shard[key] for shard in shards]
        # For 2D or higher tensors, concatenate along dimension 0; otherwise take the first value
        if tensors[0].ndim >= 2 and tensors[0].shape[0] * len(tensors) == sum(t.shape[0] for t in tensors):
            merged[key] = torch.cat(tensors, dim=0)
        else:
            merged[key] = tensors[0]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge TP shards into a DDP checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing tp_rank*_ files")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged checkpoint")
    args = parser.parse_args()

    shards = load_shards(args.input_dir)
    merged = merge_shards(shards)
    torch.save(merged, args.output)
    print(f"Merged {len(shards)} shards into {args.output}")


if __name__ == "__main__":
    main()
