"""Benchmark throughput and latency for DDP and a simple TP linear layer.

This script measures the forward pass latency of a linear projection under
Distributed Data Parallel (DDP) and a trivial Tensor Parallel (TP) sharding.
It does not perform training but serves as a demonstrative benchmark.  The
latencies for each strategy are collected and reported as p50/p95/p99 values
using the helper in `profiling/latency_utils.py`.

To run on two GPUs:

    CUDA_VISIBLE_DEVICES=0,1 python profiling/benchmark_tp_vs_ddp.py \
        --hidden_dim 4096 --batch_size 32 --num_steps 100

For DDP mode, launch with `torchrun`:

    torchrun --nproc_per_node=2 profiling/benchmark_tp_vs_ddp.py --ddp

"""

import argparse
import os
import time
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from profiling.latency_utils import latency_percentiles

try:
    # Import our toy TP layer
    from tp_demo.tensor_parallel_linear import TensorParallelLinear
except ImportError:
    # When executed from another directory, adjust the path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from tp_demo.tensor_parallel_linear import TensorParallelLinear


def ddp_benchmark(hidden_dim: int, batch_size: int, num_steps: int) -> None:
    """Run a simple forward benchmark under DDP."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    model = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
    model = DDP(model, device_ids=[rank])

    torch.cuda.synchronize()
    durations: List[float] = []
    for _ in range(num_steps):
        x = torch.randn(batch_size, hidden_dim, device=device)
        torch.cuda.synchronize()
        start = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        durations.append(time.time() - start)

    # Gather durations to rank 0
    durations_tensor = torch.tensor(durations, device=device)
    gather_list = [torch.empty_like(durations_tensor) for _ in range(world_size)] if rank == 0 else None
    dist.gather(durations_tensor, gather_list=gather_list, dst=0)
    if rank == 0:
        all_durations = torch.cat(gather_list).cpu().tolist()
        p50, p95, p99 = latency_percentiles(all_durations)
        print(f"DDP: p50={p50*1000:.3f}ms, p95={p95*1000:.3f}ms, p99={p99*1000:.3f}ms")


def tp_benchmark(hidden_dim: int, batch_size: int, num_steps: int) -> None:
    """Run a simple forward benchmark with a toy TP linear layer."""
    # Build a TP linear layer across visible GPUs
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if not devices:
        raise RuntimeError("No CUDA devices detected for TP benchmark")
    layer = TensorParallelLinear(hidden_dim, hidden_dim, devices)

    durations: List[float] = []
    for _ in range(num_steps):
        x = torch.randn(batch_size, hidden_dim, device=devices[0])
        torch.cuda.synchronize()
        start = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        durations.append(time.time() - start)

    p50, p95, p99 = latency_percentiles(durations)
    print(f"TP: p50={p50*1000:.3f}ms, p95={p95*1000:.3f}ms, p99={p99*1000:.3f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DDP vs TP")
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--ddp", action="store_true", help="Run in DDP mode (must be launched with torchrun)")
    args = parser.parse_args()

    if args.ddp:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        ddp_benchmark(args.hidden_dim, args.batch_size, args.num_steps)
        dist.barrier()
        dist.destroy_process_group()
    else:
        tp_benchmark(args.hidden_dim, args.batch_size, args.num_steps)


if __name__ == "__main__":
    main()
