"""Run a simple demonstration of the TensorParallelLinear layer.

This script constructs a TensorParallelLinear and performs a forward pass
to illustrate how the output dimension is distributed across devices.
"""

import argparse
import torch

from tensor_parallel_linear import TensorParallelLinear


def main():
    parser = argparse.ArgumentParser(description="Run tensor parallel demo")
    parser.add_argument("--in_features", type=int, default=1024)
    parser.add_argument("--out_features", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if not devices:
        raise RuntimeError("No CUDA devices available for TP demo")
    layer = TensorParallelLinear(args.in_features, args.out_features, devices)
    x = torch.randn(args.batch_size, args.in_features, device=devices[0])
    out = layer(x)
    print(f"Input shape: {x.shape}, output shape: {out.shape}")
    print(f"Output resides on device {out.device}")


if __name__ == "__main__":
    main()
