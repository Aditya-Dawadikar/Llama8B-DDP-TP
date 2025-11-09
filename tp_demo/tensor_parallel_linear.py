"""A minimal tensor parallel linear layer for demonstration purposes.

This layer splits the output dimension of a linear projection across multiple
CUDA devices.  On forward, the input is broadcast to all devices, each
partition performs its own matrix multiplication and bias addition, and the
outputs are gathered back to the first device.

Note: This implementation is simplistic and intended for educational use.
Optimisations such as overlapping communication and computation are omitted.
"""

from typing import Iterable, List
import torch
import torch.nn as nn


class TensorParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, devices: Iterable[torch.device]):
        super().__init__()
        self.devices: List[torch.device] = list(devices)
        if len(self.devices) == 0:
            raise ValueError("At least one CUDA device is required")

        # Compute partition sizes along the output dimension
        partitions = self._partition_sizes(out_features, len(self.devices))
        self.parts: nn.ModuleList = nn.ModuleList()
        for part_size, device in zip(partitions, self.devices):
            linear = nn.Linear(in_features, part_size, bias=True).to(device)
            self.parts.append(linear)

    @staticmethod
    def _partition_sizes(total: int, n: int) -> List[int]:
        base = total // n
        sizes = [base] * n
        # Distribute remainder
        remainder = total - base * n
        for i in range(remainder):
            sizes[i] += 1
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  Assumes input `x` resides on the first device."""
        # Broadcast input to all devices
        inputs = [x.to(dev) if x.device != dev else x for dev in self.devices]
        outputs: List[torch.Tensor] = []
        for inp, linear in zip(inputs, self.parts):
            out = linear(inp)
            outputs.append(out)
        # Gather back to first device along last dimension
        return torch.cat([out.to(self.devices[0]) for out in outputs], dim=-1)
