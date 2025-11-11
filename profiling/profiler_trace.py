"""Example of using PyTorch profiler to capture forward/backward traces.

This script loads a small model (e.g. an MLP) and runs a few training steps
while collecting profiling information.  The resulting trace can be viewed in
TensorBoard or with Chrome tracing tools.  Customize the model and data as
needed.

Run with:

    python profiling/profiler_trace.py --steps 20 --warmup 2 --active 5

"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule


class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def run_profiling(steps: int, warmup: int, active: int, hidden_dim: int, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Dummy data
    data = torch.randn(batch_size, hidden_dim, device=device)
    target = torch.randn(batch_size, hidden_dim, device=device)

    sched = schedule(wait=warmup, warmup=warmup, active=active, repeat=1)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched,
                 on_trace_ready=lambda prof: prof.export_chrome_trace("results/trace.json"),
                 record_shapes=True,
                 profile_memory=True) as prof:
        for step in range(steps):
            optimizer.zero_grad()
            with record_function("forward"):
                out = model(data)
                loss = (out - target).pow(2).mean()
            with record_function("backward"):
                loss.backward()
                optimizer.step()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("Trace saved to results/trace.json")


def main():
    parser = argparse.ArgumentParser(description="PyTorch profiler example")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps before profiling")
    parser.add_argument("--active", type=int, default=5, help="Number of active profiling steps")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_profiling(args.steps, args.warmup, args.active, args.hidden_dim, args.batch_size)


if __name__ == "__main__":
    main()
