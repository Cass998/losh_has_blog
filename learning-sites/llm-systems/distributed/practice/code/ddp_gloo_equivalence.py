#!/usr/bin/env python3
"""Two-rank CPU/Gloo DDP correctness gate.

Run:
  torchrun --standalone --nproc-per-node=2 ddp_gloo_equivalence.py
  torchrun --standalone --nproc-per-node=2 ddp_gloo_equivalence.py --accumulate

This checks DDP against a single-process reference using the same global batch.
It is a correctness exercise, not a GPU/NCCL performance benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


INPUTS = torch.tensor(
    [
        [1.0, 2.0, -1.0],
        [0.5, -1.0, 3.0],
        [2.0, 0.0, 1.0],
        [-1.0, 2.5, 0.5],
    ]
)
TARGETS = torch.tensor([[0.25], [1.5], [-0.5], [2.0]])


def build_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(3, 4, bias=True),
        nn.Tanh(),
        nn.Linear(4, 1, bias=False),
    )
    return model


def flat_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().view(-1) for parameter in model.parameters()])


def flat_gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().view(-1) for parameter in model.parameters()])


def reference_step(initial_state: dict[str, torch.Tensor], learning_rate: float):
    model = build_model()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = nn.functional.mse_loss(model(INPUTS), TARGETS, reduction="mean")
    loss.backward()
    gradients = flat_gradients(model).clone()
    optimizer.step()
    return float(loss), gradients, flat_parameters(model).clone()


def distributed_step(
    ddp: DDP,
    rank: int,
    learning_rate: float,
    accumulate: bool,
):
    optimizer = torch.optim.SGD(ddp.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)

    local_inputs = INPUTS[rank * 2 : (rank + 1) * 2]
    local_targets = TARGETS[rank * 2 : (rank + 1) * 2]

    if accumulate:
        local_loss_sum = 0.0
        for microbatch in range(2):
            sync_context = ddp.no_sync() if microbatch == 0 else contextlib.nullcontext()
            with sync_context:
                output = ddp(local_inputs[microbatch : microbatch + 1])
                # Each rank has two equally sized microbatches. Dividing by two
                # makes their accumulated local gradient a local-batch mean;
                # DDP then averages those means across two equal-size ranks.
                loss = nn.functional.mse_loss(
                    output,
                    local_targets[microbatch : microbatch + 1],
                    reduction="mean",
                ) / 2
                loss.backward()
                local_loss_sum += float(loss.detach())
        local_loss = local_loss_sum
    else:
        output = ddp(local_inputs)
        loss = nn.functional.mse_loss(output, local_targets, reduction="mean")
        loss.backward()
        local_loss = float(loss.detach())

    gradients = flat_gradients(ddp.module).clone()
    optimizer.step()
    return local_loss, gradients, flat_parameters(ddp.module).clone()


def all_ranks_equal(value: torch.Tensor, *, atol: float = 1e-7) -> None:
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    for peer_rank, peer_value in enumerate(gathered):
        torch.testing.assert_close(
            value,
            peer_value,
            atol=atol,
            rtol=0,
            msg=lambda message: f"rank state differs from rank {peer_rank}: {message}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Use two microbatches and DDP.no_sync() on the first one.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This gate intentionally expects world_size=2, got {world_size}")

    torch.manual_seed(2026)
    model = build_model()
    initial_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    reference_loss, reference_gradients, reference_parameters = reference_step(
        initial_state, args.learning_rate
    )

    ddp = DDP(model)
    local_loss, ddp_gradients, ddp_parameters = distributed_step(
        ddp, rank, args.learning_rate, args.accumulate
    )

    torch.testing.assert_close(ddp_gradients, reference_gradients, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ddp_parameters, reference_parameters, atol=1e-6, rtol=1e-6)
    all_ranks_equal(ddp_gradients)
    all_ranks_equal(ddp_parameters)

    local_loss_tensor = torch.tensor(local_loss)
    dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
    distributed_global_loss = float(local_loss_tensor / world_size)
    if abs(distributed_global_loss - reference_loss) > 1e-6:
        raise AssertionError(
            f"global loss mismatch: DDP={distributed_global_loss}, reference={reference_loss}"
        )

    if rank == 0:
        print(
            "PASS",
            {
                "backend": dist.get_backend(),
                "world_size": world_size,
                "accumulate": args.accumulate,
                "global_loss": distributed_global_loss,
                "gradient_checksum": float(ddp_gradients.sum()),
                "parameter_checksum_after_step": float(ddp_parameters.sum()),
                "pid": os.getpid(),
            },
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
