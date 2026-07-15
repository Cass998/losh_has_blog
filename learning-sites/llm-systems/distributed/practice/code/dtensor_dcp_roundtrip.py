#!/usr/bin/env python3
"""Save and load a sharded CPU DTensor with Distributed Checkpoint.

Run with a new checkpoint path:
  torchrun --standalone --nproc-per-node=2 dtensor_dcp_roundtrip.py \
    --checkpoint /tmp/dcp-dtensor-lab

The script deliberately preallocates the target DTensor before dcp.load().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=16)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.numel % world_size:
        raise ValueError("--numel must be evenly divisible by world_size for this lab")

    already_exists = torch.tensor(int(args.checkpoint.exists()), dtype=torch.int32)
    dist.all_reduce(already_exists, op=dist.ReduceOp.MAX)
    if already_exists.item():
        raise FileExistsError(
            f"{args.checkpoint} already exists; use a new path or remove the old lab output"
        )

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("shard",))
    expected = torch.arange(args.numel, dtype=torch.float32) * 1.5 - 3
    sharded_weight = distribute_tensor(expected, mesh, [Shard(0)], src_data_rank=0)
    step = torch.tensor(7, dtype=torch.int64)

    state = {"model.weight": sharded_weight, "train.step": step}
    dcp.save(state, checkpoint_id=str(args.checkpoint))
    dist.barrier()

    target_weight = distribute_tensor(
        torch.zeros_like(expected), mesh, [Shard(0)], src_data_rank=0
    )
    target_step = torch.tensor(-1, dtype=torch.int64)
    target_state = {"model.weight": target_weight, "train.step": target_step}
    dcp.load(target_state, checkpoint_id=str(args.checkpoint))

    restored = target_state["model.weight"].full_tensor()
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    if int(target_state["train.step"]) != 7:
        raise AssertionError(f"step was not restored: {target_state['train.step']}")

    local = target_state["model.weight"].to_local()
    local_summaries = [None for _ in range(world_size)]
    dist.all_gather_object(
        local_summaries,
        {"rank": rank, "shape": tuple(local.shape), "sum": float(local.sum())},
    )
    if rank == 0:
        print(
            "PASS",
            {
                "checkpoint": str(args.checkpoint),
                "world_size": world_size,
                "logical_shape": tuple(restored.shape),
                "logical_sum": float(restored.sum()),
                "step": int(target_state["train.step"]),
                "local_shards": local_summaries,
            },
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
