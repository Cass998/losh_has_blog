#!/usr/bin/env python3
"""CPU-only Ray runtime probe used by the veRL source course.

The script records observed values instead of asserting implementation-specific PIDs
or scheduler timing.  Run one mode at a time and keep the JSON artifact as evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


# Keep handles alive until the optional observation window finishes.  Otherwise
# Python reference counting may release a toy actor before `ray list actors` runs.
_LIVE_ACTOR_HANDLES: list[Any] = []
_LIVE_PLACEMENT_GROUPS: list[Any] = []
_PENDING_REFS_TO_CANCEL: list[ray.ObjectRef] = []


@ray.remote(num_cpus=0.25)
def task_probe(value: int) -> dict[str, Any]:
    context = ray.get_runtime_context()
    return {
        "kind": "task",
        "pid": os.getpid(),
        "node_id": context.get_node_id(),
        "task_id": context.get_task_id(),
        "value": value * 2,
    }


@ray.remote(num_cpus=0.25)
class Counter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> dict[str, Any]:
        self.value += 1
        context = ray.get_runtime_context()
        return {
            "kind": "actor_task",
            "pid": os.getpid(),
            "node_id": context.get_node_id(),
            "actor_id": context.get_actor_id(),
            "task_id": context.get_task_id(),
            "value": self.value,
        }


@ray.remote(num_cpus=0.25, max_restarts=1, max_task_retries=0)
class CrashOnce:
    def identity(self) -> dict[str, Any]:
        context = ray.get_runtime_context()
        return {
            "pid": os.getpid(),
            "actor_id": context.get_actor_id(),
            "node_id": context.get_node_id(),
        }

    def exit_process(self) -> None:
        os._exit(23)


def ref_record(ref: ray.ObjectRef) -> dict[str, str]:
    return {"repr": repr(ref), "hex": ref.hex()}


def exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
    }


def run_baseline() -> dict[str, Any]:
    task_ref = task_probe.remote(21)
    counter = Counter.options(name="course-counter").remote()
    _LIVE_ACTOR_HANDLES.append(counter)
    actor_refs = [counter.increment.remote() for _ in range(3)]
    return {
        "task_ref_before_get": ref_record(task_ref),
        "actor_refs_before_get": [ref_record(ref) for ref in actor_refs],
        "task_result": ray.get(task_ref),
        "actor_results": ray.get(actor_refs),
        "cluster_resources": ray.cluster_resources(),
        "available_resources_snapshot": ray.available_resources(),
    }


def run_placement_group() -> dict[str, Any]:
    pg = placement_group(
        bundles=[{"CPU": 0.5}, {"CPU": 0.5}],
        strategy="STRICT_PACK",
        name="course-pg",
    )
    ready_ref = pg.ready()
    ray.get(ready_ref)
    _LIVE_PLACEMENT_GROUPS.append(pg)
    workers = [
        Counter.options(
            name=f"course-rank-{bundle_index}",
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            ),
        ).remote()
        for bundle_index in range(2)
    ]
    _LIVE_ACTOR_HANDLES.extend(workers)
    results = ray.get([worker.increment.remote() for worker in workers])
    return {
        "placement_group_id": pg.id.hex(),
        "ready_ref": ref_record(ready_ref),
        "bundle_specs": pg.bundle_specs,
        "actor_results": results,
    }


def run_unschedulable(timeout_seconds: float) -> dict[str, Any]:
    @ray.remote(resources={"course_missing_resource": 1})
    def impossible_task() -> str:
        return "this should not run without the custom resource"

    ref = impossible_task.remote()
    ready, remaining = ray.wait([ref], timeout=timeout_seconds)
    if remaining:
        # Preserve the unsatisfied demand during --hold-seconds so a second shell
        # can observe it.  main() cancels it immediately before Ray shuts down.
        _PENDING_REFS_TO_CANCEL.extend(remaining)
    return {
        "ref": ref_record(ref),
        "wait_timeout_seconds": timeout_seconds,
        "ready_count": len(ready),
        "remaining_count": len(remaining),
        "expected_interpretation": (
            "remaining_count=1 means the logical resource request could not be admitted "
            "during the observation window; confirm the cause with `ray status`."
        ),
    }


def run_actor_crash() -> dict[str, Any]:
    actor = CrashOnce.options(name="course-crash-once").remote()
    _LIVE_ACTOR_HANDLES.append(actor)
    before = ray.get(actor.identity.remote())
    failure: dict[str, str] | None = None
    try:
        ray.get(actor.exit_process.remote())
    except BaseException as exc:  # the exact Ray exception depends on restart timing
        failure = exception_record(exc)

    after: dict[str, Any] | None = None
    restart_failure: dict[str, str] | None = None
    try:
        after = ray.get(actor.identity.remote(), timeout=30)
    except BaseException as exc:
        restart_failure = exception_record(exc)

    return {
        "before": before,
        "crashing_call_failure": failure,
        "after_restart_attempt": after,
        "after_restart_failure": restart_failure,
        "interpretation": (
            "A stable actor_id with a changed pid indicates process reconstruction. "
            "It does not restore application state unless the application does so."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("baseline", "placement-group", "unschedulable", "actor-crash"),
        default="baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ray-artifacts"))
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--wait-timeout", type=float, default=3.0)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="start the local dashboard; keep it private or use an SSH tunnel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = ray.init(num_cpus=2, include_dashboard=args.dashboard)
    try:
        runners = {
            "baseline": run_baseline,
            "placement-group": run_placement_group,
            "unschedulable": lambda: run_unschedulable(args.wait_timeout),
            "actor-crash": run_actor_crash,
        }
        artifact = {
            "mode": args.mode,
            "ray_version": ray.__version__,
            "driver_pid": os.getpid(),
            "dashboard_url": context.dashboard_url,
            "observed_at_unix_seconds": time.time(),
            "result": runners[args.mode](),
        }
        output_path = args.output_dir / f"{args.mode}.json"
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(output_path)
        if args.hold_seconds > 0:
            print(f"holding Ray for {args.hold_seconds}s; inspect it from another shell")
            time.sleep(args.hold_seconds)
    finally:
        for ref in _PENDING_REFS_TO_CANCEL:
            ray.cancel(ref, force=True)
        for pg in _LIVE_PLACEMENT_GROUPS:
            remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
