#!/usr/bin/env python3
"""Pure-stdlib rank/coordinate/group audit for a lexicographic teaching mesh.

This is not a replacement for printing the framework's actual process groups.
It makes an expected table that you can diff against runtime logs.
"""

from __future__ import annotations

import argparse
import itertools
import math


def parse_axis(value: str) -> tuple[str, int]:
    try:
        name, raw_degree = value.split("=", 1)
        degree = int(raw_degree)
    except ValueError as error:
        raise argparse.ArgumentTypeError("axis must look like NAME=DEGREE") from error
    if not name or degree < 1:
        raise argparse.ArgumentTypeError("axis name must be nonempty and degree >= 1")
    return name, degree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "axes",
        nargs="+",
        type=parse_axis,
        help="Lexicographic mesh axes, for example dp=2 tp=4 pp=2",
    )
    parser.add_argument("--world-size", type=int)
    args = parser.parse_args()

    names = [name for name, _ in args.axes]
    degrees = [degree for _, degree in args.axes]
    if len(set(names)) != len(names):
        raise ValueError(f"axis names must be unique: {names}")

    world_size = math.prod(degrees)
    if args.world_size is not None and args.world_size != world_size:
        raise ValueError(
            f"degree product is {world_size}, but --world-size is {args.world_size}"
        )

    coordinates = list(itertools.product(*(range(degree) for degree in degrees)))
    rank_by_coordinate = {coordinate: rank for rank, coordinate in enumerate(coordinates)}

    print("rank\t" + "\t".join(names) + "\tgroups")
    for rank, coordinate in enumerate(coordinates):
        group_descriptions = []
        for axis_index, axis_name in enumerate(names):
            members = []
            for axis_coordinate in range(degrees[axis_index]):
                peer_coordinate = list(coordinate)
                peer_coordinate[axis_index] = axis_coordinate
                members.append(rank_by_coordinate[tuple(peer_coordinate)])
            group_descriptions.append(f"{axis_name}={members}")
        print(
            f"{rank}\t"
            + "\t".join(map(str, coordinate))
            + "\t"
            + "; ".join(group_descriptions)
        )


if __name__ == "__main__":
    main()
