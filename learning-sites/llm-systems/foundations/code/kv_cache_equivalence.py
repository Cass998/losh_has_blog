#!/usr/bin/env python3
"""Prove tiny causal-attention KV caching and calculate cache memory."""

from __future__ import annotations

import argparse
import json
import math


TOKENS = (
    (1.0, 0.0, 0.5, -0.5),
    (0.0, 1.0, -0.5, 0.5),
    (0.5, 0.5, 1.0, 0.0),
    (-0.5, 1.0, 0.0, 1.0),
)
W_Q = ((0.5, -0.2, 0.3, 0.1), (-0.1, 0.4, 0.2, 0.6))
W_K = ((0.3, 0.1, -0.4, 0.5), (0.2, -0.3, 0.6, 0.1))
W_V = ((0.7, 0.0, 0.2, -0.1), (-0.2, 0.5, 0.1, 0.4))


def project(vector: tuple[float, ...], weights: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(sum(value * weight for value, weight in zip(vector, row)) for row in weights)


def attention(
    query: tuple[float, ...],
    keys: list[tuple[float, ...]],
    values: list[tuple[float, ...]],
) -> tuple[float, ...]:
    scale = math.sqrt(len(query))
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    peak = max(scores)
    exponentials = [math.exp(score - peak) for score in scores]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    return tuple(
        sum(probability * value[index] for probability, value in zip(probabilities, values))
        for index in range(len(values[0]))
    )


def full_recompute(tokens: tuple[tuple[float, ...], ...]) -> list[tuple[float, ...]]:
    outputs = []
    for end in range(1, len(tokens) + 1):
        prefix = tokens[:end]
        query = project(prefix[-1], W_Q)
        keys = [project(token, W_K) for token in prefix]
        values = [project(token, W_V) for token in prefix]
        outputs.append(attention(query, keys, values))
    return outputs


def cached_decode(tokens: tuple[tuple[float, ...], ...]) -> list[tuple[float, ...]]:
    keys: list[tuple[float, ...]] = []
    values: list[tuple[float, ...]] = []
    outputs = []
    for token in tokens:
        query = project(token, W_Q)
        keys.append(project(token, W_K))
        values.append(project(token, W_V))
        outputs.append(attention(query, keys, values))
    return outputs


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=positive_int, default=32)
    parser.add_argument("--kv-heads", type=positive_int, default=8)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--bytes-per-element", type=positive_int, default=2)
    parser.add_argument("--tokens", type=positive_int, default=8192)
    parser.add_argument("--requests", type=positive_int, default=1)
    args = parser.parse_args()

    expected = full_recompute(TOKENS)
    actual = cached_decode(TOKENS)
    for step, (left, right) in enumerate(zip(expected, actual), start=1):
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(left, right)):
            raise AssertionError(f"step {step} differs: {left} != {right}")

    bytes_per_token = 2 * args.layers * args.kv_heads * args.head_dim * args.bytes_per_element
    total_bytes = bytes_per_token * args.tokens * args.requests
    print("PASS: cached decode matches full causal-attention recomputation for every step")
    print(
        json.dumps(
            {
                "layers": args.layers,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "bytes_per_element": args.bytes_per_element,
                "tokens_per_request": args.tokens,
                "requests": args.requests,
                "bytes_per_token": bytes_per_token,
                "total_gib": total_bytes / 2**30,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
