#!/usr/bin/env python3
"""Trajectory-length measurement utilities for vLLM + MATH dataset.

This module is primarily consumed by ``sweep_trajectory_lengths.py`` but also
works as a standalone CLI for measuring a single checkpoint.

Module API
----------
.. code-block:: python

    from measure_avg_trajectory_length import (
        load_math_dataset,
        format_math_prompt,
        compute_token_counts,
        print_stats,
    )

    dataset, _ = load_math_dataset(split="test", num_samples=200, seed=42)
    prompts = [format_math_prompt(row["problem"]) for row in dataset]
    counts = compute_token_counts(
        model="./Qwen2.5-Math-1.5B",
        prompts=prompts,
        max_tokens=4096,
    )
    print_stats("./Qwen2.5-Math-1.5B", "test", 200, 4096, counts)

Standalone CLI
--------------
python scripts/measure_avg_trajectory_length.py \\
    --model ./Qwen2.5-Math-1.5B \\
    --num-samples 100 \\
    --max-tokens 4096
"""

import argparse
import statistics

MATH_DATASET_CANDIDATES = [
    "hendrycks/competition_math",
    "EleutherAI/hendrycks_math",
    "Maxwell-Jia/MATH",
    "jeggers/competition_math",
]


def format_math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. Show your reasoning and provide "
        "the final answer.\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


def load_math_dataset(split: str, num_samples: int, seed: int):
    """Load and shuffle ``num_samples`` rows from the MATH dataset.

    Tries each ID in ``MATH_DATASET_CANDIDATES`` in order, falling back to the
    next on failure.  Returns ``(dataset, source_id)``.
    """
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    errors = {}
    for dataset_id in MATH_DATASET_CANDIDATES:
        try:
            try:
                ds = load_dataset(dataset_id, split=split)
            except Exception:
                config_names = get_dataset_config_names(dataset_id)
                if not config_names:
                    raise
                ds = concatenate_datasets(
                    [load_dataset(dataset_id, name=name, split=split) for name in config_names]
                )
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(num_samples, len(ds))))
            return ds, dataset_id
        except Exception as exc:
            errors[dataset_id] = str(exc)

    error_lines = "\n".join([f"  - {ds}: {err}" for ds, err in errors.items()])
    raise RuntimeError(
        "Failed to load MATH dataset from known sources.\n"
        f"Tried:\n{error_lines}"
    )


def compute_generations(
    model: str,
    prompts: list[str],
    max_tokens: int = 16384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    gpu_memory_utilization: float = 0.95,
    tensor_parallel_size: int = 1,
) -> tuple[list[int], list[str]]:
    """Run vLLM inference and return ``(token_counts, generated_texts)``.

    The LLM is created, used, and then explicitly destroyed so that callers
    (e.g. ``sweep_trajectory_lengths.py``) can call this in a loop across
    multiple checkpoints without leaking GPU memory.
    """
    import gc

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        n=1,
    )

    outputs = llm.generate(prompts, sampling_params)
    counts = [len(out.outputs[0].token_ids) for out in outputs]
    texts = [out.outputs[0].text for out in outputs]

    # Explicit cleanup so GPU memory is released before the next checkpoint.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return counts, texts


def compute_token_counts(
    model: str,
    prompts: list[str],
    max_tokens: int = 16384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    gpu_memory_utilization: float = 0.95,
    tensor_parallel_size: int = 1,
) -> list[int]:
    """Run vLLM inference and return per-prompt output token counts."""
    counts, _ = compute_generations(
        model=model,
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    return counts


def print_stats(model: str, split: str, num_samples: int, max_tokens: int, token_counts: list[int]) -> None:
    n = len(token_counts)
    if n == 0:
        print("No outputs collected.")
        return

    sorted_counts = sorted(token_counts)
    mean = statistics.mean(token_counts)
    std = statistics.stdev(token_counts) if n > 1 else 0.0
    minimum = sorted_counts[0]
    maximum = sorted_counts[-1]
    median = statistics.median(token_counts)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_counts[lo] + (sorted_counts[hi] - sorted_counts[lo]) * (idx - lo)

    p25 = percentile(25)
    p75 = percentile(75)
    p95 = percentile(95)

    print(f"\nModel:      {model}")
    print(f"Split:      {split}")
    print(f"Samples:    {num_samples} requested, {n} completed")
    print(f"Max tokens: {max_tokens}")
    print()
    print("Trajectory length (output tokens only):")
    print(f"  Mean:   {mean:.1f}")
    print(f"  Std:    {std:.1f}")
    print(f"  Min:    {minimum}")
    print(f"  p25:    {p25:.0f}")
    print(f"  Median: {median:.0f}")
    print(f"  p75:    {p75:.0f}")
    print(f"  p95:    {p95:.0f}")
    print(f"  Max:    {maximum}")


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(
        description="Measure average trajectory length (output tokens) on MATH problems."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of MATH problems to run")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory fraction (default: 0.95)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible dataset slicing")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(f"Loading dataset (split={args.split}, num_samples={args.num_samples}, seed={args.seed})...")
    dataset, source_id = load_math_dataset(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} examples from {source_id}")

    prompts = [format_math_prompt(row["problem"]) for row in dataset]

    print(f"Initialising vLLM with model={args.model}, tp={args.tensor_parallel_size}...")
    token_counts = compute_token_counts(
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print_stats(
        model=args.model,
        split=args.split,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        token_counts=token_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
