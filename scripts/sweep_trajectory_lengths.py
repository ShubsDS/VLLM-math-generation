#!/usr/bin/env python3
"""Sweep trajectory-length distributions across all checkpoints in a directory.

For each checkpoint found under ``--checkpoint-dir`` the script runs vLLM
inference via ``measure_avg_trajectory_length`` and produces:

* **Per-checkpoint histogram** saved to ``<image-dir>/dist_checkpoint_{N}.png``
* **Mean trajectory plot** saved to ``<image-dir>/mean_trajectory_length.png``
* **Summary table** printed to stdout at the end

Checkpoint discovery
--------------------
Sub-directories whose name ends with ``_{integer}`` are treated as checkpoints::

    /bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b/
        qwen2_5_math_1_5b_1/huggingface/    ← checkpoint 1
        qwen2_5_math_1_5b_2/huggingface/    ← checkpoint 2
        ...

The ``huggingface/`` sub-directory is used as the model path when present;
otherwise the checkpoint directory itself is passed to vLLM.  Checkpoints are
processed in ascending order of their trailing integer.

Typical usage
-------------
python scripts/sweep_trajectory_lengths.py \\
    --checkpoint-dir /bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b \\
    --num-samples 200 \\
    --max-tokens 4096 \\
    --image-dir images/sweep_qwen_1_5b
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(checkpoint_num, model_path)`` pairs sorted by checkpoint number.

    ``model_path`` points to the ``huggingface/`` sub-directory when present,
    otherwise to the checkpoint directory itself.
    """
    pattern = re.compile(r"^.+_(\d+)$")
    checkpoints: list[tuple[int, Path]] = []

    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m is None:
            continue
        ckpt_num = int(m.group(1))
        hf_sub = entry / "huggingface"
        model_path = hf_sub if hf_sub.is_dir() else entry
        checkpoints.append((ckpt_num, model_path))

    checkpoints.sort(key=lambda t: t[0])
    return checkpoints


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distribution(
    token_counts: list[int],
    ckpt_num: int,
    max_tokens: int,
    image_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    mean = statistics.mean(token_counts)
    median = statistics.median(token_counts)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_counts, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(mean, color="tomato", linestyle="--", linewidth=1.5, label=f"mean={mean:.0f}")
    ax.axvline(median, color="gold", linestyle=":", linewidth=1.5, label=f"median={median:.0f}")
    ax.set_xlabel("Output tokens (trajectory length)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trajectory length distribution — checkpoint {ckpt_num}")
    ax.set_xlim(left=0, right=max_tokens)
    ax.legend()
    fig.tight_layout()

    out_path = image_dir / f"dist_checkpoint_{ckpt_num}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved distribution plot: {out_path}")


def plot_mean_trajectory(
    ckpt_nums: list[int],
    means: list[float],
    image_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ckpt_nums, means, marker="o", color="steelblue", linewidth=1.8, markersize=6)
    for x, y in zip(ckpt_nums, means):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean output tokens")
    ax.set_title("Mean trajectory length across checkpoints")
    ax.set_xticks(ckpt_nums)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = image_dir / "mean_trajectory_length.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved mean trajectory plot: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(ckpt_nums: list[int], counts_by_ckpt: dict[int, list[int]]) -> None:
    header = f"{'Ckpt':>6}  {'N':>5}  {'Mean':>8}  {'Std':>8}  {'Min':>6}  {'p50':>6}  {'p95':>6}  {'Max':>6}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("Mean trajectory length per checkpoint")
    print(sep)
    print(header)
    print("-" * len(header))

    for ckpt_num in ckpt_nums:
        counts = counts_by_ckpt[ckpt_num]
        n = len(counts)
        if n == 0:
            print(f"{ckpt_num:>6}  {'0':>5}  {'FAILED':>8}")
            continue
        sorted_c = sorted(counts)
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if n > 1 else 0.0

        def pct(p: float) -> float:
            idx = (p / 100) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            return sorted_c[lo] + (sorted_c[hi] - sorted_c[lo]) * (idx - lo)

        print(
            f"{ckpt_num:>6}  {n:>5}  {mean:>8.1f}  {std:>8.1f}  "
            f"{sorted_c[0]:>6}  {statistics.median(counts):>6.0f}  {pct(95):>6.0f}  {sorted_c[-1]:>6}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import torch

    # Import module-level functions from the sibling script.
    sys.path.insert(0, str(Path(__file__).parent))
    from measure_avg_trajectory_length import (
        compute_token_counts,
        format_math_prompt,
        load_math_dataset,
        print_stats,
    )

    parser = argparse.ArgumentParser(
        description="Sweep trajectory-length distributions across all checkpoints in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing checkpoint sub-folders",
    )
    parser.add_argument("--num-samples", type=int, default=500, help="MATH problems per checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="vLLM GPU memory fraction")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs for tensor parallelism")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible dataset slicing")
    parser.add_argument("--image-dir", default="images", help="Directory where plots are saved")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        print(f"ERROR: --checkpoint-dir does not exist: {checkpoint_dir}", file=sys.stderr)
        return 1

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}", file=sys.stderr)
        return 1

    ckpt_nums = [c[0] for c in checkpoints]
    print(f"Found {len(checkpoints)} checkpoints: {ckpt_nums}")

    # Load dataset once — all checkpoints use the same shuffled sample.
    print(f"\nLoading dataset (split={args.split}, num_samples={args.num_samples}, seed={args.seed})...")
    dataset, source_id = load_math_dataset(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} examples from {source_id}")
    prompts = [format_math_prompt(row["problem"]) for row in dataset]

    counts_by_ckpt: dict[int, list[int]] = {}

    for ckpt_num, model_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint {ckpt_num}: {model_path}")
        print(f"{'='*60}")
        try:
            counts = compute_token_counts(
                model=str(model_path),
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping checkpoint {ckpt_num}.")
            counts = []

        counts_by_ckpt[ckpt_num] = counts

        if counts:
            print_stats(str(model_path), args.split, args.num_samples, args.max_tokens, counts)
            plot_distribution(counts, ckpt_num, args.max_tokens, image_dir)

    # Mean trajectory plot across all successful checkpoints.
    valid = [(n, counts_by_ckpt[n]) for n in ckpt_nums if counts_by_ckpt[n]]
    if valid:
        plot_mean_trajectory([n for n, _ in valid], [statistics.mean(c) for _, c in valid], image_dir)

    print_summary(ckpt_nums, counts_by_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
