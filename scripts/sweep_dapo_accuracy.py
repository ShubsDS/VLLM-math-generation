#!/usr/bin/env python3
"""Evaluate MATH test accuracy across RL training steps for multiple DAPO runs.

Each DAPO run starts from a different SFT checkpoint (e.g. epoch 1, 3, 5, 10).
For each run, every saved RL checkpoint (global_step_N/actor/) is merged from
FSDP shards to a temporary HuggingFace model, evaluated on the MATH test set,
then discarded.  All runs are plotted on a single accuracy-vs-RL-step figure.

Usage
-----
    python scripts/sweep_dapo_accuracy.py --sft-epochs 1 3 5 10
    python scripts/sweep_dapo_accuracy.py --sft-epochs 1 3 5 10 \\
        --num-samples 200 --max-tokens 8192 --image-dir images/dapo_sweep

Checkpoint layout expected
--------------------------
    <base-ckpt-dir>/
        dapo_from_qwen2_5_math_1_5b_<N>/
            global_step_20/actor/
                model_world_size_2_rank_0.pt
                model_world_size_2_rank_1.pt
                huggingface/        ← tokenizer + config (no weights)
            global_step_40/actor/
            ...
"""

import argparse
import gc
import re
import sys
import tempfile
from pathlib import Path

# vLLM v1 spawns an EngineCore subprocess.  Force 'spawn' to avoid inherited
# CUDA state when a second LLM is created after the first is destroyed.
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

VERL_DIR = Path("/u/fvc9ch/nlp_research/verl")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_rl_steps(dapo_run_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(rl_step, actor_dir)`` pairs sorted by step."""
    pattern = re.compile(r"^global_step_(\d+)$")
    steps: list[tuple[int, Path]] = []
    for entry in dapo_run_dir.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m is None:
            continue
        actor_dir = entry / "actor"
        if actor_dir.is_dir():
            steps.append((int(m.group(1)), actor_dir))
    steps.sort(key=lambda t: t[0])
    return steps


# ---------------------------------------------------------------------------
# FSDP → HuggingFace conversion
# ---------------------------------------------------------------------------

def merge_fsdp_to_hf(actor_dir: Path, target_dir: Path) -> None:
    """Merge FSDP shards in *actor_dir* into a HuggingFace model at *target_dir*."""
    # Add verl to sys.path so legacy_model_merger can import verl utilities.
    verl_str = str(VERL_DIR)
    if verl_str not in sys.path:
        sys.path.insert(0, verl_str)

    from scripts.legacy_model_merger import FSDPModelMerger, ModelMergerConfig  # type: ignore

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_dir),
        hf_model_config_path=str(actor_dir),  # auto-detects actor_dir/huggingface/
        target_dir=str(target_dir),
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS  = ["steelblue", "seagreen", "tomato", "darkorange", "mediumpurple"]
MARKERS = ["o", "s", "^", "D", "v"]


def plot_accuracy_curves(
    results: dict[int, dict[int, float]],
    image_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, sft_epoch in enumerate(sorted(results)):
        step_acc = results[sft_epoch]
        steps = sorted(step_acc)
        accs  = [step_acc[s] * 100 for s in steps]
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(steps, accs, marker=marker, color=color, linewidth=1.8,
                markersize=6, label=f"SFT epoch {sft_epoch}")
        for x, y in zip(steps, accs):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    ax.set_xlabel("RL training step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MATH test accuracy across RL training steps (DAPO)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = image_dir / "dapo_accuracy_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved accuracy plot: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict[int, dict[int, float]]) -> None:
    all_steps = sorted({s for step_acc in results.values() for s in step_acc})
    col_w = 10
    header = f"{'SFT epoch':>10}" + "".join(f"{s:>{col_w}}" for s in all_steps)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("Accuracy (%) by SFT epoch and RL step")
    print(sep)
    print(header)
    print("-" * len(header))
    for sft_epoch in sorted(results):
        step_acc = results[sft_epoch]
        row = f"{sft_epoch:>10}" + "".join(
            f"{step_acc[s]*100:>{col_w}.1f}" if s in step_acc else f"{'n/a':>{col_w}}"
            for s in all_steps
        )
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from measure_avg_trajectory_length import (  # type: ignore
        format_math_prompt,
        load_math_dataset,
        compute_generations,
    )
    from run_local_pipeline import compare_answers  # type: ignore

    parser = argparse.ArgumentParser(
        description="Sweep MATH test accuracy across RL steps for multiple DAPO runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sft-epochs", type=int, nargs="+", default=[1, 3, 5, 10],
        help="SFT epoch numbers to evaluate (each maps to a dapo_from_qwen2_5_math_1_5b_N run)",
    )
    parser.add_argument(
        "--base-ckpt-dir", default="/bigtemp/fvc9ch/checkpoints",
        help="Root directory containing dapo_from_* subdirectories",
    )
    parser.add_argument("--num-samples", type=int, default=500,
                        help="MATH test problems per checkpoint")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="GPUs for tensor parallelism (default: all visible)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", default="images",
                        help="Directory where the plot is saved")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    num_gpus = torch.cuda.device_count()
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else num_gpus
    tp_size = max(1, min(tp_size, num_gpus))

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt_dir = Path(args.base_ckpt_dir)

    # Load dataset once — all checkpoints use the same shuffled sample.
    print(f"Loading MATH test set (num_samples={args.num_samples}, seed={args.seed})...")
    dataset, source_id = load_math_dataset(
        split="test", num_samples=args.num_samples, seed=args.seed,
    )
    print(f"Loaded {len(dataset)} examples from {source_id}")
    prompts   = [format_math_prompt(row["problem"]) for row in dataset]
    solutions = [row["solution"] for row in dataset]

    results: dict[int, dict[int, float]] = {}

    for sft_epoch in args.sft_epochs:
        dapo_run_dir = base_ckpt_dir / f"dapo_from_qwen2_5_math_1_5b_{sft_epoch}"
        if not dapo_run_dir.is_dir():
            print(f"\nWARNING: DAPO run dir not found for SFT epoch {sft_epoch}: {dapo_run_dir}")
            continue

        rl_steps = discover_rl_steps(dapo_run_dir)
        if not rl_steps:
            print(f"\nWARNING: No RL step checkpoints found in {dapo_run_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"SFT epoch {sft_epoch}: {len(rl_steps)} RL checkpoints")
        print(f"Steps: {[s for s, _ in rl_steps]}")
        print(f"{'='*60}")

        results[sft_epoch] = {}

        for rl_step, actor_dir in rl_steps:
            print(f"\n  -- RL step {rl_step}: merging FSDP shards from {actor_dir} --")

            with tempfile.TemporaryDirectory(
                prefix=f"dapo_hf_e{sft_epoch}_s{rl_step}_"
            ) as tmp_dir:
                try:
                    merge_fsdp_to_hf(actor_dir, Path(tmp_dir))
                except Exception as exc:
                    print(f"  ERROR merging shards: {exc} — skipping step {rl_step}.")
                    continue

                try:
                    _, texts = compute_generations(
                        model=tmp_dir,
                        prompts=prompts,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        tensor_parallel_size=tp_size,
                    )
                except Exception as exc:
                    print(f"  ERROR during inference: {exc} — skipping step {rl_step}.")
                    continue
            # tmp_dir is deleted here; GPU memory released by compute_generations.

            correct = sum(
                1 for text, sol in zip(texts, solutions)
                if compare_answers(text, sol)[0]
            )
            accuracy = correct / len(texts)
            results[sft_epoch][rl_step] = accuracy
            print(f"  RL step {rl_step}: {correct}/{len(texts)} = {accuracy*100:.1f}%")

            gc.collect()

    if not any(results.values()):
        print("\nNo results collected. Exiting.")
        return 1

    plot_accuracy_curves(results, image_dir)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
