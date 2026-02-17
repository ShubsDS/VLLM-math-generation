"""
Run end-to-end local MATH inference without Slurm.

This script:
1. Downloads/prepares the MATH dataset prompts/metadata.
2. Runs vLLM inference on the prepared prompts.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch

from load_math_dataset import prepare_math_dataset
from run_vllm_inference import (
    run_inference,
    validate_numpy_compatibility,
    validate_transformers_compatibility,
)


def build_default_output_file(output_dir: str, split: str) -> str:
    """Create a timestamped output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(output_dir) / f"math_{split}_solutions_{timestamp}.jsonl")


def run_local_pipeline(
    data_dir: str,
    output_dir: str,
    output_file: str | None,
    split: str,
    subset: str,
    max_samples: int | None,
    model_name: str,
    tensor_parallel_size: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> str:
    """Prepare dataset then run inference in one local process."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    print("=== Local Pipeline Start ===")
    print(f"VLLM_WORKER_MULTIPROC_METHOD={os.environ['VLLM_WORKER_MULTIPROC_METHOD']}")
    print(f"PYTORCH_NVML_BASED_CUDA_CHECK={os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']}")

    print("\n[1/2] Preparing dataset...")
    prompts_file, metadata_file = prepare_math_dataset(
        output_dir=data_dir,
        split=split,
        subset=subset,
        max_samples=max_samples,
    )
    print(f"Prompts file: {prompts_file}")
    print(f"Metadata file: {metadata_file}")

    print("\n[2/2] Running inference...")
    validate_transformers_compatibility()
    validate_numpy_compatibility()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline requires at least one GPU."
        )

    num_gpus = torch.cuda.device_count()
    if tensor_parallel_size is None:
        tensor_parallel_size = num_gpus
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1.")
    if tensor_parallel_size > num_gpus:
        print(
            f"Requested {tensor_parallel_size} GPUs, but only {num_gpus} are available. "
            f"Using {num_gpus}."
        )
        tensor_parallel_size = num_gpus

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if output_file is None:
        output_file = build_default_output_file(output_dir=output_dir, split=split)

    run_inference(
        model_name=model_name,
        prompts_file=str(prompts_file),
        output_file=output_file,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    print("\n=== Local Pipeline Complete ===")
    print(f"Saved generated solutions to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dataset preparation + vLLM inference locally (no Slurm)."
    )

    parser.add_argument("--data-dir", type=str, default="data", help="Directory for prompts/metadata")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for solutions")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional explicit output JSONL path (overrides --output-dir naming)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subject subset (e.g. algebra, geometry, all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for dataset size",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="HuggingFace model name or local model path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: all visible GPUs)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )

    args = parser.parse_args()

    run_local_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_file=args.output_file,
        split=args.split,
        subset=args.subset,
        max_samples=args.max_samples,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
