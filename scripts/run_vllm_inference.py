"""
Run distributed VLLM inference on MATH dataset using multiple GPUs.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch
import transformers
import numpy as np


DEPRECATED_MODEL_ALIASES = {
    "Qwen/Qwen2.5-Math-14B-Instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
}


def resolve_model_name(model_name: str) -> str:
    """Map deprecated/invalid model IDs to valid HuggingFace IDs."""
    if model_name in DEPRECATED_MODEL_ALIASES:
        replacement = DEPRECATED_MODEL_ALIASES[model_name]
        print(
            f"Model '{model_name}' is not available on Hugging Face. "
            f"Using '{replacement}' instead."
        )
        return replacement
    return model_name


def validate_transformers_compatibility() -> None:
    """Fail fast with a clear message when transformers major version is unsupported."""
    version = transformers.__version__
    major_str = version.split(".", 1)[0]
    try:
        major = int(major_str)
    except ValueError:
        # Keep running for non-standard version strings (e.g. nightly tags).
        return

    if major >= 5:
        raise RuntimeError(
            f"Incompatible transformers version detected: {version}. "
            "This project currently requires transformers<5 for vLLM tokenizer compatibility. "
            "Run: uv pip install 'transformers>=4.55.2,<5' 'tokenizers>=0.21.1,<0.23'"
        )


def validate_numpy_compatibility() -> None:
    """Fail fast when NumPy is incompatible with numba used by vLLM."""
    version = np.__version__
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return

    if major > 2 or (major == 2 and minor > 2):
        raise RuntimeError(
            f"Incompatible numpy version detected: {version}. "
            "vLLM's numba dependency requires numpy<=2.2. "
            "Run: uv pip install 'numpy>=1.24.0,<2.3'"
        )


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
    return prompts


def save_outputs(outputs: List[Dict], output_file: str):
    """Save model outputs to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')


def run_inference(
    model_name: str,
    prompts_file: str,
    output_file: str,
    tensor_parallel_size: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 32,
):
    """
    Run VLLM inference on prompts.

    Args:
        model_name: HuggingFace model name or path
        prompts_file: Path to JSONL file with prompts
        output_file: Path to save outputs
        tensor_parallel_size: Number of GPUs to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        top_p: Nucleus sampling parameter
        batch_size: Number of prompts to process at once
    """
    model_name = resolve_model_name(model_name)
    validate_transformers_compatibility()
    validate_numpy_compatibility()

    print(f"Initializing VLLM with {tensor_parallel_size} GPUs...")
    print(f"Model: {model_name}")

    # Initialize VLLM with tensor parallelism
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=16384,  # Adjust based on model capacity
        gpu_memory_utilization=0.95,  # High utilization for large memory GPUs (H100 100GB)
        dtype="bfloat16",  # or "float16" depending on GPU support
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )

    print(f"Loading prompts from {prompts_file}...")
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    print(f"Running inference with batch_size={batch_size}...")

    # Run inference in batches
    all_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]

        # Generate outputs
        outputs = llm.generate(batch_prompts, sampling_params)

        # Process outputs
        for idx, output in enumerate(outputs):
            prompt_idx = i + idx
            generated_text = output.outputs[0].text

            all_outputs.append({
                'id': prompt_idx,
                'prompt': batch_prompts[idx],
                'generated_solution': generated_text,
                'num_tokens': len(output.outputs[0].token_ids),
            })

    print(f"\nSaving {len(all_outputs)} outputs to {output_file}...")
    save_outputs(all_outputs, output_file)

    print("Inference complete!")
    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM inference on MATH dataset")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to JSONL file containing prompts"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/math_solutions.jsonl",
        help="Path to save generated solutions"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per solution"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p parameter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    validate_transformers_compatibility()
    validate_numpy_compatibility()

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPUs.")

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < args.tensor_parallel_size:
        print(f"Warning: Requested {args.tensor_parallel_size} GPUs but only {num_gpus} available.")
        print(f"Adjusting tensor_parallel_size to {num_gpus}")
        args.tensor_parallel_size = num_gpus

    run_inference(
        model_name=args.model_name,
        prompts_file=args.prompts_file,
        output_file=args.output_file,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
