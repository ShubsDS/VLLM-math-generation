"""
Load and prepare the MATH dataset for VLLM inference.
"""
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import argparse


def format_math_prompt(problem: str) -> str:
    """Format a MATH problem into a prompt for the model."""
    return f"""Solve the following math problem step by step. Show your reasoning and provide the final answer.

Problem: {problem}

Solution:"""


def prepare_math_dataset(
    output_dir: str = "data",
    split: str = "test",
    subset: str = "all",
    max_samples: int = None
):
    """
    Load MATH dataset and prepare it for inference.

    Args:
        output_dir: Directory to save the prepared dataset
        split: Dataset split to use ('train' or 'test')
        subset: Subject subset (e.g., 'algebra', 'geometry', 'all')
        max_samples: Maximum number of samples to process (None for all)
    """
    print(f"Loading MATH dataset (split={split}, subset={subset})...")

    # Load the MATH dataset from HuggingFace
    if subset == "all":
        dataset = load_dataset("hendrycks/competition_math", split=split)
    else:
        dataset = load_dataset("hendrycks/competition_math", split=split)
        # Filter by subject type
        dataset = dataset.filter(lambda x: x['type'] == subset)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} problems")

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process and save the dataset
    prompts = []
    metadata = []

    for idx, example in enumerate(tqdm(dataset, desc="Processing problems")):
        prompt = format_math_prompt(example['problem'])

        prompts.append(prompt)
        metadata.append({
            'id': idx,
            'problem': example['problem'],
            'solution': example['solution'],
            'level': example['level'],
            'type': example['type']
        })

    # Save prompts for inference
    prompts_file = output_path / f"math_{split}_prompts.jsonl"
    with open(prompts_file, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + '\n')

    # Save metadata with ground truth
    metadata_file = output_path / f"math_{split}_metadata.jsonl"
    with open(metadata_file, 'w') as f:
        for meta in metadata:
            f.write(json.dumps(meta) + '\n')

    print(f"\nSaved {len(prompts)} prompts to {prompts_file}")
    print(f"Saved metadata to {metadata_file}")

    return prompts_file, metadata_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MATH dataset for inference")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subject subset (e.g., 'algebra', 'geometry', or 'all')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    args = parser.parse_args()

    prepare_math_dataset(
        output_dir=args.output_dir,
        split=args.split,
        subset=args.subset,
        max_samples=args.max_samples
    )
