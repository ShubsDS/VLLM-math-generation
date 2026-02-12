"""
Load and prepare the MATH dataset for VLLM inference.
"""
import json
from pathlib import Path
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
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

    dataset, dataset_source = load_math_dataset(split=split, subset=subset)
    print(f"Loaded from dataset source: {dataset_source}")

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


def normalize_subset_name(value: str) -> str:
    """Normalize subset/type names for matching across dataset variants."""
    return value.strip().lower().replace("&", "and").replace(" ", "_")


def filter_by_type(dataset, subset: str):
    """Filter by `type` column with flexible matching (e.g. 'counting_and_probability')."""
    if "type" not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'type' column for subset filtering.")

    normalized_subset = normalize_subset_name(subset)
    filtered = dataset.filter(
        lambda x: normalize_subset_name(x["type"]) == normalized_subset
    )
    if len(filtered) == 0:
        available_types = sorted({normalize_subset_name(t) for t in dataset["type"]})
        raise ValueError(
            f"No rows matched subset '{subset}'. Available subsets include: {available_types}"
        )
    return filtered


def load_math_dataset(split: str, subset: str):
    """
    Load MATH dataset from known HF IDs.

    Handles both:
    - Single dataset layout with full split (e.g. Maxwell-Jia/MATH)
    - Multi-config layout by subject type (e.g. EleutherAI/hendrycks_math)
    """
    # Try legacy first for backwards compatibility, then known active mirrors.
    candidate_datasets = [
        "hendrycks/competition_math",
        "EleutherAI/hendrycks_math",
        "Maxwell-Jia/MATH",
        "jeggers/competition_math",
    ]

    errors = {}
    normalized_subset = normalize_subset_name(subset)

    for dataset_id in candidate_datasets:
        try:
            if normalized_subset == "all":
                try:
                    return load_dataset(dataset_id, split=split), dataset_id
                except Exception:
                    # Some dataset repos use per-subject configs; combine them for "all".
                    config_names = get_dataset_config_names(dataset_id)
                    if not config_names:
                        raise
                    combined = concatenate_datasets(
                        [load_dataset(dataset_id, name=name, split=split) for name in config_names]
                    )
                    return combined, dataset_id

            # Try subset as config name first (works for per-subject repos)
            try:
                return load_dataset(dataset_id, name=normalized_subset, split=split), dataset_id
            except Exception:
                pass

            # Fallback: load full split and filter by `type` column
            full_split = load_dataset(dataset_id, split=split)
            return filter_by_type(full_split, subset), dataset_id

        except Exception as exc:
            errors[dataset_id] = str(exc)

    error_lines = "\n".join([f"- {ds}: {err}" for ds, err in errors.items()])
    raise RuntimeError(
        "Failed to load MATH dataset from all known sources.\n"
        f"Tried:\n{error_lines}"
    )


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
