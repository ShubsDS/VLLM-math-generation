#!/usr/bin/env python3
"""
Prepare data for SFT training.

This script:
1. Converts JSONL files to Parquet format
2. Optionally splits data into train/val sets
3. Validates data format
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def prepare_sft_data(
    input_jsonl: str,
    output_dir: str,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    prompt_key: str = "prompt",
    response_key: str = "generated_solution",
):
    """
    Prepare SFT training data from JSONL files.

    Args:
        input_jsonl: Path to input JSONL file(s) (can use glob pattern)
        output_dir: Directory to save output parquet files
        val_split: Fraction of data to use for validation (0-1)
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for shuffling
        prompt_key: Key in JSONL for prompts
        response_key: Key in JSONL for responses
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all matching JSONL files
    input_path = Path(input_jsonl)
    if "*" in input_jsonl:
        # Glob pattern
        jsonl_files = list(input_path.parent.glob(input_path.name))
    else:
        jsonl_files = [input_path]

    print(f"Found {len(jsonl_files)} JSONL file(s)")

    # Load and combine all data
    all_data = []
    for jsonl_file in jsonl_files:
        print(f"Loading {jsonl_file}...")
        data = load_jsonl(jsonl_file)
        all_data.extend(data)
        print(f"  Loaded {len(data)} examples")

    print(f"\nTotal examples: {len(all_data)}")

    # Verify keys exist
    if all_data:
        first_item = all_data[0]
        if prompt_key not in first_item:
            raise ValueError(f"Key '{prompt_key}' not found. Available: {list(first_item.keys())}")
        if response_key not in first_item:
            raise ValueError(f"Key '{response_key}' not found. Available: {list(first_item.keys())}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Shuffle if requested
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split into train/val
    if val_split > 0:
        val_size = int(len(df) * val_split)
        train_df = df.iloc[val_size:]
        val_df = df.iloc[:val_size]

        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        print(f"\nSaved train set: {len(train_df)} examples -> {train_path}")
        print(f"Saved val set: {len(val_df)} examples -> {val_path}")
    else:
        # Save all as train
        train_path = output_dir / "train.parquet"
        df.to_parquet(train_path, index=False)
        print(f"\nSaved train set: {len(df)} examples -> {train_path}")

    # Print sample
    print("\n" + "=" * 50)
    print("Sample data:")
    print("=" * 50)
    print(f"Prompt: {df[prompt_key].iloc[0][:200]}...")
    print(f"\nResponse: {df[response_key].iloc[0][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("input_jsonl", type=str, help="Input JSONL file or glob pattern")
    parser.add_argument("-o", "--output-dir", type=str, default="./data/sft", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Key for prompts")
    parser.add_argument("--response-key", type=str, default="generated_solution", help="Key for responses")

    args = parser.parse_args()

    prepare_sft_data(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        val_split=args.val_split,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
    )


if __name__ == "__main__":
    main()
