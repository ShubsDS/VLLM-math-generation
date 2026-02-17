#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format for verl SFT training.

This script converts the generated math solutions JSONL files into the parquet
format expected by verl's SFTDataset.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_jsonl_to_parquet(
    jsonl_path: str,
    output_path: str | None = None,
    prompt_key: str = "prompt",
    response_key: str = "generated_solution",
):
    """
    Convert a JSONL file to Parquet format for SFT training.

    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output parquet file (if None, uses same name with .parquet)
        prompt_key: Key in JSONL for the prompt/input text
        response_key: Key in JSONL for the response/solution text
    """
    jsonl_path = Path(jsonl_path)

    if output_path is None:
        output_path = jsonl_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)

    # Read JSONL file
    data = []
    print(f"Reading {jsonl_path}...")
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue

    print(f"Loaded {len(data)} examples")

    # Verify required keys exist
    if data:
        first_item = data[0]
        if prompt_key not in first_item:
            raise ValueError(f"Key '{prompt_key}' not found in JSONL. Available keys: {list(first_item.keys())}")
        if response_key not in first_item:
            raise ValueError(f"Key '{response_key}' not found in JSONL. Available keys: {list(first_item.keys())}")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save as parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} examples to {output_path}")

    # Print sample
    if len(df) > 0:
        print("\nSample entry:")
        print(f"Prompt: {df[prompt_key].iloc[0][:100]}...")
        print(f"Response: {df[response_key].iloc[0][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet for SFT training")
    parser.add_argument("jsonl_path", type=str, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", type=str, help="Path to output parquet file")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Key for prompt in JSONL")
    parser.add_argument("--response-key", type=str, default="generated_solution", help="Key for response in JSONL")

    args = parser.parse_args()

    convert_jsonl_to_parquet(
        jsonl_path=args.jsonl_path,
        output_path=args.output,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
    )


if __name__ == "__main__":
    main()
