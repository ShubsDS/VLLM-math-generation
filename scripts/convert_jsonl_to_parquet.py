#!/usr/bin/env python3
"""Convert correct-teacher JSONL into train/val parquet files for SFT."""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num} in {path}: {exc}") from exc
    return rows


def validate_schema(rows: list[dict], prompt_key: str, response_key: str) -> None:
    if not rows:
        raise ValueError("Input JSONL contains no rows.")

    sample = rows[0]
    missing = [key for key in (prompt_key, response_key) if key not in sample]
    if missing:
        raise ValueError(
            f"Input rows are missing required keys {missing}. "
            f"Available keys in first row: {list(sample.keys())}"
        )


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float,
    shuffle: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    val_size = int(len(df) * val_ratio)
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert correct-teacher JSONL to train.parquet and val.parquet"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input correct-teacher JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sft",
        help="Output directory for train.parquet and val.parquet",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable row shuffling before split",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Prompt field key in JSONL",
    )
    parser.add_argument(
        "--response-key",
        default="generated_solution",
        help="Response field key in JSONL",
    )

    args = parser.parse_args()

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(input_path)
    validate_schema(rows, prompt_key=args.prompt_key, response_key=args.response_key)

    df = pd.DataFrame(rows)
    train_df, val_df = split_dataframe(
        df=df,
        val_ratio=args.val_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Train rows: {len(train_df)} -> {train_path}")
    print(f"Val rows:   {len(val_df)} -> {val_path}")
    print(f"Row conservation check: {len(train_df) + len(val_df)} == {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
