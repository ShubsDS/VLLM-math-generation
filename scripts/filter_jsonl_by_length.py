#!/usr/bin/env python3
"""Filter a correct-teacher JSONL by total sequence length (prompt + output tokens)."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows from a correct-teacher JSONL whose total token count "
            "(prompt_tokens + output_tokens) exceeds --max-tokens."
        )
    )
    parser.add_argument("--input", required=True, help="Input correct-teacher JSONL file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Maximum allowed total tokens (prompt + output). Rows exceeding this are dropped.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path. Defaults to <input_stem>.filtered<N>k<suffix> "
            "where N is --max-tokens // 1000."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        k = args.max_tokens // 1000
        # Preserve .correct.jsonl or .jsonl suffix
        name = input_path.name
        if name.endswith(".correct.jsonl"):
            stem = name[: -len(".correct.jsonl")]
            output_path = input_path.parent / f"{stem}.filtered{k}k.correct.jsonl"
        else:
            stem = input_path.stem
            output_path = input_path.parent / f"{stem}.filtered{k}k.jsonl"

    kept, dropped = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_num}: {exc}") from exc

            total = row.get("prompt_tokens", 0) + row.get("output_tokens", 0)
            if total > args.max_tokens:
                dropped += 1
            else:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    total_in = kept + dropped
    print(f"Input rows:  {total_in}")
    print(f"Kept:        {kept}  ({100 * kept / total_in:.1f}%)")
    print(f"Dropped:     {dropped}  ({100 * dropped / total_in:.1f}%)")
    print(f"Output:      {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
