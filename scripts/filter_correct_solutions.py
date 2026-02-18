#!/usr/bin/env python3
"""
Filter a solutions JSONL file to only correct answers.

Usage:
    python scripts/filter_correct_solutions.py \
        --input outputs/math_test_solutions.jsonl \
        --metadata data/math_test_metadata.jsonl \
        --output outputs/correct_solutions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import MATHEvaluator


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL solutions to only correct answers")
    parser.add_argument("--input", required=True, help="Input solutions JSONL file")
    parser.add_argument("--metadata", required=True, help="Ground truth metadata JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file for correct solutions")
    args = parser.parse_args()

    evaluator = MATHEvaluator(
        solutions_file=args.input,
        metadata_file=args.metadata,
    )
    results = evaluator.evaluate_all()

    correct_ids = {r["id"] for r in results if r["is_correct"]}
    total = len(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for problem_id, solution in evaluator.solutions.items():
            if problem_id in correct_ids:
                f.write(json.dumps(solution) + "\n")

    print(f"Filtered {len(correct_ids)}/{total} correct solutions ({len(correct_ids)/total:.1%}) -> {output_path}")


if __name__ == "__main__":
    main()
