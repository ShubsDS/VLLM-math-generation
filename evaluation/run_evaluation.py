#!/usr/bin/env python3
"""
Main script to run MATH dataset evaluation.

This script evaluates generated solutions against ground truth from the MATH dataset
and produces comprehensive accuracy reports.

Usage:
    python evaluation/run_evaluation.py \\
        --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \\
        --metadata-file data/math_test_metadata.jsonl \\
        --output-dir eval_results \\
        --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \\
        --report-format all

Example with verbose output:
    python evaluation/run_evaluation.py \\
        --solutions-file outputs/math_test_solutions_*.jsonl \\
        --metadata-file data/math_test_metadata.jsonl \\
        --verbose
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import MATHEvaluator
from evaluation.generate_report import ReportGenerator

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def save_correct_solutions(evaluator: MATHEvaluator, results: list, output_path: str) -> int:
    """
    Save only correct solutions to a parquet file for SFT training.

    Args:
        evaluator: MATHEvaluator instance with loaded solutions
        results: List of evaluation results
        output_path: Path to save the parquet file

    Returns:
        Number of correct solutions saved
    """
    if not HAS_PANDAS:
        print("Error: pandas is required to save parquet files. Install with: pip install pandas")
        return 0

    # Filter correct results
    correct_results = [r for r in results if r.get("is_correct", False)]

    if not correct_results:
        print("Warning: No correct solutions found. Parquet file not created.")
        return 0

    # Extract original solution data for correct problems
    correct_solutions = []
    for result in correct_results:
        problem_id = result["id"]
        if problem_id in evaluator.solutions:
            solution_data = evaluator.solutions[problem_id].copy()
            # Add evaluation metadata
            solution_data["is_correct"] = True
            solution_data["confidence"] = result.get("confidence", "exact")
            solution_data["level"] = result.get("level", "Unknown")
            solution_data["type"] = result.get("type", "Unknown")
            correct_solutions.append(solution_data)

    # Convert to DataFrame
    df = pd.DataFrame(correct_solutions)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False)

    print(f"\n✓ Saved {len(correct_solutions)} correct solutions to: {output_path}")
    print(f"  Accuracy: {len(correct_solutions)}/{len(results)} = {len(correct_solutions)/len(results):.2%}")
    print(f"  Ready for SFT training with: data.train_files={output_path}")

    return len(correct_solutions)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MATH dataset solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluation/run_evaluation.py \\
      --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \\
      --metadata-file data/math_test_metadata.jsonl

  # With custom output directory and model name
  python evaluation/run_evaluation.py \\
      --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \\
      --metadata-file data/math_test_metadata.jsonl \\
      --output-dir eval_results/qwen25_7b \\
      --model-name "Qwen/Qwen2.5-Math-7B-Instruct"

  # Generate only JSON report with verbose output
  python evaluation/run_evaluation.py \\
      --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \\
      --metadata-file data/math_test_metadata.jsonl \\
      --report-format json \\
      --verbose

  # Save only correct solutions for SFT training
  python evaluation/run_evaluation.py \\
      --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \\
      --metadata-file data/math_test_metadata.jsonl \\
      --save-correct data/sft/correct_solutions.parquet
        """
    )

    parser.add_argument(
        "--solutions-file",
        type=str,
        required=True,
        help="Path to generated solutions JSONL file"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="Path to ground truth metadata JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for evaluation results (default: eval_results)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unknown",
        help="Model name for report labeling (default: unknown)"
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="Output format for reports (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-problem evaluation results"
    )
    parser.add_argument(
        "--no-detailed",
        action="store_true",
        help="Skip generating detailed per-problem results file"
    )
    parser.add_argument(
        "--save-correct",
        type=str,
        metavar="PATH",
        help="Save only correct solutions to parquet file for SFT training (e.g., data/sft/correct_solutions.parquet)"
    )

    return parser.parse_args()


def main():
    """Main evaluation workflow."""
    args = parse_args()

    print("=" * 80)
    print("MATH DATASET EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Solutions: {args.solutions_file}")
    print(f"Ground Truth: {args.metadata_file}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80 + "\n")

    try:
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = MATHEvaluator(
            solutions_file=args.solutions_file,
            metadata_file=args.metadata_file,
            model_name=args.model_name,
            verbose=args.verbose
        )

        # Run evaluation
        results = evaluator.evaluate_all()

        if not results:
            print("Error: No results generated. Check that solution and metadata files have matching IDs.")
            return 1

        # Compute metrics
        print("\nComputing metrics...")
        metrics = evaluator.compute_metrics()

        # Generate reports
        report_gen = ReportGenerator(
            results=results,
            metrics=metrics,
            output_dir=args.output_dir,
            model_name=args.model_name
        )

        # Print summary to console
        report_gen.print_summary()

        # Generate requested report formats
        if args.report_format == "all":
            report_paths = report_gen.generate_all_reports()
        else:
            print("\nGenerating evaluation reports...")
            report_paths = {}

            if args.report_format == "json":
                report_paths["json"] = report_gen.generate_json_report()
            elif args.report_format == "csv":
                report_paths["csv"] = report_gen.generate_csv_report()
            elif args.report_format == "markdown":
                report_paths["markdown"] = report_gen.generate_markdown_report()

            # Always generate detailed results unless explicitly disabled
            if not args.no_detailed:
                report_paths["detailed"] = report_gen.save_detailed_results()

        # Save correct solutions for SFT training if requested
        if args.save_correct:
            num_saved = save_correct_solutions(evaluator, results, args.save_correct)
            if num_saved > 0:
                if report_paths is None:
                    report_paths = {}
                report_paths["correct_solutions"] = args.save_correct

        # Final summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Model: {args.model_name}")
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
        print(f"Correct: {metrics['overall']['correct']} / {metrics['overall']['total']}")
        print(f"\nResults saved to: {args.output_dir}")

        # Show report file paths
        if report_paths:
            print("\nGenerated reports:")
            for report_type, path in report_paths.items():
                if report_type == "correct_solutions":
                    print(f"  - {report_type.replace('_', ' ').title()}: {Path(path).name} (for SFT training)")
                else:
                    print(f"  - {report_type.capitalize()}: {Path(path).name}")

        print("=" * 80 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check that the file paths are correct.")
        return 1
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nUnexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
