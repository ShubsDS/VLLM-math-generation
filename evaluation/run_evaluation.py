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
