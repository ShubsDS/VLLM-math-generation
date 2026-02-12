"""
Report generation for MATH dataset evaluation.

Generates evaluation reports in multiple formats:
- JSON: Machine-readable format with complete metrics
- CSV: Spreadsheet-friendly format for analysis
- Markdown: Human-readable summary report
- JSONL: Detailed per-problem results
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def __init__(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_dir: str,
        model_name: str = "unknown"
    ):
        """
        Initialize report generator.

        Args:
            results: List of per-problem evaluation results
            metrics: Computed metrics dictionary
            output_dir: Directory to save reports
            model_name: Model name for report headers
        """
        self.results = results
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.timestamp = datetime.now()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "detailed").mkdir(exist_ok=True)

        # Generate timestamp string for filenames
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")

    def generate_json_report(self) -> str:
        """
        Generate JSON report with all metrics.

        Returns:
            Path to generated JSON file
        """
        output_file = self.output_dir / "reports" / f"evaluation_{self.timestamp_str}.json"

        report = {
            "metadata": {
                "model_name": self.model_name,
                "timestamp": self.timestamp.isoformat(),
                "total_problems": len(self.results),
                "evaluation_version": "1.0"
            },
            "metrics": self.metrics
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"JSON report saved to: {output_file}")
        return str(output_file)

    def generate_csv_report(self) -> str:
        """
        Generate CSV report for spreadsheet analysis.

        Creates a flat CSV with one row per metric category.

        Returns:
            Path to generated CSV file
        """
        output_file = self.output_dir / "reports" / f"evaluation_{self.timestamp_str}.csv"

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['category', 'subcategory', 'accuracy', 'correct', 'total'])

            # Overall metrics
            overall = self.metrics.get('overall', {})
            writer.writerow([
                'overall',
                'all',
                f"{overall.get('accuracy', 0):.4f}",
                overall.get('correct', 0),
                overall.get('total', 0)
            ])

            # By-level metrics
            by_level = self.metrics.get('by_level', {})
            for level, level_metrics in sorted(by_level.items()):
                writer.writerow([
                    'level',
                    level,
                    f"{level_metrics.get('accuracy', 0):.4f}",
                    level_metrics.get('correct', 0),
                    level_metrics.get('total', 0)
                ])

            # By-type metrics
            by_type = self.metrics.get('by_type', {})
            for prob_type, type_metrics in sorted(by_type.items()):
                writer.writerow([
                    'type',
                    prob_type,
                    f"{type_metrics.get('accuracy', 0):.4f}",
                    type_metrics.get('correct', 0),
                    type_metrics.get('total', 0)
                ])

        print(f"CSV report saved to: {output_file}")
        return str(output_file)

    def generate_markdown_report(self) -> str:
        """
        Generate human-readable markdown report.

        Returns:
            Path to generated markdown file
        """
        output_file = self.output_dir / "reports" / f"evaluation_{self.timestamp_str}.md"

        overall = self.metrics.get('overall', {})
        by_level = self.metrics.get('by_level', {})
        by_type = self.metrics.get('by_type', {})
        detailed_stats = self.metrics.get('detailed_stats', {})

        lines = []

        # Header
        lines.append("# MATH Dataset Evaluation Report\n")
        lines.append(f"**Model:** {self.model_name}\n")
        lines.append(f"**Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Total Problems:** {overall.get('total', 0)}\n")
        lines.append("\n---\n")

        # Overall Performance
        lines.append("## Overall Performance\n")
        lines.append(f"- **Accuracy:** {overall.get('accuracy', 0):.2%}\n")
        lines.append(f"- **Correct:** {overall.get('correct', 0)} / {overall.get('total', 0)}\n")
        lines.append(f"- **Error Rate:** {detailed_stats.get('error_rate', 0):.2%}\n")
        if overall.get('avg_tokens'):
            lines.append(f"- **Avg Tokens/Solution:** {overall.get('avg_tokens', 0):.1f}\n")
        lines.append("\n")

        # Performance by Difficulty Level
        if by_level:
            lines.append("## Performance by Difficulty Level\n")
            lines.append("| Level | Accuracy | Correct | Total | % of Dataset |\n")
            lines.append("|-------|----------|---------|-------|-------------|\n")

            for level in sorted(by_level.keys(), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0):
                level_metrics = by_level[level]
                lines.append(
                    f"| {level} | "
                    f"{level_metrics.get('accuracy', 0):.2%} | "
                    f"{level_metrics.get('correct', 0)} | "
                    f"{level_metrics.get('total', 0)} | "
                    f"{level_metrics.get('percentage_of_dataset', 0):.1%} |\n"
                )
            lines.append("\n")

        # Performance by Problem Type
        if by_type:
            lines.append("## Performance by Problem Type\n")
            lines.append("| Type | Accuracy | Correct | Total | % of Dataset |\n")
            lines.append("|------|----------|---------|-------|-------------|\n")

            # Sort by accuracy (best to worst)
            sorted_types = sorted(
                by_type.items(),
                key=lambda x: x[1].get('accuracy', 0),
                reverse=True
            )

            for prob_type, type_metrics in sorted_types:
                lines.append(
                    f"| {prob_type} | "
                    f"{type_metrics.get('accuracy', 0):.2%} | "
                    f"{type_metrics.get('correct', 0)} | "
                    f"{type_metrics.get('total', 0)} | "
                    f"{type_metrics.get('percentage_of_dataset', 0):.1%} |\n"
                )
            lines.append("\n")

        # Key Insights
        if detailed_stats:
            lines.append("## Key Insights\n")

            if 'easiest_level' in detailed_stats:
                easiest = detailed_stats['easiest_level']
                lines.append(f"- **Easiest Level:** {easiest['level']} ({easiest['accuracy']:.2%})\n")

            if 'hardest_level' in detailed_stats:
                hardest = detailed_stats['hardest_level']
                lines.append(f"- **Hardest Level:** {hardest['level']} ({hardest['accuracy']:.2%})\n")

            if 'easiest_type' in detailed_stats:
                easiest = detailed_stats['easiest_type']
                lines.append(f"- **Easiest Type:** {easiest['type']} ({easiest['accuracy']:.2%})\n")

            if 'hardest_type' in detailed_stats:
                hardest = detailed_stats['hardest_type']
                lines.append(f"- **Hardest Type:** {hardest['type']} ({hardest['accuracy']:.2%})\n")

            lines.append("\n")

        # Footer
        lines.append("---\n")
        lines.append(f"*Generated by MATH Evaluation Pipeline v1.0*\n")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"Markdown report saved to: {output_file}")
        return str(output_file)

    def save_detailed_results(self) -> str:
        """
        Save per-problem detailed results as JSONL.

        Useful for error analysis and debugging.

        Returns:
            Path to generated JSONL file
        """
        output_file = self.output_dir / "detailed" / f"detailed_results_{self.timestamp_str}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Detailed results saved to: {output_file}")
        return str(output_file)

    def generate_all_reports(self) -> Dict[str, str]:
        """
        Generate all report formats.

        Returns:
            Dictionary mapping format names to file paths
        """
        print("\nGenerating evaluation reports...")

        report_paths = {
            "json": self.generate_json_report(),
            "csv": self.generate_csv_report(),
            "markdown": self.generate_markdown_report(),
            "detailed": self.save_detailed_results()
        }

        print("\nAll reports generated successfully!")
        return report_paths

    def print_summary(self):
        """Print a brief summary to console."""
        overall = self.metrics.get('overall', {})

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Total Problems: {overall.get('total', 0)}")
        print(f"Correct: {overall.get('correct', 0)}")
        print(f"Accuracy: {overall.get('accuracy', 0):.2%}")
        print("=" * 80 + "\n")
