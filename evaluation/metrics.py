"""
Metrics computation for MATH dataset evaluation.

Provides comprehensive accuracy metrics including overall accuracy,
breakdown by difficulty level, and breakdown by problem type.
"""

from typing import List, Dict, Any
from collections import defaultdict


class MetricsCalculator:
    """Calculate various metrics from evaluation results."""

    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize metrics calculator with evaluation results.

        Args:
            results: List of evaluation results, each containing:
                - id: Problem ID
                - problem: Problem text
                - generated_answer: Extracted answer from model
                - ground_truth_answer: Extracted answer from ground truth
                - is_correct: Boolean correctness
                - level: Difficulty level (e.g., "Level 1")
                - type: Problem type (e.g., "Algebra")
                - num_tokens: Number of tokens generated (optional)
        """
        self.results = results

    def compute_overall_metrics(self) -> Dict[str, Any]:
        """
        Compute overall accuracy and statistics.

        Returns:
            Dictionary with:
            - accuracy: Overall accuracy (0-1)
            - correct: Number of correct answers
            - total: Total number of problems
            - avg_tokens: Average tokens per solution (if available)
        """
        if not self.results:
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "avg_tokens": 0.0
            }

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0.0

        # Calculate average tokens if available
        tokens = [r.get("num_tokens", 0) for r in self.results if r.get("num_tokens")]
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_tokens": avg_tokens
        }

    def compute_by_level_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute accuracy breakdown by difficulty level.

        Returns:
            Dictionary mapping level names to metrics:
            {
                "Level 1": {"accuracy": 0.85, "correct": 425, "total": 500},
                "Level 2": {"accuracy": 0.78, "correct": 780, "total": 1000},
                ...
            }
        """
        if not self.results:
            return {}

        # Group results by level
        level_results = defaultdict(list)
        for result in self.results:
            level = result.get("level", "Unknown")
            level_results[level].append(result)

        # Calculate metrics for each level
        level_metrics = {}
        for level, level_items in sorted(level_results.items()):
            total = len(level_items)
            correct = sum(1 for r in level_items if r.get("is_correct", False))
            accuracy = correct / total if total > 0 else 0.0

            level_metrics[level] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "percentage_of_dataset": total / len(self.results) if self.results else 0.0
            }

        return level_metrics

    def compute_by_type_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute accuracy breakdown by problem type.

        Returns:
            Dictionary mapping type names to metrics:
            {
                "Algebra": {"accuracy": 0.75, "correct": 600, "total": 800},
                "Geometry": {"accuracy": 0.68, "correct": 340, "total": 500},
                ...
            }
        """
        if not self.results:
            return {}

        # Group results by type
        type_results = defaultdict(list)
        for result in self.results:
            prob_type = result.get("type", "Unknown")
            type_results[prob_type].append(result)

        # Calculate metrics for each type
        type_metrics = {}
        for prob_type, type_items in sorted(type_results.items()):
            total = len(type_items)
            correct = sum(1 for r in type_items if r.get("is_correct", False))
            accuracy = correct / total if total > 0 else 0.0

            type_metrics[prob_type] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "percentage_of_dataset": total / len(self.results) if self.results else 0.0
            }

        return type_metrics

    def compute_detailed_statistics(self) -> Dict[str, Any]:
        """
        Compute additional detailed statistics.

        Returns:
            Dictionary with additional stats:
            - error_rate: Overall error rate
            - hardest_level: Level with lowest accuracy
            - easiest_level: Level with highest accuracy
            - hardest_type: Type with lowest accuracy
            - easiest_type: Type with highest accuracy
        """
        if not self.results:
            return {}

        overall = self.compute_overall_metrics()
        by_level = self.compute_by_level_metrics()
        by_type = self.compute_by_type_metrics()

        stats = {
            "error_rate": 1.0 - overall["accuracy"]
        }

        # Find hardest and easiest levels
        if by_level:
            sorted_levels = sorted(
                by_level.items(),
                key=lambda x: x[1]["accuracy"]
            )
            stats["hardest_level"] = {
                "level": sorted_levels[0][0],
                "accuracy": sorted_levels[0][1]["accuracy"]
            }
            stats["easiest_level"] = {
                "level": sorted_levels[-1][0],
                "accuracy": sorted_levels[-1][1]["accuracy"]
            }

        # Find hardest and easiest types
        if by_type:
            sorted_types = sorted(
                by_type.items(),
                key=lambda x: x[1]["accuracy"]
            )
            stats["hardest_type"] = {
                "type": sorted_types[0][0],
                "accuracy": sorted_types[0][1]["accuracy"]
            }
            stats["easiest_type"] = {
                "type": sorted_types[-1][0],
                "accuracy": sorted_types[-1][1]["accuracy"]
            }

        return stats

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics in one call.

        Returns:
            Dictionary containing:
            - overall: Overall metrics
            - by_level: Level-wise breakdown
            - by_type: Type-wise breakdown
            - detailed_stats: Additional statistics
        """
        return {
            "overall": self.compute_overall_metrics(),
            "by_level": self.compute_by_level_metrics(),
            "by_type": self.compute_by_type_metrics(),
            "detailed_stats": self.compute_detailed_statistics()
        }
