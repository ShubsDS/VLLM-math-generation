"""
MATH Dataset Evaluation Pipeline

This package provides tools for evaluating mathematical problem-solving models
on the MATH dataset.

Key components:
- answer_matcher: Extract and normalize answers from solutions
- metrics: Compute accuracy metrics (overall, by-level, by-type)
- evaluator: Core evaluation logic
- generate_report: Multi-format report generation
- run_evaluation: Main CLI entry point
"""

from .answer_matcher import extract_answer, normalize_answer, compare_answers
from .metrics import MetricsCalculator
from .evaluator import MATHEvaluator
from .generate_report import ReportGenerator

__version__ = "1.0.0"

__all__ = [
    "extract_answer",
    "normalize_answer",
    "compare_answers",
    "MetricsCalculator",
    "MATHEvaluator",
    "ReportGenerator",
]
