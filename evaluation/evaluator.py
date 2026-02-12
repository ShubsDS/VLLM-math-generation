"""
Core evaluation logic for MATH dataset.

Provides the main MATHEvaluator class that orchestrates the evaluation process:
loading data, matching solutions with ground truth, and computing metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .answer_matcher import extract_answer, normalize_answer, compare_answers
from .metrics import MetricsCalculator


class MATHEvaluator:
    """Evaluator for MATH dataset solutions."""

    def __init__(
        self,
        solutions_file: str,
        metadata_file: str,
        model_name: str = "unknown",
        verbose: bool = False
    ):
        """
        Initialize the MATH evaluator.

        Args:
            solutions_file: Path to generated solutions JSONL file
            metadata_file: Path to ground truth metadata JSONL file
            model_name: Model name for reporting purposes
            verbose: If True, print per-problem evaluation results
        """
        self.solutions_file = Path(solutions_file)
        self.metadata_file = Path(metadata_file)
        self.model_name = model_name
        self.verbose = verbose

        self.ground_truth: Optional[Dict[int, Dict[str, Any]]] = None
        self.solutions: Optional[Dict[int, Dict[str, Any]]] = None
        self.results: List[Dict[str, Any]] = []

        # Validate files exist
        if not self.solutions_file.exists():
            raise FileNotFoundError(f"Solutions file not found: {self.solutions_file}")
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

    def load_ground_truth(self) -> Dict[int, Dict[str, Any]]:
        """
        Load ground truth metadata indexed by problem ID.

        Returns:
            Dictionary mapping problem ID to metadata dict containing:
            - id, problem, solution, level, type
        """
        print(f"Loading ground truth from {self.metadata_file}...")
        ground_truth = {}

        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    problem_id = item.get("id")
                    if problem_id is not None:
                        ground_truth[problem_id] = item
                    else:
                        print(f"Warning: Line {line_num} missing 'id' field")
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

        print(f"Loaded {len(ground_truth)} ground truth problems")
        self.ground_truth = ground_truth
        return ground_truth

    def load_solutions(self) -> Dict[int, Dict[str, Any]]:
        """
        Load generated solutions indexed by problem ID.

        Returns:
            Dictionary mapping problem ID to solution dict containing:
            - id, prompt, generated_solution, num_tokens
        """
        print(f"Loading solutions from {self.solutions_file}...")
        solutions = {}

        with open(self.solutions_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    problem_id = item.get("id")
                    if problem_id is not None:
                        solutions[problem_id] = item
                    else:
                        print(f"Warning: Line {line_num} missing 'id' field")
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

        print(f"Loaded {len(solutions)} generated solutions")
        self.solutions = solutions
        return solutions

    def evaluate_single(
        self,
        solution: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single problem.

        Args:
            solution: Generated solution dict with 'generated_solution' field
            ground_truth: Ground truth dict with 'solution', 'level', 'type' fields

        Returns:
            Dictionary with evaluation results for this problem:
            - id, problem, generated_answer, ground_truth_answer,
              is_correct, confidence, level, type, num_tokens
        """
        problem_id = solution.get("id")

        # Extract answers
        generated_solution = solution.get("generated_solution", "")
        ground_truth_solution = ground_truth.get("solution", "")

        generated_answer = extract_answer(generated_solution)
        ground_truth_answer = extract_answer(ground_truth_solution)

        # Compare answers
        is_correct, confidence = compare_answers(generated_answer, ground_truth_answer)

        # Build result
        result = {
            "id": problem_id,
            "problem": ground_truth.get("problem", ""),
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "is_correct": is_correct,
            "confidence": confidence,
            "level": ground_truth.get("level", "Unknown"),
            "type": ground_truth.get("type", "Unknown"),
            "num_tokens": solution.get("num_tokens", 0)
        }

        # Optional: Include full solutions for detailed analysis
        if self.verbose:
            result["generated_solution"] = generated_solution
            result["ground_truth_solution"] = ground_truth_solution

        return result

    def evaluate_all(self) -> List[Dict[str, Any]]:
        """
        Evaluate all problems.

        Matches solutions with ground truth by ID and evaluates each pair.
        Handles mismatches gracefully (warns about missing IDs).

        Returns:
            List of per-problem evaluation results
        """
        # Load data if not already loaded
        if self.ground_truth is None:
            self.load_ground_truth()
        if self.solutions is None:
            self.load_solutions()

        print("\nEvaluating solutions...")

        # Find common IDs
        solution_ids = set(self.solutions.keys())
        ground_truth_ids = set(self.ground_truth.keys())

        common_ids = solution_ids & ground_truth_ids
        missing_in_solutions = ground_truth_ids - solution_ids
        missing_in_ground_truth = solution_ids - ground_truth_ids

        # Warn about mismatches
        if missing_in_solutions:
            print(f"Warning: {len(missing_in_solutions)} problems in ground truth but not in solutions")
        if missing_in_ground_truth:
            print(f"Warning: {len(missing_in_ground_truth)} problems in solutions but not in ground truth")

        # Evaluate each problem
        results = []
        for problem_id in tqdm(sorted(common_ids), desc="Evaluating"):
            solution = self.solutions[problem_id]
            ground_truth = self.ground_truth[problem_id]

            result = self.evaluate_single(solution, ground_truth)
            results.append(result)

            if self.verbose:
                status = "✓" if result["is_correct"] else "✗"
                print(f"{status} Problem {problem_id}: {result['generated_answer']} vs {result['ground_truth_answer']}")

        self.results = results
        print(f"\nEvaluation complete: {len(results)} problems evaluated")
        return results

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics from evaluation results.

        Must call evaluate_all() first to populate results.

        Returns:
            Dictionary with all computed metrics:
            - overall, by_level, by_type, detailed_stats
        """
        if not self.results:
            print("Warning: No results available. Call evaluate_all() first.")
            return {}

        calculator = MetricsCalculator(self.results)
        metrics = calculator.compute_all_metrics()

        # Add metadata
        metrics["metadata"] = {
            "model_name": self.model_name,
            "solutions_file": str(self.solutions_file),
            "metadata_file": str(self.metadata_file),
            "total_problems_evaluated": len(self.results)
        }

        return metrics

    def get_incorrect_problems(self) -> List[Dict[str, Any]]:
        """
        Get list of incorrectly solved problems for error analysis.

        Returns:
            List of evaluation results for incorrect problems
        """
        return [r for r in self.results if not r.get("is_correct", False)]

    def get_correct_problems(self) -> List[Dict[str, Any]]:
        """
        Get list of correctly solved problems.

        Returns:
            List of evaluation results for correct problems
        """
        return [r for r in self.results if r.get("is_correct", False)]

    def get_problems_by_level(self, level: str) -> List[Dict[str, Any]]:
        """
        Get all problems of a specific difficulty level.

        Args:
            level: Level name (e.g., "Level 1", "Level 2")

        Returns:
            List of evaluation results for that level
        """
        return [r for r in self.results if r.get("level") == level]

    def get_problems_by_type(self, prob_type: str) -> List[Dict[str, Any]]:
        """
        Get all problems of a specific type.

        Args:
            prob_type: Problem type (e.g., "Algebra", "Geometry")

        Returns:
            List of evaluation results for that type
        """
        return [r for r in self.results if r.get("type") == prob_type]
