"""
Answer extraction and matching utilities for MATH dataset evaluation.

Refactored and enhanced from sanity_check.py for reusability across the evaluation pipeline.
"""

import re
from typing import Tuple, Optional


def extract_answer(solution_text: str) -> str:
    """
    Extract the final answer from a solution text.

    Strategy (in priority order):
    1. Look for \boxed{...} notation (most reliable for MATH dataset)
    2. Try "answer is X" patterns
    3. Fall back to last line extraction with prefix removal

    Args:
        solution_text: The complete solution text

    Returns:
        Extracted answer string
    """
    if not solution_text or not solution_text.strip():
        return ""

    # Strategy 1: Look for boxed answer (common in MATH dataset)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Strategy 2: Try to find "answer is" pattern
    answer_match = re.search(
        r'(?:answer|result|solution) is:?\s*([^\n.]+)',
        solution_text,
        re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Strategy 3: Try to find final number or expression in the last line
    lines = solution_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Remove common prefixes
        last_line = re.sub(
            r'^(?:Therefore|Thus|So|Hence),?\s*',
            '',
            last_line,
            flags=re.IGNORECASE
        )
        return last_line

    return solution_text.strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer specifically from \boxed{} notation.

    Args:
        text: Text potentially containing \boxed{...}

    Returns:
        Content inside \boxed{} or None if not found
    """
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer from text.

    Looks for standalone numbers, fractions, or simple expressions.

    Args:
        text: Text potentially containing numeric answer

    Returns:
        Extracted numeric answer or None if not found
    """
    # Look for fractions like 1/2, 3/4
    fraction_match = re.search(r'\b(\d+/\d+)\b', text)
    if fraction_match:
        return fraction_match.group(1)

    # Look for decimals like 3.14, 0.5
    decimal_match = re.search(r'\b(\d+\.\d+)\b', text)
    if decimal_match:
        return decimal_match.group(1)

    # Look for integers
    integer_match = re.search(r'\b(\d+)\b', text)
    if integer_match:
        return integer_match.group(1)

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Normalizations applied:
    - Strip whitespace
    - Remove LaTeX dollar signs
    - Normalize whitespace (collapse multiple spaces)
    - Convert to lowercase
    - Remove common LaTeX commands

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Remove common formatting
    answer = answer.strip()

    # Remove dollar signs (LaTeX math mode)
    answer = re.sub(r'\$', '', answer)

    # Remove backslashes from common LaTeX commands but keep the content
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)

    # Remove remaining backslashes
    answer = re.sub(r'\\', '', answer)

    # Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer)

    # Convert to lowercase for case-insensitive comparison
    return answer.lower().strip()


def compare_answers(generated: str, ground_truth: str) -> Tuple[bool, float]:
    """
    Compare generated answer with ground truth.

    Uses flexible substring matching to handle variations in formatting.
    This matches the approach from sanity_check.py for consistency.

    Args:
        generated: Generated answer from the model
        ground_truth: Ground truth answer from dataset

    Returns:
        Tuple of (is_correct: bool, confidence: float)
        - is_correct: True if answers match
        - confidence: 1.0 for exact match, 0.8 for substring match, 0.0 for no match
    """
    if not generated or not ground_truth:
        return (False, 0.0)

    # Normalize both answers
    gen_norm = normalize_answer(generated)
    truth_norm = normalize_answer(ground_truth)

    # Exact match (highest confidence)
    if gen_norm == truth_norm:
        return (True, 1.0)

    # Flexible substring matching (like sanity_check.py)
    # Check if either is a substring of the other
    is_correct = (truth_norm in gen_norm) or (gen_norm in truth_norm)

    if is_correct:
        return (True, 0.8)

    return (False, 0.0)


def evaluate_correctness(expected_answer: str, extracted_answer: str) -> bool:
    """
    Legacy function for compatibility with sanity_check.py patterns.

    Args:
        expected_answer: Ground truth answer
        extracted_answer: Generated answer

    Returns:
        True if answers match, False otherwise
    """
    is_correct, _ = compare_answers(extracted_answer, expected_answer)
    return is_correct
