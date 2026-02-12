"""
Sanity check script to verify Qwen3 is working correctly on sample MATH problems.
"""
import json
import re
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams


# Sample MATH problems with known answers for testing
SAMPLE_PROBLEMS = [
    {
        "problem": "If $x + 2x + 3x + 4x + 5x = 90$, what is the value of $x$?",
        "expected_answer": "6",
        "level": "Level 1",
        "type": "algebra"
    },
    {
        "problem": "What is the sum of the first 10 positive integers?",
        "expected_answer": "55",
        "level": "Level 1",
        "type": "counting_and_probability"
    },
    {
        "problem": "Simplify: $2(3x - 5) + 4(x + 3)$",
        "expected_answer": "10x + 2",
        "level": "Level 2",
        "type": "algebra"
    },
]


def format_math_prompt(problem: str) -> str:
    """Format a MATH problem into a prompt for the model."""
    return f"""Solve the following math problem step by step. Show your reasoning and provide the final answer.

Problem: {problem}

Solution:"""


def extract_answer(solution_text: str) -> str:
    """
    Extract the final answer from the solution text.
    Looks for patterns like "The answer is X" or boxed answers.
    """
    # Try to find boxed answer (common in MATH dataset)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try to find "answer is" pattern
    answer_match = re.search(r'(?:answer|result|solution) is:?\s*([^\n.]+)', solution_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Try to find final number or expression in the last line
    lines = solution_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Remove common prefixes
        last_line = re.sub(r'^(?:Therefore|Thus|So|Hence),?\s*', '', last_line, flags=re.IGNORECASE)
        return last_line

    return solution_text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove common formatting
    answer = answer.strip()
    answer = re.sub(r'\\$', '', answer)  # Remove dollar signs
    answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
    return answer.lower()


def run_sanity_check(
    model_name: str = "Qwen/Qwen2.5-Math-14B-Instruct",
    tensor_parallel_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    """
    Run sanity check on sample problems.

    Args:
        model_name: HuggingFace model name
        tensor_parallel_size: Number of GPUs to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print(f"=" * 80)
    print(f"SANITY CHECK: Testing {model_name}")
    print(f"Using {tensor_parallel_size} GPU(s)")
    print(f"=" * 80)

    # Initialize VLLM
    print("\nInitializing model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )

    print("\nRunning sanity check on sample problems...\n")

    results = []
    correct = 0
    total = len(SAMPLE_PROBLEMS)

    for idx, problem_data in enumerate(SAMPLE_PROBLEMS):
        print(f"\n{'=' * 80}")
        print(f"Problem {idx + 1}/{total} [{problem_data['level']}, {problem_data['type']}]")
        print(f"{'=' * 80}")
        print(f"Problem: {problem_data['problem']}")
        print(f"Expected Answer: {problem_data['expected_answer']}")

        # Format prompt and generate
        prompt = format_math_prompt(problem_data['problem'])
        outputs = llm.generate([prompt], sampling_params)

        generated_solution = outputs[0].outputs[0].text
        extracted_answer = extract_answer(generated_solution)

        print(f"\nGenerated Solution:\n{generated_solution}")
        print(f"\n{'-' * 80}")
        print(f"Extracted Answer: {extracted_answer}")

        # Check if answer is correct (flexible matching)
        expected_norm = normalize_answer(problem_data['expected_answer'])
        extracted_norm = normalize_answer(extracted_answer)

        is_correct = expected_norm in extracted_norm or extracted_norm in expected_norm

        print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")

        if is_correct:
            correct += 1

        results.append({
            'problem': problem_data['problem'],
            'expected_answer': problem_data['expected_answer'],
            'generated_solution': generated_solution,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
        })

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Correct: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"{'=' * 80}\n")

    # Save results
    output_file = "outputs/sanity_check_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'model': model_name,
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'results': results,
        }, f, indent=2)

    print(f"Results saved to {output_file}")

    return correct == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for MATH inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Math-14B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    success = run_sanity_check(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    exit(0 if success else 1)
