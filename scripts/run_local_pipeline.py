#!/usr/bin/env python3
"""Run MATH inference + evaluation locally in a single script."""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MATH_DATASET_CANDIDATES = [
    "hendrycks/competition_math",
    "EleutherAI/hendrycks_math",
    "Maxwell-Jia/MATH",
    "jeggers/competition_math",
]


def normalize_subset_name(value: str) -> str:
    return value.strip().lower().replace("&", "and").replace(" ", "_")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def format_math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. Show your reasoning and provide "
        "the final answer.\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


def validate_transformers_compatibility() -> None:
    import transformers

    version = transformers.__version__
    major_str = version.split(".", 1)[0]
    try:
        major = int(major_str)
    except ValueError:
        return

    if major >= 5:
        raise RuntimeError(
            f"Incompatible transformers version detected: {version}. "
            "This project requires transformers<5 for vLLM tokenizer compatibility. "
            "Run: uv pip install 'transformers>=4.55.2,<5' 'tokenizers>=0.21.1,<0.23'"
        )


def validate_numpy_compatibility() -> None:
    import numpy as np

    version = np.__version__
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return

    if major > 2 or (major == 2 and minor > 2):
        raise RuntimeError(
            f"Incompatible numpy version detected: {version}. "
            "vLLM's numba dependency requires numpy<=2.2. "
            "Run: uv pip install 'numpy>=1.24.0,<2.3'"
        )


def load_math_split(split: str, subset: str):
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    errors = {}
    normalized_subset = normalize_subset_name(subset)

    for dataset_id in MATH_DATASET_CANDIDATES:
        try:
            if normalized_subset == "all":
                try:
                    return load_dataset(dataset_id, split=split), dataset_id
                except Exception:
                    config_names = get_dataset_config_names(dataset_id)
                    if not config_names:
                        raise
                    combined = concatenate_datasets(
                        [load_dataset(dataset_id, name=name, split=split) for name in config_names]
                    )
                    return combined, dataset_id

            try:
                return load_dataset(dataset_id, name=normalized_subset, split=split), dataset_id
            except Exception:
                pass

            full_split = load_dataset(dataset_id, split=split)
            if "type" not in full_split.column_names:
                raise ValueError("dataset is missing 'type' column")

            filtered = full_split.filter(
                lambda x: normalize_subset_name(x["type"]) == normalized_subset
            )
            if len(filtered) == 0:
                available = sorted(
                    {normalize_subset_name(t) for t in full_split["type"] if isinstance(t, str)}
                )
                raise ValueError(
                    f"subset '{subset}' not found in dataset types: {available[:20]}"
                )
            return filtered, dataset_id

        except Exception as exc:
            errors[dataset_id] = str(exc)

    error_lines = "\n".join([f"- {ds}: {err}" for ds, err in errors.items()])
    raise RuntimeError(
        "Failed to load MATH dataset from known sources.\n"
        f"Tried:\n{error_lines}"
    )


def prepare_examples(
    split: str,
    subset: str,
    max_samples: int | None,
    shuffle: bool = False,
    seed: int = 42,
):
    dataset, source_id = load_math_split(split=split, subset=subset)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    examples = []
    for idx, row in enumerate(dataset):
        examples.append(
            {
                "id": idx,
                "problem": row["problem"],
                "solution": row["solution"],
                "level": row.get("level", "Unknown"),
                "type": row.get("type", "Unknown"),
                "prompt": format_math_prompt(row["problem"]),
            }
        )

    return examples, source_id


def _extract_boxed_str(text: str) -> str:
    """Extract the last \\boxed{} content using brace counting. Display only."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return text.strip()
    i = idx
    depth = 0
    start = None
    while i < len(text):
        if text[i] == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i].strip()
        i += 1
    return text.strip()


def compare_answers(generated_solution: str, ground_truth_solution: str) -> tuple[bool, str]:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

    extraction_config = [
        LatexExtractionConfig(boxed_match_priority=0),
        ExprExtractionConfig(),
    ]
    gold = parse(ground_truth_solution, extraction_config=extraction_config)
    pred = parse(generated_solution, extraction_config=extraction_config)

    if not gold or not pred:
        return False, "parse_failure"

    return bool(verify(gold, pred)), "sympy_equiv"


def evaluate_outputs(outputs: list[dict], examples: list[dict]) -> tuple[list[dict], dict]:
    by_id = {example["id"]: example for example in examples}
    results = []
    parse_failures = 0

    for output in outputs:
        problem_id = output["id"]
        example = by_id.get(problem_id)
        if example is None:
            continue

        generated_solution = output.get("generated_solution", "")
        ground_truth_solution = example.get("solution", "")
        is_correct, match_type = compare_answers(generated_solution, ground_truth_solution)
        if match_type == "parse_failure":
            parse_failures += 1
        generated_answer = _extract_boxed_str(generated_solution)
        ground_truth_answer = _extract_boxed_str(ground_truth_solution)

        results.append(
            {
                "id": problem_id,
                "sample_idx": output.get("sample_idx", 0),
                "level": example.get("level", "Unknown"),
                "type": example.get("type", "Unknown"),
                "is_correct": is_correct,
                "match_type": match_type,
                "generated_answer": generated_answer,
                "ground_truth_answer": ground_truth_answer,
                "output_tokens": output.get("output_tokens", 0),
                "prompt_tokens": output.get("prompt_tokens", 0),
            }
        )

    # Any-correct deduplication: a problem counts as correct if any sample is correct.
    any_correct: dict[int, bool] = {}
    any_level: dict[int, str] = {}
    any_type: dict[int, str] = {}
    for row in results:
        pid = row["id"]
        if pid not in any_correct:
            any_correct[pid] = row["is_correct"]
            any_level[pid] = row["level"]
            any_type[pid] = row["type"]
        elif row["is_correct"]:
            any_correct[pid] = True

    total = len(any_correct)
    correct = sum(1 for v in any_correct.values() if v)
    accuracy = correct / total if total else 0.0

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for pid, is_correct in any_correct.items():
        lvl = any_level[pid]
        typ = any_type[pid]
        by_level[lvl]["total"] += 1
        by_type[typ]["total"] += 1
        if is_correct:
            by_level[lvl]["correct"] += 1
            by_type[typ]["correct"] += 1

    total_samples = len(results)
    metrics = {
        "overall": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "total_samples": total_samples,
            "parse_failures": parse_failures,
            "avg_output_tokens": sum(r["output_tokens"] for r in results) / total_samples if total_samples else 0.0,
            "avg_prompt_tokens": sum(r["prompt_tokens"] for r in results) / total_samples if total_samples else 0.0,
        },
        "by_level": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": (value["correct"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_level.items())
        },
        "by_type": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": (value["correct"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_type.items())
        },
    }

    return results, metrics


def write_markdown_report(
    report_path: Path,
    model_name: str,
    split: str,
    subset: str,
    dataset_source: str,
    metrics: dict,
) -> None:
    overall = metrics["overall"]
    by_level = metrics["by_level"]
    by_type = metrics["by_type"]

    total_samples = overall.get("total_samples", overall["total"])
    parse_pct = overall["parse_failures"] / total_samples if total_samples else 0.0

    lines = [
        "# MATH Evaluation Report\n",
        f"- Model: `{model_name}`\n",
        f"- Dataset source: `{dataset_source}`\n",
        f"- Split: `{split}`\n",
        f"- Subset: `{subset}`\n",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n",
        "\n",
        "## Overall\n",
        f"- Accuracy: **{overall['accuracy']:.2%}**\n",
        f"- Correct: **{overall['correct']} / {overall['total']}**\n",
        f"- Avg output tokens: **{overall['avg_output_tokens']:.1f}**\n",
        f"- Avg prompt tokens: **{overall['avg_prompt_tokens']:.1f}**\n",
        f"- Parse Failures: **{overall['parse_failures']}** ({parse_pct:.2%})\n",
        "\n",
    ]

    if by_level:
        lines.extend(
            [
                "## Accuracy by Level\n",
                "| Level | Accuracy | Correct | Total |\n",
                "|---|---:|---:|---:|\n",
            ]
        )
        for key, value in by_level.items():
            lines.append(
                f"| {key} | {value['accuracy']:.2%} | {value['correct']} | {value['total']} |\n"
            )
        lines.append("\n")

    if by_type:
        lines.extend(
            [
                "## Accuracy by Type\n",
                "| Type | Accuracy | Correct | Total |\n",
                "|---|---:|---:|---:|\n",
            ]
        )
        for key, value in by_type.items():
            lines.append(
                f"| {key} | {value['accuracy']:.2%} | {value['correct']} | {value['total']} |\n"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def run_inference(
    model_name: str,
    examples: list[dict],
    tensor_parallel_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    num_samples: int = 1,
) -> list[dict]:
    from vllm import LLM, SamplingParams

    prompts = [row["prompt"] for row in examples]

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        n=num_samples,
    )

    outputs = llm.generate(prompts, sampling_params)
    rows = []
    for idx, output in enumerate(outputs):
        ex = examples[idx]
        for sample_idx, gen in enumerate(output.outputs):
            rows.append({
                "id": ex["id"],
                "problem": ex["problem"],
                "solution": ex["solution"],
                "level": ex["level"],
                "type": ex["type"],
                "prompt": ex["prompt"],
                "generated_solution": gen.text,
                "sample_idx": sample_idx,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(gen.token_ids),
            })

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="Run MATH inference + evaluation locally")
    parser.add_argument("--model", required=True, help="HF model ID or local model path")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="MATH split")
    parser.add_argument("--subset", default="all", help="MATH subject subset, e.g. algebra or all")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before applying --max-samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of generations per problem")
    parser.add_argument("--output-dir", default="outputs", help="Directory for full generations")
    parser.add_argument("--eval-dir", default="eval_results", help="Directory for markdown report")
    parser.add_argument(
        "--save-correct-jsonl",
        action="store_true",
        help="Also save a correct-only JSONL file",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="GPUs for tensor parallelism (default: all visible)",
    )
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max new tokens")
    parser.add_argument("--max-model-len", type=int, default=16384, help="vLLM max context length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory vLLM may use (default: 0.95)",
    )

    args = parser.parse_args()

    validate_transformers_compatibility()
    validate_numpy_compatibility()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pipeline.")

    model_name = args.model
    model_slug = slugify(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_gpus = torch.cuda.device_count()
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else num_gpus
    tp_size = max(1, min(tp_size, num_gpus))

    print("=== MATH local pipeline ===")
    print(f"Model: {model_name}")
    print(f"Split/subset: {args.split}/{args.subset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Shuffle: {args.shuffle} (seed={args.seed})")
    print(f"Num samples per problem: {args.num_samples}")
    print(f"Tensor parallel size: {tp_size} (visible GPUs: {num_gpus})")

    examples, dataset_source = prepare_examples(
        split=args.split,
        subset=args.subset,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Loaded {len(examples)} examples from {dataset_source}")

    outputs = run_inference(
        model_name=model_name,
        examples=examples,
        tensor_parallel_size=tp_size,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_samples=args.num_samples,
    )

    output_path = Path(args.output_dir) / f"{model_slug}_{args.split}_{timestamp}.jsonl"
    write_jsonl(output_path, outputs)
    print(f"Saved generations: {output_path}")

    results, metrics = evaluate_outputs(outputs=outputs, examples=examples)
    report_path = Path(args.eval_dir) / f"{model_slug}_{timestamp}.md"
    write_markdown_report(
        report_path=report_path,
        model_name=model_name,
        split=args.split,
        subset=args.subset,
        dataset_source=dataset_source,
        metrics=metrics,
    )
    print(f"Saved markdown report: {report_path}")

    if args.save_correct_jsonl:
        correct_ids = {row["id"] for row in results if row["is_correct"]}
        correct_rows = [row for row in outputs if row["id"] in correct_ids]
        correct_path = Path(args.output_dir) / f"{model_slug}_{args.split}_{timestamp}.correct.jsonl"
        write_jsonl(correct_path, correct_rows)
        print(f"Saved correct-only JSONL: {correct_path} ({len(correct_rows)}/{len(outputs)})")

    print(
        f"Accuracy: {metrics['overall']['accuracy']:.2%} "
        f"({metrics['overall']['correct']}/{metrics['overall']['total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
