# Evaluation Pipeline Quick Start

This guide shows you how to quickly evaluate any model on the MATH dataset.

## Prerequisites

Make sure you have:
1. Prepared the MATH dataset: `data/math_test_metadata.jsonl` exists
2. Generated solutions with your model: `outputs/math_test_solutions_*.jsonl` exists

## Quick Evaluation

### Basic Usage

Evaluate your model with a single command:

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"
```

**Output:**
```
================================================================================
EVALUATION SUMMARY
================================================================================
Model: Qwen/Qwen2.5-Math-7B-Instruct
Total Problems: 5000
Correct: 3913
Accuracy: 78.26%
================================================================================
```

Results saved to `eval_results/` with:
- JSON report (machine-readable)
- CSV report (spreadsheet-friendly)
- Markdown report (human-readable)
- Detailed per-problem results (for error analysis)

## Complete Workflow

### 1. Load the MATH Dataset

```bash
python scripts/load_math_dataset.py --split test --subset all
```

Creates:
- `data/math_test_prompts.jsonl` - Prompts for inference
- `data/math_test_metadata.jsonl` - Ground truth solutions

### 2. Run Inference with Any Model

**Example 1: Qwen2.5-Math-7B**
```bash
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --tensor-parallel-size 1
```

**Example 2: Qwen2.5-Math-1.5B (smaller model)**
```bash
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --tensor-parallel-size 1
```

**Example 3: Qwen2.5-Math-72B (larger model)**
```bash
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-72B-Instruct" \
    --tensor-parallel-size 4
```

**Example 4: Any Other HuggingFace Model**
```bash
python scripts/run_local_pipeline.py \
    --model-name "meta-llama/Llama-3-8B-Instruct" \
    --tensor-parallel-size 1
```

This generates: `outputs/math_test_solutions_<timestamp>.jsonl`

### 3. Evaluate

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_<timestamp>.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Your/Model-Name"
```

## Example: Compare Multiple Models

### Evaluate Qwen2.5-Math-7B

```bash
# Run inference
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"

# Evaluate
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output-dir eval_results/qwen25_7b
```

### Evaluate Qwen2.5-Math-1.5B

```bash
# Run inference
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Evaluate
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --output-dir eval_results/qwen25_1.5b
```

### Compare Results

```bash
# View Qwen2.5-Math-7B results
cat eval_results/qwen25_7b/reports/evaluation_*.md

# View Qwen2.5-Math-1.5B results
cat eval_results/qwen25_1.5b/reports/evaluation_*.md
```

## Understanding the Results

### Markdown Report Preview

```markdown
# MATH Dataset Evaluation Report
**Model:** Qwen/Qwen2.5-Math-7B-Instruct
**Date:** 2026-02-12 10:23:23
**Total Problems:** 5000

## Overall Performance
- **Accuracy:** 78.26%
- **Correct:** 3913 / 5000
- **Error Rate:** 21.74%
- **Avg Tokens/Solution:** 663.1

## Performance by Difficulty Level
| Level | Accuracy | Correct | Total | % of Dataset |
|-------|----------|---------|-------|-------------|
| Level 1 | 89.93% | 393 | 437 | 8.7% |
| Level 2 | 87.81% | 785 | 894 | 17.9% |
| Level 3 | 82.58% | 934 | 1131 | 22.6% |
| Level 4 | 77.10% | 936 | 1214 | 24.3% |
| Level 5 | 65.33% | 865 | 1324 | 26.5% |

## Performance by Problem Type
| Type | Accuracy | Correct | Total | % of Dataset |
|------|----------|---------|-------|-------------|
| Prealgebra | 89.44% | 779 | 871 | 17.4% |
| Algebra | 86.86% | 1031 | 1187 | 23.7% |
| Number Theory | 86.48% | 467 | 540 | 10.8% |
| Counting & Probability | 78.69% | 373 | 474 | 9.5% |
| Geometry | 69.31% | 332 | 479 | 9.6% |
| Intermediate Algebra | 64.56% | 583 | 903 | 18.1% |
| Precalculus | 63.74% | 348 | 546 | 10.9% |

## Key Insights
- **Easiest Level:** Level 1 (89.93%)
- **Hardest Level:** Level 5 (65.33%)
- **Easiest Type:** Prealgebra (89.44%)
- **Hardest Type:** Precalculus (63.74%)
```

### Key Metrics Explained

- **Overall Accuracy**: Percentage of correctly solved problems
- **By Difficulty Level**: Shows if the model struggles with harder problems
- **By Problem Type**: Identifies which math topics the model handles best
- **Easiest/Hardest**: Quick insights into model strengths and weaknesses

## Advanced Options

### Verbose Output

See per-problem results as they're evaluated:

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --verbose
```

### Generate Only JSON

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --report-format json
```

### Custom Output Directory

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --output-dir my_custom_results
```

## Analyzing Errors

To see which problems the model got wrong:

```bash
# The detailed results file contains all problems with correctness flags
cat eval_results/detailed/detailed_results_*.jsonl | jq 'select(.is_correct == false)'
```

To count errors by type:

```bash
cat eval_results/detailed/detailed_results_*.jsonl | \
    jq -r 'select(.is_correct == false) | .type' | \
    sort | uniq -c | sort -rn
```

To see errors for a specific type (e.g., Geometry):

```bash
cat eval_results/detailed/detailed_results_*.jsonl | \
    jq 'select(.is_correct == false and .type == "Geometry")'
```

## Using Slurm

For large-scale evaluations on a cluster:

```bash
# Run inference via Slurm
sbatch slurm/run_inference.slurm

# After job completes, run evaluation
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_solutions_<JOB_ID>.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"
```

## Troubleshooting

### "File not found" error

Make sure you're in the project root directory:
```bash
cd /u/fvc9ch/nlp_research/VLLM-math-generation
python evaluation/run_evaluation.py ...
```

### No results generated

Check that your solutions file has matching IDs with the metadata:
```bash
# Check solutions file
head -1 outputs/math_test_solutions_*.jsonl | jq '.id'

# Check metadata file
head -1 data/math_test_metadata.jsonl | jq '.id'
```

### Low accuracy

Use `--verbose` to see which problems are being marked incorrect:
```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --verbose | head -50
```

## Need More Help?

- **Detailed documentation**: See [evaluation/README.md](evaluation/README.md)
- **Main project README**: See [README.md](README.md)
- **Inference help**: See scripts in [scripts/](scripts/)

## Quick Reference

```bash
# Full pipeline (one model)
python scripts/load_math_dataset.py --split test --subset all
python scripts/run_local_pipeline.py --model-name "MODEL_NAME"
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "MODEL_NAME"

# View results
cat eval_results/reports/evaluation_*.md
```

That's it! You can now evaluate any HuggingFace model on the MATH dataset. 🎉
