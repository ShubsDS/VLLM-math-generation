# MATH Dataset Evaluation Pipeline

This subdirectory contains the evaluation pipeline for assessing model performance on the MATH dataset. The pipeline computes comprehensive accuracy metrics and generates reports in multiple formats.

## Overview

The evaluation pipeline consists of several modular components:

- **[answer_matcher.py](answer_matcher.py)**: Extract and normalize answers from solutions
- **[metrics.py](metrics.py)**: Compute accuracy metrics (overall, by-level, by-type)
- **[evaluator.py](evaluator.py)**: Core evaluation logic
- **[generate_report.py](generate_report.py)**: Multi-format report generation
- **[run_evaluation.py](run_evaluation.py)**: Main CLI entry point

## Quick Start

### Basic Evaluation

Evaluate generated solutions on the MATH dataset:

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"
```

This will:
1. Load ground truth from the metadata file
2. Load generated solutions from the solutions file
3. Match and evaluate each problem
4. Compute comprehensive metrics
5. Generate reports in all formats (JSON, CSV, Markdown)
6. Save results to `eval_results/`

### With Custom Output Directory

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output-dir eval_results/qwen25_7b
```

### Save Correct Solutions for SFT Training

**NEW!** Extract only correct solutions for supervised fine-tuning:

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --save-correct data/sft/correct_solutions.parquet
```

This will:
1. Evaluate all solutions and identify correct ones
2. Filter only the correct solutions (where predicted answer matches ground truth)
3. Save them to a parquet file ready for SFT training
4. Include metadata: `is_correct`, `confidence`, `level`, `type`

The output parquet file can be used directly with the SFT framework:

```bash
# Use correct solutions for training
TRAIN_FILE=data/sft/correct_solutions.parquet \
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
bash scripts/run_sft_training.sh
```

**Why this is useful:**
- **Self-improvement**: Fine-tune the model on its own correct solutions
- **Bootstrapping**: Create training data without manual annotation
- **Quality filtering**: Ensure training data only includes successful reasoning
- **Iterative refinement**: Generate → Evaluate → Fine-tune → Repeat

### Verbose Output (Show Per-Problem Results)

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --verbose
```

### Generate Only JSON Report

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_20260212_024306.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --report-format json
```

## Full Pipeline Example

Here's a complete workflow from loading the dataset to evaluation:

```bash
# Step 1: Load MATH dataset (if not already done)
python scripts/load_math_dataset.py --split test --subset all

# Step 2: Run inference with any HuggingFace model
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output-file outputs/qwen25_7b_solutions.jsonl

# Step 3: Evaluate
python evaluation/run_evaluation.py \
    --solutions-file outputs/qwen25_7b_solutions.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"
```

## Testing Different Models

You can easily test any model from HuggingFace:

```bash
# Test Qwen2.5-Math-1.5B
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct"
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Test Qwen2.5-Math-72B
python scripts/run_local_pipeline.py \
    --model-name "Qwen/Qwen2.5-Math-72B-Instruct" \
    --tensor-parallel-size 4
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-72B-Instruct"
```

## Output Files

The evaluation pipeline generates several output files:

### Reports Directory (`eval_results/reports/`)

1. **JSON Report** (`evaluation_<timestamp>.json`)
   - Complete metrics in machine-readable format
   - Contains metadata, overall metrics, by-level breakdown, by-type breakdown

2. **CSV Report** (`evaluation_<timestamp>.csv`)
   - Flat CSV format for spreadsheet analysis
   - One row per metric category
   - Easy to import into Excel, Google Sheets, etc.

3. **Markdown Report** (`evaluation_<timestamp>.md`)
   - Human-readable summary with formatted tables
   - Includes key insights (easiest/hardest levels and types)
   - Great for sharing results

### Detailed Directory (`eval_results/detailed/`)

4. **Detailed Results** (`detailed_results_<timestamp>.jsonl`)
   - Per-problem evaluation results
   - Contains problem text, generated answer, ground truth answer, correctness
   - Useful for error analysis and debugging

## Evaluation Metrics

The pipeline computes comprehensive metrics:

### Overall Metrics
- **Accuracy**: Percentage of correct answers
- **Correct/Total**: Count of correct answers out of total problems
- **Error Rate**: Percentage of incorrect answers
- **Avg Tokens/Solution**: Average number of tokens generated per solution

### By Difficulty Level
- Accuracy breakdown for Level 1, 2, 3, 4, 5
- Shows how model performs on problems of different difficulty
- Includes problem count and percentage of dataset for each level

### By Problem Type
- Accuracy breakdown by subject area:
  - Prealgebra
  - Algebra
  - Number Theory
  - Counting & Probability
  - Geometry
  - Intermediate Algebra
  - Precalculus
- Identifies which topics the model handles best/worst

### Key Insights
- Easiest and hardest difficulty levels
- Easiest and hardest problem types
- Helps identify areas for improvement

## Answer Extraction Strategy

The pipeline uses a robust answer extraction strategy:

1. **Priority 1**: Look for `\boxed{answer}` notation (most reliable for MATH dataset)
2. **Priority 2**: Search for "answer is X" patterns
3. **Priority 3**: Extract last line with prefix removal

Answers are normalized before comparison:
- Converted to lowercase
- Whitespace normalized
- LaTeX formatting removed
- Flexible substring matching (handles variations in format)

## Command-Line Options

```
usage: run_evaluation.py [-h] --solutions-file SOLUTIONS_FILE
                         --metadata-file METADATA_FILE
                         [--output-dir OUTPUT_DIR]
                         [--model-name MODEL_NAME]
                         [--report-format {json,csv,markdown,all}]
                         [--verbose] [--no-detailed]

Arguments:
  --solutions-file      Path to generated solutions JSONL file (required)
  --metadata-file       Path to ground truth metadata JSONL file (required)
  --output-dir          Output directory for results (default: eval_results)
  --model-name          Model name for report labeling (default: unknown)
  --report-format       Output format: json, csv, markdown, all (default: all)
  --verbose             Print per-problem evaluation results
  --no-detailed         Skip generating detailed per-problem results file
```

## Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================
Model: Qwen/Qwen2.5-Math-7B-Instruct
Total Problems: 5000
Correct: 3913
Accuracy: 78.26%
================================================================================

Performance by Difficulty Level:
  Level 1: 89.93%
  Level 2: 87.81%
  Level 3: 82.58%
  Level 4: 77.10%
  Level 5: 65.33%

Performance by Problem Type:
  Prealgebra: 89.44%
  Algebra: 86.86%
  Number Theory: 86.48%
  Counting & Probability: 78.69%
  Geometry: 69.31%
  Intermediate Algebra: 64.56%
  Precalculus: 63.74%
```

## File Format Requirements

### Solutions File Format

The solutions file must be in JSONL format with these fields:

```json
{
  "id": 0,
  "prompt": "Solve the following math problem...",
  "generated_solution": "Step 1: ...\nFinal Answer: 42",
  "num_tokens": 256
}
```

Required fields:
- `id`: Unique problem identifier (must match metadata file)
- `generated_solution`: The complete generated solution text

Optional fields:
- `prompt`: The prompt used (not used in evaluation)
- `num_tokens`: Token count (used for average calculation)

### Metadata File Format

The metadata file must be in JSONL format with these fields:

```json
{
  "id": 0,
  "problem": "If x + 2x + 3x + 4x + 5x = 90, what is the value of x?",
  "solution": "Combining like terms: 15x = 90, so x = \boxed{6}",
  "level": "Level 1",
  "type": "Algebra"
}
```

Required fields:
- `id`: Unique problem identifier (must match solutions file)
- `solution`: Ground truth solution with final answer
- `level`: Difficulty level (e.g., "Level 1", "Level 2")
- `type`: Problem type (e.g., "Algebra", "Geometry")

Optional fields:
- `problem`: Problem text (used in detailed results)

## Performance

The evaluation pipeline is highly efficient:

- **~5000 problems evaluated in <1 second**
- Memory efficient (loads all data into memory, ~100MB for full MATH dataset)
- Fast answer extraction and comparison
- Parallel processing not needed for MATH dataset scale

## Integration with Existing Code

The evaluation pipeline integrates seamlessly with the existing inference infrastructure:

- Uses same JSONL format as `run_vllm_inference.py`
- ID-based matching ensures correct problem-solution pairs
- Reuses answer extraction logic from `sanity_check.py`
- Consistent file handling patterns across the project

## Error Handling

The pipeline handles common issues gracefully:

- **Mismatched IDs**: Warns if solution ID not in ground truth (or vice versa)
- **Malformed JSON**: Skips invalid lines and continues
- **Missing fields**: Uses default values and logs warnings
- **Answer extraction failures**: Logs problems where extraction failed

## Extending the Pipeline

The modular design makes it easy to extend:

### Add New Metrics

Edit `metrics.py` and add methods to `MetricsCalculator`:

```python
def compute_by_custom_metric(self) -> Dict[str, Any]:
    # Your custom metric logic
    pass
```

### Add New Report Formats

Edit `generate_report.py` and add methods to `ReportGenerator`:

```python
def generate_latex_report(self) -> str:
    # Generate LaTeX report
    pass
```

### Custom Answer Extraction

Edit `answer_matcher.py` to add domain-specific extraction logic:

```python
def extract_chemistry_answer(text: str) -> str:
    # Custom extraction for chemistry problems
    pass
```

## Troubleshooting

### No Results Generated

If evaluation produces no results:
- Check that solution and metadata files have matching IDs
- Verify both files are valid JSONL format
- Use `--verbose` to see which problems are evaluated

### Low Accuracy

If accuracy is unexpectedly low:
- Check answer extraction with `--verbose`
- Verify the model output format matches expected format
- Review detailed results file for error patterns

### File Not Found

If you get file not found errors:
- Use absolute paths or ensure you're in the project root
- Check that files exist: `ls -lh outputs/*.jsonl data/*.jsonl`
- Verify file permissions

## Citation

If you use this evaluation pipeline, please cite the MATH dataset:

```bibtex
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora
          and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

## License

This evaluation pipeline is provided as-is for research purposes. Check the main project README for license information.
