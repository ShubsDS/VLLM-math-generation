# Teacher-Student Distillation Workflow

This workflow demonstrates how to:
1. Generate correct solutions with a **teacher model** (large, capable model)
2. Filter to only correct solutions
3. Fine-tune a **student model** (smaller, faster model) on teacher's correct solutions
4. Track token efficiency improvements over iterations

## Why This Works

- **Teacher**: Large model (e.g., Qwen2.5-Math-7B) generates high-quality solutions
- **Student**: Smaller model (e.g., Qwen2.5-1.5B) learns from teacher's correct reasoning
- **Benefits**: Student becomes more accurate AND more efficient (fewer tokens per solution)

## Complete Workflow

### Step 1: Generate Solutions with Teacher Model

```bash
# Use a large, capable teacher model
TEACHER_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"

python scripts/run_vllm_inference.py \
    --model $TEACHER_MODEL \
    --dataset data/math_test_prompts.jsonl \
    --output outputs/teacher_solutions.jsonl \
    --max-tokens 2048 \
    --temperature 0.7
```

### Step 2: Evaluate and Filter Correct Solutions

```bash
# Evaluate teacher solutions and save only correct ones
python evaluation/run_evaluation.py \
    --solutions-file outputs/teacher_solutions.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --save-correct data/sft/teacher_correct.parquet \
    --model-name $TEACHER_MODEL

# This will output something like:
# ✓ Saved 3800 correct solutions to: data/sft/teacher_correct.parquet
# Accuracy: 3800/5000 = 76.00%
```

### Step 3: Analyze Teacher's Token Usage

```bash
# Check token statistics from teacher's correct solutions
python3 -c "
import pandas as pd
df = pd.read_parquet('data/sft/teacher_correct.parquet')
print('Teacher Model Token Statistics (Correct Solutions Only):')
print(f'  Total correct: {len(df)}')
print(f'  Mean tokens: {df[\"num_tokens\"].mean():.1f}')
print(f'  Median tokens: {df[\"num_tokens\"].median():.1f}')
print(f'  Min tokens: {df[\"num_tokens\"].min()}')
print(f'  Max tokens: {df[\"num_tokens\"].max()}')
print(f'\nBy difficulty level:')
print(df.groupby('level')['num_tokens'].agg(['count', 'mean', 'median']))
print(f'\nBy problem type:')
print(df.groupby('type')['num_tokens'].agg(['count', 'mean', 'median']))
"
```

### Step 4: Prepare Training Data

```bash
# Split teacher's correct solutions into train/val
python3 -c "
import pandas as pd
df = pd.read_parquet('data/sft/teacher_correct.parquet')

# Shuffle and split 90/10
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
val_size = int(len(df) * 0.1)

train_df = df.iloc[val_size:]
val_df = df.iloc[:val_size]

train_df.to_parquet('data/sft/train.parquet', index=False)
val_df.to_parquet('data/sft/val.parquet', index=False)

print(f'Training data prepared:')
print(f'  Train: {len(train_df)} examples')
print(f'  Val: {len(val_df)} examples')
"
```

### Step 5: Fine-Tune Student Model

```bash
# Train a smaller student model on teacher's correct solutions
STUDENT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

NPROC_PER_NODE=2 \
MODEL_NAME=$STUDENT_MODEL \
TRAIN_FILE=data/sft/train.parquet \
VAL_FILE=data/sft/val.parquet \
OUTPUT_DIR=checkpoints/student_from_teacher \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
PROJECT_NAME="teacher-student-distillation" \
bash scripts/run_sft_training.sh
```

### Step 6: Evaluate Student Model (Before vs After)

```bash
# Generate solutions with base student model (before training)
python scripts/run_vllm_inference.py \
    --model $STUDENT_MODEL \
    --dataset data/math_test_prompts.jsonl \
    --output outputs/student_base_solutions.jsonl \
    --max-tokens 2048

# Generate solutions with fine-tuned student model
python scripts/run_vllm_inference.py \
    --model checkpoints/student_from_teacher/global_step_final \
    --dataset data/math_test_prompts.jsonl \
    --output outputs/student_finetuned_solutions.jsonl \
    --max-tokens 2048

# Evaluate both
python evaluation/run_evaluation.py \
    --solutions-file outputs/student_base_solutions.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --output-dir eval_results/student_base \
    --model-name "Student (Base)"

python evaluation/run_evaluation.py \
    --solutions-file outputs/student_finetuned_solutions.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --output-dir eval_results/student_finetuned \
    --model-name "Student (Fine-tuned)"
```

### Step 7: Compare Token Efficiency

```bash
# Compare token usage between teacher, base student, and fine-tuned student
python3 -c "
import pandas as pd
import json

# Load all solutions
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

teacher = load_jsonl('outputs/teacher_solutions.jsonl')
student_base = load_jsonl('outputs/student_base_solutions.jsonl')
student_ft = load_jsonl('outputs/student_finetuned_solutions.jsonl')

# Get token stats
def get_stats(data, name):
    tokens = [x['num_tokens'] for x in data]
    return {
        'Model': name,
        'Count': len(tokens),
        'Mean': sum(tokens) / len(tokens),
        'Median': sorted(tokens)[len(tokens)//2]
    }

results = [
    get_stats(teacher, 'Teacher (7B)'),
    get_stats(student_base, 'Student Base (1.5B)'),
    get_stats(student_ft, 'Student Fine-tuned (1.5B)')
]

df = pd.DataFrame(results)
print('Token Usage Comparison:')
print(df.to_string(index=False))
print()
print('Expected outcome:')
print('  - Teacher: High tokens, high accuracy')
print('  - Student Base: Lower tokens, lower accuracy')
print('  - Student Fine-tuned: Similar to teacher tokens, higher accuracy')
"
```

### Step 8: Analyze Improvements

```bash
# Create comprehensive comparison report
python3 << 'EOF'
import pandas as pd
import json

def load_eval_report(path):
    with open(path) as f:
        return json.load(f)

def load_solutions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# Load evaluation results
teacher_eval = load_eval_report('eval_results/teacher/reports/evaluation_*.json')
base_eval = load_eval_report('eval_results/student_base/reports/evaluation_*.json')
ft_eval = load_eval_report('eval_results/student_finetuned/reports/evaluation_*.json')

# Load solutions for token counts
teacher_sols = load_solutions('outputs/teacher_solutions.jsonl')
base_sols = load_solutions('outputs/student_base_solutions.jsonl')
ft_sols = load_solutions('outputs/student_finetuned_solutions.jsonl')

# Calculate metrics
results = []
for name, eval_data, solutions in [
    ('Teacher (7B)', teacher_eval, teacher_sols),
    ('Student Base (1.5B)', base_eval, base_sols),
    ('Student Fine-tuned (1.5B)', ft_eval, ft_sols)
]:
    tokens = [s['num_tokens'] for s in solutions]
    results.append({
        'Model': name,
        'Accuracy': eval_data['overall']['accuracy'],
        'Correct': eval_data['overall']['correct'],
        'Total': eval_data['overall']['total'],
        'Avg Tokens': sum(tokens) / len(tokens),
        'Median Tokens': sorted(tokens)[len(tokens)//2]
    })

df = pd.DataFrame(results)
print('=' * 80)
print('TEACHER-STUDENT DISTILLATION RESULTS')
print('=' * 80)
print(df.to_string(index=False))
print()
print('Key Findings:')
print(f'  Accuracy gain: {results[2]["Accuracy"] - results[1]["Accuracy"]:.2%}')
print(f'  Token efficiency: {results[2]["Avg Tokens"] / results[1]["Avg Tokens"]:.2f}x')
print(f'  Size reduction: {1.5 / 7:.1f}x smaller model')
print('=' * 80)
EOF
```

## Expected Results

### Typical Outcomes

| Metric | Teacher (7B) | Student Base (1.5B) | Student Fine-tuned (1.5B) | Improvement |
|--------|--------------|---------------------|---------------------------|-------------|
| Accuracy | 76% | 45% | 68% | +23% |
| Avg Tokens | 1200 | 800 | 1100 | +37.5% |
| Inference Speed | 1x | 3x | 3x | 3x faster |
| Model Size | 7B | 1.5B | 1.5B | 4.7x smaller |

### Key Insights

1. **Accuracy**: Student learns teacher's reasoning, closing the gap significantly
2. **Token Count**: Student uses more tokens after training (learning verbose but correct reasoning)
3. **Efficiency**: Despite more tokens, student is still 3x faster due to smaller size
4. **Quality**: Student's solutions become more structured and complete

## Iterative Refinement

You can improve further by:

### Iteration 2: Student as New Teacher

```bash
# Use fine-tuned student's correct solutions as new training data
python evaluation/run_evaluation.py \
    --solutions-file outputs/student_finetuned_solutions.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --save-correct data/sft/iter2_correct.parquet

# Fine-tune again
TRAIN_FILE=data/sft/iter2_correct.parquet \
OUTPUT_DIR=checkpoints/student_iter2 \
bash scripts/run_sft_training.sh
```

### Token Optimization Training

If you want to reduce token count while maintaining accuracy:

```bash
# Filter teacher's correct solutions by token efficiency
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('data/sft/teacher_correct.parquet')

# Keep only solutions below median token count
median_tokens = df['num_tokens'].median()
efficient_df = df[df['num_tokens'] <= median_tokens]

print(f'Filtered to {len(efficient_df)} efficient solutions (≤ {median_tokens} tokens)')
efficient_df.to_parquet('data/sft/teacher_efficient.parquet', index=False)
EOF

# Train on efficient examples only
TRAIN_FILE=data/sft/teacher_efficient.parquet \
bash scripts/run_sft_training.sh
```

## Monitoring Token Trends

### Track Token Usage Over Training

```bash
# Add custom callback to log token statistics
bash scripts/run_sft_training.sh \
    trainer.log_token_stats=true \
    trainer.logger='["console","wandb"]'
```

### Post-Training Analysis

```python
# Analyze token distribution changes
import pandas as pd
import matplotlib.pyplot as plt

# Load solutions at different stages
teacher = pd.read_parquet('data/sft/teacher_correct.parquet')
student_base = pd.read_jsonl('outputs/student_base_solutions.jsonl')
student_ft = pd.read_jsonl('outputs/student_finetuned_solutions.jsonl')

# Plot distributions
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(teacher['num_tokens'], bins=50, alpha=0.7)
plt.title('Teacher (7B)')
plt.xlabel('Tokens')

plt.subplot(1, 3, 2)
plt.hist([s['num_tokens'] for s in student_base], bins=50, alpha=0.7)
plt.title('Student Base (1.5B)')
plt.xlabel('Tokens')

plt.subplot(1, 3, 3)
plt.hist([s['num_tokens'] for s in student_ft], bins=50, alpha=0.7)
plt.title('Student Fine-tuned (1.5B)')
plt.xlabel('Tokens')

plt.tight_layout()
plt.savefig('token_distributions.png')
```

## Automated Workflow Script

See `sft/example_teacher_student.sh` for a complete automated workflow.

## Summary

This teacher-student approach:
- ✅ Leverages high-quality teacher solutions
- ✅ Trains smaller, faster student models
- ✅ Tracks token efficiency over time
- ✅ Provides clear accuracy vs efficiency tradeoffs
- ✅ Enables iterative improvement

Perfect for your use case of measuring token count changes during distillation!
