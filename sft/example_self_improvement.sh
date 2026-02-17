#!/bin/bash
# Self-Improvement Workflow: Generate → Evaluate → Filter → Fine-tune
#
# This script demonstrates a complete self-improvement loop:
# 1. Generate solutions with a base model
# 2. Evaluate solutions against ground truth
# 3. Filter only correct solutions
# 4. Fine-tune the model on correct solutions
# 5. (Optionally) Repeat with the fine-tuned model

set -e

echo "=========================================="
echo "SELF-IMPROVEMENT WORKFLOW"
echo "=========================================="
echo "This example shows the complete pipeline:"
echo "  1. Generate solutions"
echo "  2. Evaluate solutions"
echo "  3. Save correct solutions"
echo "  4. Fine-tune on correct solutions"
echo "=========================================="
echo

# Configuration
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="./self_improvement_example"
SOLUTIONS_FILE="outputs/math_test_solutions_20260212_111413.jsonl"
METADATA_FILE="data/math_test_metadata.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR/data"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "Step 1: Evaluate existing solutions"
echo "--------------------"
python evaluation/run_evaluation.py \
    --solutions-file "$SOLUTIONS_FILE" \
    --metadata-file "$METADATA_FILE" \
    --output-dir "$OUTPUT_DIR/eval" \
    --save-correct "$OUTPUT_DIR/data/correct_solutions.parquet" \
    --model-name "$BASE_MODEL"

echo
echo "Step 2: Check how many correct solutions we have"
echo "--------------------"
python3 -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT_DIR/data/correct_solutions.parquet')
print(f'✓ Found {len(df)} correct solutions')
print(f'  Levels: {df[\"level\"].value_counts().to_dict()}')
print(f'  Types: {df[\"type\"].value_counts().to_dict()}')
print(f'\nSample solution:')
print(f'  Problem ID: {df[\"id\"].iloc[0]}')
print(f'  Level: {df[\"level\"].iloc[0]}')
print(f'  Type: {df[\"type\"].iloc[0]}')
print(f'  Tokens: {df[\"num_tokens\"].iloc[0]}')
"

echo
echo "Step 3: Prepare data for SFT (create train/val split)"
echo "--------------------"
python3 -c "
import pandas as pd
from pathlib import Path

df = pd.read_parquet('$OUTPUT_DIR/data/correct_solutions.parquet')

# 90/10 split
val_size = int(len(df) * 0.1)
train_df = df.iloc[val_size:]
val_df = df.iloc[:val_size]

# Save splits
train_df.to_parquet('$OUTPUT_DIR/data/train.parquet', index=False)
val_df.to_parquet('$OUTPUT_DIR/data/val.parquet', index=False)

print(f'✓ Created train/val split:')
print(f'  Train: {len(train_df)} examples')
print(f'  Val: {len(val_df)} examples')
"

echo
echo "Step 4: Fine-tune on correct solutions"
echo "--------------------"
echo "Running SFT training (this may take a while)..."
echo

NPROC_PER_NODE=1 \
MODEL_NAME="$BASE_MODEL" \
TRAIN_FILE="$OUTPUT_DIR/data/train.parquet" \
VAL_FILE="$OUTPUT_DIR/data/val.parquet" \
OUTPUT_DIR="$OUTPUT_DIR/checkpoints" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=1 \
PROJECT_NAME="self-improvement-example" \
bash scripts/run_sft_training.sh \
    trainer.logger=console \
    trainer.save_freq=500

echo
echo "=========================================="
echo "SELF-IMPROVEMENT COMPLETE!"
echo "=========================================="
echo "✓ Evaluated solutions: $SOLUTIONS_FILE"
echo "✓ Filtered to correct solutions: $OUTPUT_DIR/data/correct_solutions.parquet"
echo "✓ Fine-tuned model saved to: $OUTPUT_DIR/checkpoints/"
echo
echo "Next steps:"
echo "  1. Generate new solutions with fine-tuned model"
echo "  2. Evaluate again to see if accuracy improved"
echo "  3. Repeat the process for iterative improvement"
echo
echo "To use the fine-tuned model for inference:"
echo "  MODEL_PATH=\"$OUTPUT_DIR/checkpoints/global_step_final\""
echo "  python scripts/run_vllm_inference.py --model \$MODEL_PATH"
echo "=========================================="
