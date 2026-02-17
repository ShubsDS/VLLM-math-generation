#!/bin/bash
# Example: End-to-end SFT training pipeline
# This script demonstrates the full workflow from JSONL to trained model

set -e

echo "======================================"
echo "SFT Training Example Pipeline"
echo "======================================"

# Step 1: Prepare data
echo -e "\n[Step 1/3] Preparing data..."
python sft/prepare_data.py \
    "outputs/math_test_solutions_*.jsonl" \
    --output-dir data/sft \
    --val-split 0.1 \
    --seed 42

# Step 2: Run training (small model for quick testing)
echo -e "\n[Step 2/3] Running SFT training..."
NPROC_PER_NODE=1 \
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_example" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=1 \
PROJECT_NAME="sft-example" \
bash scripts/run_sft_training.sh \
    trainer.logger=console \
    trainer.save_freq=100

echo -e "\n[Step 3/3] Training complete!"
echo "======================================"
echo "Model checkpoint saved to: ./checkpoints/sft_example"
echo "======================================"
