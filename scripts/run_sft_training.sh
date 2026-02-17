#!/bin/bash
# SFT Training Script using verl with FSDP
# This script fine-tunes a model on math problem solutions

set -e  # Exit on error

# Default values
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MODEL_NAME=${MODEL_NAME:-"./Qwen2.5-Math-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"./data/sft/train.parquet"}
VAL_FILE=${VAL_FILE:-"./data/sft/val.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/sft_qwen2.5_math_1.5b"}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
EPOCHS=${EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-16384}
PROJECT_NAME=${PROJECT_NAME:-"math-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"math-sft-$(date +%Y%m%d_%H%M%S)"}

# Check if train file is provided
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: TRAIN_FILE must be set"
    echo "Usage: TRAIN_FILE=path/to/train.parquet VAL_FILE=path/to/val.parquet $0"
    exit 1
fi

# Check if val file is provided
if [ -z "$VAL_FILE" ]; then
    echo "Warning: VAL_FILE not set, using train file for validation"
    VAL_FILE=$TRAIN_FILE
fi

echo "=========================================="
echo "SFT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "GPUs: $NPROC_PER_NODE"
echo "=========================================="

# Run training
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=prompt \
    data.response_key=generated_solution \
    data.micro_batch_size_per_gpu=$BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    model.partial_pretrain=$MODEL_NAME \
    optim.lr=$LEARNING_RATE \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.logger='["console","wandb"]' \
    "$@"

echo "Training complete! Model saved to: $OUTPUT_DIR"
