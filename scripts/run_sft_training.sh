#!/bin/bash
# Minimal SFT training script (verl FSDP), checkpoint every epoch as HF format.

set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MODEL_NAME=${MODEL_NAME:-"./Qwen2.5-Math-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"./data/sft/train.parquet"}
VAL_FILE=${VAL_FILE:-"./data/sft/val.parquet"}
MODEL_SLUG=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/sft_${MODEL_SLUG}"}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-10}
EPOCHS=${EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-16384}
PROJECT_NAME=${PROJECT_NAME:-"math-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"math-sft-$(date +%Y%m%d_%H%M%S)"}
ATTN_IMPL=${ATTN_IMPL:-"sdpa"}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
WANDB_ENABLED=${WANDB_ENABLED:-false}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS))}

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: train file does not exist: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: val file does not exist: $VAL_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

STEPS_PER_EPOCH=$(TRAIN_FILE="$TRAIN_FILE" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" uv run python3 - << 'PY'
import math, os
import pandas as pd
rows = len(pd.read_parquet(os.environ["TRAIN_FILE"]))
batch = int(os.environ["TRAIN_BATCH_SIZE"])
print(max(1, math.ceil(rows / batch)))
PY
)

SAVE_FREQ=$STEPS_PER_EPOCH

if [ "$WANDB_ENABLED" = "true" ]; then
    LOGGER_LIST='["console","wandb"]'
else
    LOGGER_LIST='["console"]'
fi

echo "=========================================="
echo "Minimal SFT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Micro batch size per GPU: $BATCH_SIZE"
echo "Global train batch size: $TRAIN_BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "GPUs: $NPROC_PER_NODE"
echo "Attention impl: $ATTN_IMPL"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Steps per epoch: $STEPS_PER_EPOCH"
echo "Checkpoint save frequency (steps): $SAVE_FREQ"
echo "Checkpoint contents: [\"hf_model\"]"
echo "Warmup steps: $WARMUP_STEPS"
echo "Loggers: $LOGGER_LIST"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.prompt_key=prompt \
    data.response_key=generated_solution \
    data.micro_batch_size_per_gpu=$BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    model.partial_pretrain=$MODEL_NAME \
    model.attn_implementation=$ATTN_IMPL \
    optim.lr=$LEARNING_RATE \
    optim.warmup_steps=$WARMUP_STEPS \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    'trainer.checkpoint.save_contents=["hf_model"]' \
    "trainer.logger=$LOGGER_LIST" \
    "$@"

echo "Training complete. Checkpoints saved to: $OUTPUT_DIR"
