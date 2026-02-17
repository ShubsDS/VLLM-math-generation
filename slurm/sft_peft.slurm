#!/bin/bash
#SBATCH --job-name=sft-peft
#SBATCH --output=logs/sft_peft_%j.out
#SBATCH --error=logs/sft_peft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# PEFT (LoRA) Fine-Tuning with FSDP on SLURM
# Submit with: sbatch slurm/sft_peft.slurm

set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
TRAIN_FILE="data/sft/train.parquet"
VAL_FILE="data/sft/val.parquet"
OUTPUT_DIR="./checkpoints/sft_peft_llama_3b_${SLURM_JOB_ID}"
NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
BATCH_SIZE=4
LEARNING_RATE=1e-4
EPOCHS=3
LORA_RANK=16
LORA_ALPHA=32

echo "PEFT Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $NPROC_PER_NODE"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA alpha: $LORA_ALPHA"
echo "=========================================="

# Run PEFT training
NPROC_PER_NODE=$NPROC_PER_NODE \
MODEL_NAME=$MODEL_NAME \
TRAIN_FILE=$TRAIN_FILE \
VAL_FILE=$VAL_FILE \
OUTPUT_DIR=$OUTPUT_DIR \
BATCH_SIZE=$BATCH_SIZE \
LEARNING_RATE=$LEARNING_RATE \
EPOCHS=$EPOCHS \
LORA_RANK=$LORA_RANK \
LORA_ALPHA=$LORA_ALPHA \
bash scripts/run_sft_peft.sh

echo "=========================================="
echo "PEFT training complete!"
echo "End time: $(date)"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
