#!/bin/bash
#SBATCH --job-name=sft-training
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=128G

# Supervised Fine-Tuning with FSDP on SLURM
# Submit with: sbatch slurm/sft_training.slurm

set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.10

# Activate virtual environment
source .venv/bin/activate

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
TRAIN_FILE="data/sft/train.parquet"
VAL_FILE="data/sft/val.parquet"
OUTPUT_DIR="./checkpoints/sft_llama_3b_${SLURM_JOB_ID}"
NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
BATCH_SIZE=2
LEARNING_RATE=1e-5
EPOCHS=3
MAX_LENGTH=2048

echo "Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $NPROC_PER_NODE"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "=========================================="

# Run training
NPROC_PER_NODE=$NPROC_PER_NODE \
MODEL_NAME=$MODEL_NAME \
TRAIN_FILE=$TRAIN_FILE \
VAL_FILE=$VAL_FILE \
OUTPUT_DIR=$OUTPUT_DIR \
BATCH_SIZE=$BATCH_SIZE \
LEARNING_RATE=$LEARNING_RATE \
EPOCHS=$EPOCHS \
MAX_LENGTH=$MAX_LENGTH \
bash scripts/run_sft_training.sh

echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
