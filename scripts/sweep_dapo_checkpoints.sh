#!/usr/bin/env bash
# Run DAPO from multiple SFT checkpoints sequentially.
#
# Usage:
#   bash scripts/sweep_dapo_checkpoints.sh 1 5 10
#   WANDB_PROJECT=my-project bash scripts/sweep_dapo_checkpoints.sh 1 3 5 7 10
#
# Each positional argument is an epoch number N corresponding to:
#   /bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b/qwen2_5_math_1_5b_N/huggingface
#
# Optional env vars forwarded to run_dapo_from_checkpoint.sh:
#   WANDB_PROJECT, NPROC_PER_NODE, TOTAL_STEPS, TRAIN_FILE, TEST_FILE, OUTPUT_BASE_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CKPT_DIR="/bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b"

if [ "$#" -eq 0 ]; then
    echo "Usage: bash $(basename "$0") <epoch1> [epoch2] ..." >&2
    echo "Example: bash $(basename "$0") 1 5 10" >&2
    exit 1
fi

echo "=========================================="
echo "DAPO Checkpoint Sweep"
echo "Base dir: ${BASE_CKPT_DIR}"
echo "Epochs:   $*"
echo "W&B project: ${WANDB_PROJECT:-dapo-sft-sweep}"
echo "=========================================="

for epoch in "$@"; do
    CHECKPOINT_PATH="${BASE_CKPT_DIR}/qwen2_5_math_1_5b_${epoch}/huggingface"
    CHECKPOINT_NAME="qwen2_5_math_1_5b_${epoch}"

    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "Error: checkpoint not found for epoch ${epoch}: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi

    echo ""
    echo "-- Starting DAPO from epoch ${epoch}: ${CHECKPOINT_PATH} --"

    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
        bash "${SCRIPT_DIR}/run_dapo_from_checkpoint.sh"

    echo "-- Finished epoch ${epoch} --"

    # Stop Ray to release GPU memory before the next checkpoint run.
    ray stop --force 2>/dev/null || true
    sleep 5
done

echo ""
echo "Sweep complete. Ran DAPO for epochs: $*"
