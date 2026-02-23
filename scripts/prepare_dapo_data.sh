#!/usr/bin/env bash
# Download DAPO-MATH-17K training data and AIME-2024 eval data to data/dapo/.
# Run from the repo root. Skips download if files already exist.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAPO_DIR="${REPO_ROOT}/data/dapo"
TRAIN_FILE="${DAPO_DIR}/dapo-math-17k.parquet"
TEST_FILE="${DAPO_DIR}/aime-2024.parquet"

mkdir -p "${DAPO_DIR}"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Downloading DAPO-MATH-17K training data..."
    wget -O "${TRAIN_FILE}" \
        "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
    echo "Saved: ${TRAIN_FILE}"
else
    echo "Already exists, skipping: ${TRAIN_FILE}"
fi

if [ ! -f "${TEST_FILE}" ]; then
    echo "Downloading AIME-2024 eval data..."
    wget -O "${TEST_FILE}" \
        "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
    echo "Saved: ${TEST_FILE}"
else
    echo "Already exists, skipping: ${TEST_FILE}"
fi

echo "Data ready in ${DAPO_DIR}/"
