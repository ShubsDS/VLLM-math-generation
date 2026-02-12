#!/bin/bash
#SBATCH --job-name=vllm-sanity-check
#SBATCH --output=logs/sanity_check_%j.out
#SBATCH --error=logs/sanity_check_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Sanity check job - uses only 1 GPU for quick validation

echo "=========================================="
echo "SANITY CHECK JOB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set up environment
module purge
module load cuda/11.8

# Activate uv virtual environment
source .venv/bin/activate

# Verify environment
which python
python --version

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache
export HF_HOME=/path/to/huggingface/cache  # UPDATE THIS PATH
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Set UV torch backend
export UV_TORCH_BACKEND=auto

# Create output directory
mkdir -p outputs logs

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Math-14B-Instruct"

echo ""
echo "Running sanity check with model: $MODEL_NAME"
echo ""

# Run sanity check
python scripts/sanity_check.py \
    --model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-tokens 1024 \
    --temperature 0.0

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Sanity check PASSED!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "WARNING: Sanity check issues detected"
    echo "Review outputs/sanity_check_results.json"
    echo "=========================================="
fi

echo ""
echo "End Time: $(date)"
