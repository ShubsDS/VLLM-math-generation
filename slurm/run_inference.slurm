#!/bin/bash
#SBATCH --job-name=vllm-math-inference
#SBATCH --output=logs/vllm_inference_%j.out
#SBATCH --error=logs/vllm_inference_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

# Job configuration
# Adjust the partition, time, and resource requests based on your cluster setup

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Set up environment
module purge
module load cuda/11.8  # Adjust CUDA version as needed
# module load cudnn/8.9  # Uncomment if needed

# Activate uv virtual environment
source .venv/bin/activate

# Verify uv environment is active
which python
python --version

# Set CUDA devices (for 4 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set HuggingFace cache directory (optional, to avoid re-downloading)
export HF_HOME=/path/to/huggingface/cache  # UPDATE THIS PATH
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Set UV torch backend for automatic CUDA version detection
export UV_TORCH_BACKEND=auto

# Create necessary directories
mkdir -p logs outputs data

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Math-14B-Instruct"
PROMPTS_FILE="data/math_test_prompts.jsonl"
OUTPUT_FILE="outputs/math_solutions_${SLURM_JOB_ID}.jsonl"
TENSOR_PARALLEL_SIZE=4
MAX_TOKENS=2048
TEMPERATURE=0.0
BATCH_SIZE=32

echo ""
echo "=========================================="
echo "Configuration:"
echo "Model: $MODEL_NAME"
echo "Prompts: $PROMPTS_FILE"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $TENSOR_PARALLEL_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Run inference
echo "Starting VLLM inference..."
python scripts/run_vllm_inference.py \
    --model-name "$MODEL_NAME" \
    --prompts-file "$PROMPTS_FILE" \
    --output-file "$OUTPUT_FILE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Inference completed successfully!"
    echo "Output saved to: $OUTPUT_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Inference failed!"
    echo "=========================================="
    exit 1
fi

# Print job statistics
echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
