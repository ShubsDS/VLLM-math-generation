# VLLM MATH Dataset Inference

Distributed inference on the MATH dataset using VLLM with Qwen2.5-Math across 4 GPUs.

This project uses [uv](https://github.com/astral-sh/uv) for fast Python environment management, following VLLM's recommended setup.

## Project Structure

```
.
├── README.md                       # This file
├── pyproject.toml                 # Project dependencies (uv)
├── .python-version                # Python version (3.12)
├── .venv/                         # Virtual environment (created by uv)
├── data/                          # Dataset storage
│   ├── math_test_prompts.jsonl   # Generated prompts (after setup)
│   └── math_test_metadata.jsonl  # Problem metadata & ground truth
├── scripts/
│   ├── load_math_dataset.py      # Dataset preparation script
│   ├── run_vllm_inference.py     # Main inference script
│   ├── run_local_pipeline.py     # End-to-end local pipeline
│   ├── sanity_check.py           # Validation script
│   ├── convert_jsonl_to_parquet.py  # JSONL to Parquet converter
│   ├── run_sft_training.sh       # SFT training launcher
│   └── run_sft_peft.sh           # SFT PEFT training launcher
├── evaluation/                    # Evaluation pipeline
│   ├── README.md                 # Evaluation documentation
│   ├── run_evaluation.py         # Main evaluation script
│   ├── evaluator.py              # Core evaluation logic
│   ├── answer_matcher.py         # Answer extraction & matching
│   ├── metrics.py                # Metrics computation
│   └── generate_report.py        # Report generation
├── sft/                           # SFT training framework
│   ├── README.md                 # Full SFT documentation
│   ├── SFT_QUICKSTART.md         # Quick start guide
│   ├── prepare_data.py           # Data preparation script
│   └── example_train.sh          # End-to-end example
├── slurm/
│   ├── run_inference.slurm       # Full inference job
│   ├── run_sanity_check.slurm    # Quick validation job
│   ├── sft_training.slurm        # SFT training job
│   └── sft_peft.slurm            # SFT PEFT training job
├── configs/
│   └── inference_config.yaml     # Configuration reference
├── outputs/                       # Generated solutions
├── eval_results/                  # Evaluation reports
│   ├── reports/                  # JSON, CSV, Markdown reports
│   └── detailed/                 # Per-problem results
└── logs/                          # Slurm job logs
```

## Setup Instructions

### 1. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Verify installation
uv --version
```

### 2. Create Environment and Install Dependencies

Following VLLM's uv quickstart, create a Python 3.12 environment:

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12 --seed

# Activate environment
source .venv/bin/activate

# Install vllm with automatic torch backend detection
# This automatically selects the right PyTorch index based on your CUDA version
uv pip install vllm --torch-backend=auto

# Install other project dependencies
uv pip install -e .

# Verify installation
python -c "import vllm; import torch; print(f'VLLM: {vllm.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Note on torch backends:**
- `--torch-backend=auto` automatically detects your CUDA version
- For specific CUDA versions, use `--torch-backend=cu118`, `--torch-backend=cu121`, or `--torch-backend=cu126`
- Set `UV_TORCH_BACKEND` environment variable for persistent configuration

### 3. Configure HuggingFace Cache (Optional but Recommended)

To avoid re-downloading models, set up a shared cache directory:

```bash
# Edit slurm scripts and update this line:
export HF_HOME=/path/to/your/huggingface/cache

# Create the directory
mkdir -p /path/to/your/huggingface/cache
```

### 4. Prepare MATH Dataset

Download and prepare the MATH dataset:

```bash
# Activate environment first
source .venv/bin/activate

# Download full test set
python scripts/load_math_dataset.py \
    --output-dir data \
    --split test \
    --subset all

# Or for testing, use a small subset
python scripts/load_math_dataset.py \
    --output-dir data \
    --split test \
    --subset all \
    --max-samples 100
```

This creates:
- `data/math_test_prompts.jsonl` - Formatted prompts for the model
- `data/math_test_metadata.jsonl` - Problem metadata and ground truth solutions

## Running Inference

### Step 1: Sanity Check (Recommended)

Before running full inference, verify that the model is working correctly:

```bash
# Interactive sanity check (if you have GPU access on login node)
source .venv/bin/activate
python scripts/sanity_check.py \
    --model-name Qwen/Qwen2.5-Math-7B-Instruct \
    --tensor-parallel-size 1

# Or submit as a Slurm job
sbatch slurm/run_sanity_check.slurm
```

This will:
- Test the model on 3 sample MATH problems
- Verify correct answers are generated
- Save results to `outputs/sanity_check_results.json`

### Step 2: Full Inference

Submit the main inference job:

```bash
# Update slurm/run_inference.slurm if needed:
# - Adjust partition name
# - Update HF_HOME path
# - Modify resource requests (time, memory, etc.)

# Submit job
sbatch slurm/run_inference.slurm

# Monitor job
squeue -u $USER
tail -f logs/vllm_inference_<JOB_ID>.out
```

### Local End-to-End Run (No Slurm)

If you have direct GPU access and want a single command that prepares data and runs inference:

```bash
source .venv/bin/activate
python scripts/run_local_pipeline.py \
    --split test \
    --subset all \
    --model-name Qwen/Qwen2.5-Math-7B-Instruct \
    --tensor-parallel-size 1
```

The script writes:
- prompts/metadata to `data/`
- generated solutions to `outputs/`

### Step 3: Check Results

After the job completes:

```bash
# View output file
head outputs/math_solutions_<JOB_ID>.jsonl

# Count solutions generated
wc -l outputs/math_solutions_<JOB_ID>.jsonl
```

## Configuration Options

### Model Selection

The default model is `Qwen/Qwen2.5-Math-7B-Instruct`. To use a different model:

```bash
# In slurm script, change:
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"  # Smaller model
MODEL_NAME="Qwen/Qwen2.5-Math-72B-Instruct" # Larger model (requires more GPUs/memory)
```

### GPU Configuration

The setup uses 4 GPUs by default. To adjust:

```bash
# In slurm script:
#SBATCH --gres=gpu:4        # Change number of GPUs
TENSOR_PARALLEL_SIZE=4       # Must match --gres value
```

### Torch Backend Selection

The slurm scripts use `UV_TORCH_BACKEND=auto` for automatic CUDA detection. You can override this:

```bash
# For specific CUDA versions:
export UV_TORCH_BACKEND=cu118  # CUDA 11.8
export UV_TORCH_BACKEND=cu121  # CUDA 12.1
export UV_TORCH_BACKEND=cu126  # CUDA 12.6
```

### Sampling Parameters

For greedy decoding (deterministic, single solution per problem):
```bash
TEMPERATURE=0.0
```

For diverse solutions (multiple trajectories per problem):
```bash
TEMPERATURE=0.7    # Or 0.8, 1.0
# Run inference multiple times or adjust the script to generate N solutions per prompt
```

### Batch Size

Adjust based on GPU memory:

```bash
BATCH_SIZE=32   # Default
BATCH_SIZE=16   # If running out of memory
BATCH_SIZE=64   # If you have memory to spare
```

## Output Format

Generated solutions are saved in JSONL format:

```json
{
  "id": 0,
  "prompt": "Solve the following math problem...",
  "generated_solution": "Step 1: ...\nStep 2: ...\nFinal Answer: 42",
  "num_tokens": 256
}
```

Metadata file contains ground truth:

```json
{
  "id": 0,
  "problem": "If x + 2x + 3x + 4x + 5x = 90, what is the value of x?",
  "solution": "Combining like terms...",
  "level": "Level 1",
  "type": "algebra"
}
```

## Troubleshooting

### uv Installation Issues

If `uv` commands fail:
```bash
# Ensure uv is in your PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### CUDA/PyTorch Version Mismatch

If you get CUDA errors:
```bash
# Check your CUDA version
nvidia-smi

# Reinstall vllm with specific torch backend
uv pip uninstall vllm torch
uv pip install vllm --torch-backend=cu118  # Replace with your CUDA version
```

### Tokenizer AttributeError (`all_special_tokens_extended`)

If you see:
`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`

your environment likely has `transformers>=5`, which is currently incompatible with this vLLM setup.

Fix:
```bash
source .venv/bin/activate
uv pip install 'transformers>=4.55.2,<5' 'tokenizers>=0.21.1,<0.23'
```

### Numba / NumPy Compatibility Error

If you see:
`ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4`

fix by downgrading NumPy:
```bash
source .venv/bin/activate
uv pip install 'numpy>=1.24.0,<2.3'
```

### Out of Memory Errors

1. Reduce batch size: `BATCH_SIZE=16`
2. Reduce max model length in `run_vllm_inference.py`:
   ```python
   max_model_len=2048  # Instead of 4096
   ```
3. Reduce GPU memory utilization:
   ```python
   gpu_memory_utilization=0.85  # Instead of 0.95
   ```

### CUDA Out of Memory with Tensor Parallelism

If you get OOM errors even with tensor parallelism:
1. Use more GPUs (e.g., 8 instead of 4)
2. Switch to a smaller model variant
3. Reduce `max_model_len` parameter

### Model Download Issues

If HuggingFace downloads are slow or fail:
1. Pre-download the model:
   ```bash
   source .venv/bin/activate
   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
              AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct'); \
              AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct')"
   ```
2. Use local model path in scripts:
   ```bash
   MODEL_NAME="/path/to/local/model"
   ```

### Slurm Job Failures

Check the error log:
```bash
cat logs/vllm_inference_<JOB_ID>.err
```

Common issues:
- Wrong partition name: Update `#SBATCH --partition=gpu`
- Module not found: Adjust `module load` commands in slurm script
- Virtual environment not found: Ensure `.venv` exists in project root
- Python version mismatch: Recreate environment with `uv venv --python 3.12`

## Performance Notes

### Expected Throughput

With 4x A100 80GB GPUs:
- ~10-20 tokens/second per GPU
- ~1000-2000 problems/hour (depending on solution length)
- Full MATH test set (~5000 problems): 2-5 hours

With 4x V100 32GB GPUs:
- ~5-10 tokens/second per GPU
- ~500-1000 problems/hour
- Full MATH test set: 5-10 hours

### Optimization Tips

1. **Use bfloat16** (already configured) for better throughput on modern GPUs
2. **Increase batch size** to maximize GPU utilization
3. **Pre-download models** to avoid download time in job
4. **Use fast storage** for data/output directories (not NFS if possible)
5. **Use uv's caching** - uv automatically caches packages for faster reinstalls

## Environment Management with uv

### Useful uv Commands

```bash
# Add a new dependency
uv pip install <package>

# Update all dependencies
uv pip install --upgrade -e .

# List installed packages
uv pip list

# Freeze current environment
uv pip freeze > requirements.txt

# Create lock file (for reproducibility)
uv pip compile pyproject.toml -o requirements.lock

# Sync environment with lock file
uv pip sync requirements.lock
```

### Recreating the Environment

If you need to recreate from scratch:
```bash
# Remove existing environment
rm -rf .venv

# Create new environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Reinstall everything
uv pip install vllm --torch-backend=auto
uv pip install -e .
```

## Evaluation

### Evaluate Model Performance

After generating solutions, use the evaluation pipeline to assess accuracy:

```bash
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --model-name "Qwen/Qwen2.5-Math-7B-Instruct"
```

This will:
- Compute overall accuracy on the MATH dataset
- Break down accuracy by difficulty level (Level 1-5)
- Break down accuracy by problem type (Algebra, Geometry, etc.)
- Generate reports in JSON, CSV, and Markdown formats
- Save detailed per-problem results for error analysis

**Example output:**
```
================================================================================
EVALUATION SUMMARY
================================================================================
Model: Qwen/Qwen2.5-Math-7B-Instruct
Total Problems: 5000
Correct: 3913
Accuracy: 78.26%
================================================================================
```

Results are saved to `eval_results/` with comprehensive metrics and insights.

For detailed usage and options, see [evaluation/README.md](evaluation/README.md).

## Supervised Fine-Tuning (SFT)

This project includes a complete SFT framework powered by [verl](https://github.com/volcengine/verl) for fine-tuning models on your generated math solutions using FSDP (Fully Sharded Data Parallel).

### Quick Start

```bash
# 1. Prepare data from JSONL outputs
python sft/prepare_data.py "outputs/math_test_solutions_*.jsonl" -o data/sft

# 2. Run SFT training (single GPU, small model)
TRAIN_FILE=data/sft/train.parquet \
VAL_FILE=data/sft/val.parquet \
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
bash scripts/run_sft_training.sh

# 3. Or use LoRA/PEFT for larger models
TRAIN_FILE=data/sft/train.parquet \
VAL_FILE=data/sft/val.parquet \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
bash scripts/run_sft_peft.sh
```

### Features

- **FSDP Support**: Efficient multi-GPU training with Fully Sharded Data Parallel
- **LoRA/PEFT**: Parameter-efficient fine-tuning for large models
- **Automatic Data Conversion**: Convert JSONL → Parquet format
- **SLURM Integration**: Ready-to-use batch scripts for cluster training
- **Flexible Configuration**: Hydra-based config system with easy overrides
- **Monitoring**: Built-in support for W&B logging and console metrics

### Training Options

| Script | Use Case | Memory Requirement |
|--------|----------|-------------------|
| `run_sft_training.sh` | Full fine-tuning | High (models < 3B) |
| `run_sft_peft.sh` | LoRA/PEFT | Low (any model size) |

### Example: Multi-GPU Training

```bash
NPROC_PER_NODE=4 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
TRAIN_FILE=data/sft/train.parquet \
VAL_FILE=data/sft/val.parquet \
OUTPUT_DIR=./checkpoints/sft_llama_3b \
BATCH_SIZE=2 \
EPOCHS=3 \
bash scripts/run_sft_training.sh
```

### SLURM Cluster Usage

```bash
# Submit full fine-tuning job (4 GPUs)
sbatch slurm/sft_training.slurm

# Submit PEFT job (2 GPUs)
sbatch slurm/sft_peft.slurm

# Monitor progress
tail -f logs/sft_*.out
```

### Documentation

- **[SFT Quickstart](sft/SFT_QUICKSTART.md)** - Get started in 30 seconds
- **[Full SFT Guide](sft/README.md)** - Comprehensive documentation
- **Example Pipeline**: `sft/example_train.sh` - End-to-end example

For detailed configuration options, troubleshooting, and advanced usage, see the SFT documentation.

## Next Steps

After evaluating your model:

1. **Error analysis**: Review detailed results to identify problem types where model struggles
2. **Solution quality**: Analyze reasoning quality and correctness patterns
3. **Compare models**: Evaluate multiple models and compare performance
4. **Generate multiple trajectories**: Run with temperature > 0 multiple times for pass@k metrics

## References

- [VLLM Documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [MATH Dataset](https://github.com/hendrycks/math)
- [Qwen2.5-Math Models](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)

## Citation

If you use this code or the MATH dataset, please cite:

```bibtex
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora
          and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

## License

This project setup is provided as-is for research purposes. Please check the licenses for:
- VLLM: Apache 2.0
- MATH dataset: MIT License
- Qwen models: Check HuggingFace model card
- uv: MIT/Apache 2.0
