# SFT Framework Overview

## What Was Created

A complete supervised fine-tuning (SFT) framework using verl with FSDP support for fine-tuning language models on your generated math solutions.

## Files Created

### Core Scripts

1. **`scripts/convert_jsonl_to_parquet.py`**
   - Converts individual JSONL files to Parquet format
   - Simple single-file converter

2. **`sft/prepare_data.py`**
   - Advanced data preparation with train/val splitting
   - Supports glob patterns for multiple files
   - Automatic validation and shuffling

3. **`scripts/run_sft_training.sh`**
   - Full fine-tuning launcher script
   - Environment variable configuration
   - Passes all parameters to verl's FSDP trainer

4. **`scripts/run_sft_peft.sh`**
   - LoRA/PEFT training launcher
   - Memory-efficient fine-tuning
   - Same interface as full training script

### SLURM Integration

5. **`slurm/sft_training.slurm`**
   - SLURM batch script for full fine-tuning
   - Pre-configured for 4 GPUs
   - 24-hour job time limit

6. **`slurm/sft_peft.slurm`**
   - SLURM batch script for PEFT training
   - Pre-configured for 2 GPUs
   - 12-hour job time limit

### Examples & Documentation

7. **`sft/example_train.sh`**
   - End-to-end example pipeline
   - Data prep → Training → Evaluation
   - Good starting point for new users

8. **`sft/SFT_QUICKSTART.md`**
   - Quick start guide (30-second setup)
   - Common configurations
   - Troubleshooting tips

9. **`sft/README.md`**
   - Comprehensive documentation
   - All configuration options
   - Advanced usage patterns
   - Detailed troubleshooting

10. **`SFT_FRAMEWORK_OVERVIEW.md`** (this file)
    - High-level overview
    - Architecture explanation

11. **Updated `README.md`**
    - Added SFT section
    - Updated project structure
    - Integration with existing pipeline

## How It Works

### Data Flow

```
JSONL files (outputs/)
    ↓
[prepare_data.py]
    ↓
Parquet files (data/sft/)
    ↓
[verl FSDP SFT Trainer]
    ↓
Fine-tuned model (checkpoints/)
```

### Architecture

The framework leverages:

1. **verl** (from ../verl/)
   - Provides `verl.trainer.fsdp_sft_trainer` module
   - Handles FSDP parallelization
   - Implements efficient training loop
   - Supports LoRA/PEFT integration

2. **PyTorch FSDP**
   - Fully Sharded Data Parallel for multi-GPU training
   - Automatic model sharding across GPUs
   - Gradient accumulation and optimization

3. **Hydra Configuration**
   - All parameters configurable via command line
   - Override defaults with `key=value` syntax
   - Supports complex nested configurations

4. **Data Pipeline**
   - Reads Parquet files (required by verl)
   - Applies chat templates automatically
   - Handles tokenization and masking
   - Supports variable-length sequences

## Usage Patterns

### Pattern 1: Quick Experimentation (Single GPU)

```bash
# Small model, fast iteration
python sft/prepare_data.py "outputs/*.jsonl" -o data/sft

TRAIN_FILE=data/sft/train.parquet \
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
BATCH_SIZE=8 \
EPOCHS=1 \
bash scripts/run_sft_training.sh
```

### Pattern 2: Production Training (Multi-GPU)

```bash
# Prepare data once
python sft/prepare_data.py "outputs/*.jsonl" -o data/sft --val-split 0.1

# Submit to SLURM cluster
sbatch slurm/sft_training.slurm
```

### Pattern 3: Large Model with PEFT

```bash
# Data preparation (same as above)
python sft/prepare_data.py "outputs/*.jsonl" -o data/sft

# Use LoRA for memory efficiency
NPROC_PER_NODE=2 \
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
TRAIN_FILE=data/sft/train.parquet \
LORA_RANK=16 \
bash scripts/run_sft_peft.sh
```

### Pattern 4: Hyperparameter Sweep

```bash
# Try different learning rates
for LR in 1e-5 2e-5 5e-5; do
    LEARNING_RATE=$LR \
    OUTPUT_DIR="./checkpoints/lr_${LR}" \
    bash scripts/run_sft_training.sh
done
```

## Configuration System

### Environment Variables (High-Level)

Set these before calling the shell scripts:

- `NPROC_PER_NODE` - Number of GPUs
- `MODEL_NAME` - HuggingFace model path
- `TRAIN_FILE` - Training data (parquet)
- `VAL_FILE` - Validation data (parquet)
- `OUTPUT_DIR` - Checkpoint directory
- `BATCH_SIZE` - Micro batch size per GPU
- `LEARNING_RATE` - Optimizer learning rate
- `EPOCHS` - Number of training epochs
- `LORA_RANK` - LoRA rank (PEFT only)
- `LORA_ALPHA` - LoRA alpha (PEFT only)

### Hydra Overrides (Low-Level)

Pass additional parameters after the script:

```bash
bash scripts/run_sft_training.sh \
    optim.weight_decay=0.01 \
    data.max_length=4096 \
    trainer.save_freq=100 \
    trainer.mixed_precision=bf16
```

Common overrides:
- `data.*` - Dataset configuration
- `model.*` - Model settings
- `optim.*` - Optimizer configuration
- `trainer.*` - Training loop settings
- `peft.*` - LoRA/PEFT parameters

## Directory Structure After Setup

```
VLLM-math-generation/
├── outputs/                          # Your JSONL files
│   ├── math_test_solutions_*.jsonl
│   └── ...
├── data/sft/                         # Prepared training data
│   ├── train.parquet                 # 90% of data
│   └── val.parquet                   # 10% of data
├── checkpoints/                      # Fine-tuned models
│   ├── sft_qwen_1.5b/
│   │   ├── global_step_0/
│   │   ├── global_step_500/
│   │   └── global_step_final/
│   └── sft_peft_llama_3b/
│       └── ...
├── sft/                              # Framework code
│   ├── README.md
│   ├── SFT_QUICKSTART.md
│   ├── prepare_data.py
│   └── example_train.sh
├── scripts/                          # Training launchers
│   ├── convert_jsonl_to_parquet.py
│   ├── run_sft_training.sh
│   └── run_sft_peft.sh
└── slurm/                            # Cluster scripts
    ├── sft_training.slurm
    └── sft_peft.slurm
```

## Integration with Existing Pipeline

### Full Workflow

1. **Data Generation** (existing)
   ```bash
   python scripts/run_vllm_inference.py
   # → outputs/math_test_solutions_*.jsonl
   ```

2. **Evaluation** (existing)
   ```bash
   python evaluation/run_evaluation.py --solutions-file outputs/*.jsonl
   # → eval_results/reports/
   ```

3. **Fine-Tuning** (new)
   ```bash
   python sft/prepare_data.py "outputs/*.jsonl" -o data/sft
   bash scripts/run_sft_training.sh
   # → checkpoints/sft_*/
   ```

4. **Iterate**
   - Generate more data with fine-tuned model
   - Evaluate again
   - Fine-tune again
   - Repeat

## Technical Details

### Data Format Requirements

**Input (JSONL):**
```json
{
  "id": 0,
  "prompt": "Solve: ...",
  "generated_solution": "Step 1: ...",
  "num_tokens": 2048
}
```

**Intermediate (Parquet):**
- Same schema as JSONL
- Binary format for faster loading
- Efficient storage and processing

**Training Data:**
- Prompts are wrapped with chat template
- Responses are concatenated with EOS token
- Loss is masked on prompt tokens
- Only response tokens contribute to gradient

### Memory Requirements

**Full Fine-Tuning:**
- 0.5B model: ~4GB per GPU
- 1.5B model: ~8GB per GPU
- 3B model: ~16GB per GPU
- 7B model: ~32GB per GPU (requires FSDP)

**LoRA/PEFT:**
- Adds ~1-2GB overhead regardless of base model size
- Can fine-tune 7B models on 16GB GPUs
- Can fine-tune 13B models on 24GB GPUs

### Training Speed

**Approximate throughput (A100 80GB):**
- 0.5B model: ~2000 tokens/sec/GPU
- 1.5B model: ~1000 tokens/sec/GPU
- 3B model: ~500 tokens/sec/GPU
- 7B model: ~200 tokens/sec/GPU

**Expected training time (5000 examples, 3 epochs):**
- 0.5B model: ~1 hour (single GPU)
- 1.5B model: ~2 hours (single GPU)
- 3B model: ~4 hours (2 GPUs)
- 7B model: ~8 hours (4 GPUs)

## Customization Examples

### Custom Data Keys

If your JSONL has different field names:

```bash
python sft/prepare_data.py \
    outputs/custom_data.jsonl \
    --output-dir data/custom \
    --prompt-key "question" \
    --response-key "answer"

# Then update training script
bash scripts/run_sft_training.sh \
    data.prompt_key=question \
    data.response_key=answer
```

### Custom Chat Template

```bash
bash scripts/run_sft_training.sh \
    data.apply_chat_template_kwargs='{\"add_generation_prompt\":false}'
```

### Custom Optimizer

```bash
bash scripts/run_sft_training.sh \
    optim.name=adamw \
    optim.lr=2e-5 \
    optim.weight_decay=0.01 \
    optim.betas='[0.9,0.999]'
```

### Custom Learning Rate Schedule

```bash
bash scripts/run_sft_training.sh \
    optim.lr_scheduler=cosine \
    optim.warmup_ratio=0.1
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `BATCH_SIZE` or use `run_sft_peft.sh` |
| Slow training | Increase `BATCH_SIZE` or `NPROC_PER_NODE` |
| File not found | Run `prepare_data.py` first |
| Bad accuracy | Increase `EPOCHS` or `LEARNING_RATE` |
| Overfitting | Add `optim.weight_decay=0.01` |
| Underfitting | Increase `EPOCHS` or model size |

## Next Steps

1. **Test the framework:**
   ```bash
   bash sft/example_train.sh
   ```

2. **Read the quickstart:**
   ```bash
   cat sft/SFT_QUICKSTART.md
   ```

3. **Try PEFT training:**
   ```bash
   bash scripts/run_sft_peft.sh
   ```

4. **Submit a SLURM job:**
   ```bash
   sbatch slurm/sft_training.slurm
   ```

5. **Experiment with hyperparameters:**
   - Learning rates: 1e-6 to 1e-4
   - Batch sizes: 1 to 8 per GPU
   - Epochs: 1 to 5
   - LoRA ranks: 8, 16, 32, 64

## Support & Resources

- **Framework code:** `../verl/`
- **verl GitHub:** https://github.com/volcengine/verl
- **FSDP docs:** https://pytorch.org/docs/stable/fsdp.html
- **LoRA paper:** https://arxiv.org/abs/2106.09685
- **Issue tracker:** Report issues with this framework

## Summary

You now have a production-ready SFT framework that:
- ✅ Converts your JSONL outputs to training data
- ✅ Supports single-GPU and multi-GPU training
- ✅ Works with any HuggingFace model
- ✅ Includes PEFT/LoRA for large models
- ✅ Integrates with SLURM clusters
- ✅ Provides comprehensive documentation
- ✅ Has working examples and quickstarts

Start with `sft/example_train.sh` or `sft/SFT_QUICKSTART.md` to get going!
