# SFT Training Framework with verl

This directory contains scripts for supervised fine-tuning (SFT) using the verl framework with FSDP (Fully Sharded Data Parallel) support.

## Overview

The framework allows you to fine-tune language models on your generated math solutions using:
- **FSDP**: Efficient distributed training across multiple GPUs
- **LoRA/PEFT**: Parameter-efficient fine-tuning option
- **Flexible data handling**: Automatic conversion from JSONL to Parquet format
- **Hydra configuration**: Easy-to-override hyperparameters

## Quick Start

### Option A: Train on All Generated Data

Convert JSONL files to Parquet format and split into train/val sets:

```bash
python sft/prepare_data.py outputs/math_test_solutions_*.jsonl \
    --output-dir data/sft \
    --val-split 0.1
```

This will create:
- `data/sft/train.parquet` (90% of data)
- `data/sft/val.parquet` (10% of data)

### Option B: Train Only on Correct Solutions (Recommended)

**NEW!** Filter to only correct solutions for higher quality training:

```bash
# 1. Evaluate and save only correct solutions
python evaluation/run_evaluation.py \
    --solutions-file outputs/math_test_solutions_*.jsonl \
    --metadata-file data/math_test_metadata.jsonl \
    --save-correct data/sft/correct_solutions.parquet

# 2. Split into train/val
python3 -c "
import pandas as pd
df = pd.read_parquet('data/sft/correct_solutions.parquet')
val_size = int(len(df) * 0.1)
df.iloc[val_size:].to_parquet('data/sft/train.parquet', index=False)
df.iloc[:val_size].to_parquet('data/sft/val.parquet', index=False)
print(f'Train: {len(df) - val_size}, Val: {val_size}')
"
```

**Why train only on correct solutions?**
- Higher quality training signal
- Avoid learning from incorrect reasoning
- Self-improvement through bootstrapping
- Better convergence and final accuracy

### 2. Run Full Fine-Tuning

Fine-tune the entire model:

```bash
# Single GPU
NPROC_PER_NODE=1 \
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen_1.5b" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
bash scripts/run_sft_training.sh

# Multi-GPU (4 GPUs)
NPROC_PER_NODE=4 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_llama_3b" \
BATCH_SIZE=2 \
bash scripts/run_sft_training.sh
```

### 3. Run LoRA Fine-Tuning (PEFT)

For larger models, use parameter-efficient fine-tuning:

```bash
NPROC_PER_NODE=2 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_peft_llama_3b" \
LORA_RANK=16 \
LORA_ALPHA=32 \
bash scripts/run_sft_peft.sh
```

## Directory Structure

```
sft/
├── README.md                    # This file
├── prepare_data.py              # Data preparation script
└── configs/                     # Configuration files (optional)

scripts/
├── convert_jsonl_to_parquet.py  # JSONL to Parquet converter
├── run_sft_training.sh          # Full fine-tuning script
└── run_sft_peft.sh              # LoRA/PEFT training script

data/sft/
├── train.parquet                # Training data (created by prepare_data.py)
└── val.parquet                  # Validation data (created by prepare_data.py)
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NPROC_PER_NODE` | Number of GPUs to use | 1 |
| `MODEL_NAME` | HuggingFace model name | Qwen/Qwen2.5-0.5B-Instruct |
| `TRAIN_FILE` | Path to training parquet file | Required |
| `VAL_FILE` | Path to validation parquet file | Same as TRAIN_FILE |
| `OUTPUT_DIR` | Output directory for checkpoints | ./sft_output |
| `BATCH_SIZE` | Micro batch size per GPU | 4 |
| `LEARNING_RATE` | Learning rate | 1e-5 |
| `EPOCHS` | Number of training epochs | 3 |
| `MAX_LENGTH` | Maximum sequence length | 2048 |
| `LORA_RANK` | LoRA rank (PEFT only) | 16 |
| `LORA_ALPHA` | LoRA alpha (PEFT only) | 32 |

### Additional Hydra Overrides

You can pass additional configuration via command line:

```bash
# Example: Enable sequence parallel and use Liger kernels
bash scripts/run_sft_training.sh \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
```

Common overrides:
- `data.max_length=4096` - Increase sequence length
- `optim.weight_decay=0.01` - Add weight decay
- `trainer.logger=console` - Disable wandb logging
- `trainer.save_freq=500` - Save checkpoint every 500 steps
- `model.use_flash_attention=true` - Enable flash attention

## Data Format

### Input JSONL Format

Your JSONL files should have the following structure:

```json
{
  "id": 0,
  "prompt": "Solve the following math problem...",
  "generated_solution": "Step 1: ...",
  "num_tokens": 2048
}
```

Required fields:
- `prompt`: The input question/problem
- `generated_solution`: The target response/solution

Optional fields (ignored during training):
- `id`: Example identifier
- `num_tokens`: Token count
- Any other metadata

### Parquet Format

The conversion scripts automatically create parquet files with the required format for verl's SFTDataset.

## Training Output

After training, your output directory will contain:

```
checkpoints/sft_qwen_1.5b/
├── global_step_0/              # Initial checkpoint
├── global_step_500/            # Intermediate checkpoints
├── global_step_1000/
└── global_step_final/          # Final checkpoint
```

Each checkpoint directory contains:
- `model.pt` - Model weights
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Scheduler state
- `train_state.pt` - Training state (for resuming)

## Advanced Usage

### Resume from Checkpoint

To resume training from a checkpoint:

```bash
bash scripts/run_sft_training.sh \
    trainer.resume_from_checkpoint=./checkpoints/sft_qwen_1.5b/global_step_500
```

### Multi-Node Training

For training across multiple nodes, use:

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    [config options...]
```

### Custom Data Keys

If your JSONL uses different field names:

```bash
python sft/prepare_data.py outputs/custom_data.jsonl \
    --output-dir data/custom \
    --prompt-key "question" \
    --response-key "answer"
```

Then update the training script accordingly:

```bash
bash scripts/run_sft_training.sh \
    data.prompt_key=question \
    data.response_key=answer
```

## Monitoring

### Weights & Biases

The scripts are configured to log to wandb by default. Make sure you have wandb installed and configured:

```bash
pip install wandb
wandb login
```

To disable wandb logging:

```bash
bash scripts/run_sft_training.sh trainer.logger=console
```

### Console Logging

Training metrics will be printed to console, including:
- Training loss
- Learning rate
- Throughput (tokens/sec)
- GPU memory usage

## Troubleshooting

### Out of Memory

1. Reduce batch size: `BATCH_SIZE=2` or `BATCH_SIZE=1`
2. Reduce sequence length: Add `data.max_length=1024`
3. Use LoRA/PEFT: `bash scripts/run_sft_peft.sh`
4. Enable gradient checkpointing: Add `model.gradient_checkpointing=true`

### Slow Training

1. Increase batch size if memory allows
2. Use multiple GPUs: `NPROC_PER_NODE=4`
3. Enable flash attention: Add `model.use_flash_attention=true`

### Data Issues

If you get errors about missing keys, verify your data format:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/sft/train.parquet')
print(df.columns)
print(df.head(1))
"
```

## References

- [verl Documentation](https://github.com/volcengine/verl)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
