# SFT Training Quickstart

This guide will help you get started with supervised fine-tuning (SFT) using your generated math solutions.

## Prerequisites

- verl framework installed (located at `../verl`)
- PyTorch with CUDA support
- At least one GPU
- Generated JSONL files in `outputs/` directory

## 30-Second Quickstart

```bash
# 1. Prepare data
python sft/prepare_data.py "outputs/math_test_solutions_*.jsonl" -o data/sft

# 2. Train (single GPU, small model)
TRAIN_FILE=data/sft/train.parquet VAL_FILE=data/sft/val.parquet \
  bash scripts/run_sft_training.sh

# 3. Check output
ls -lh ./sft_output/
```

## Step-by-Step Guide

### Step 1: Prepare Your Data

The framework expects Parquet files. Convert your JSONL outputs:

```bash
# Option A: Use the all-in-one preparation script
python sft/prepare_data.py \
    "outputs/math_test_solutions_*.jsonl" \
    --output-dir data/sft \
    --val-split 0.1

# Option B: Manual conversion (single file)
python scripts/convert_jsonl_to_parquet.py \
    outputs/math_test_solutions_20260212_111413.jsonl \
    -o data/train.parquet
```

**Output:**
- `data/sft/train.parquet` - Training data (90%)
- `data/sft/val.parquet` - Validation data (10%)

### Step 2: Choose Your Training Mode

#### Option A: Full Fine-Tuning (Smaller Models)

Best for models < 3B parameters:

```bash
NPROC_PER_NODE=1 \
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen_1.5b" \
EPOCHS=3 \
bash scripts/run_sft_training.sh
```

#### Option B: LoRA/PEFT (Larger Models)

Best for models > 3B parameters or limited GPU memory:

```bash
NPROC_PER_NODE=1 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_peft_llama" \
bash scripts/run_sft_peft.sh
```

#### Option C: Multi-GPU Training

For faster training or larger models:

```bash
NPROC_PER_NODE=4 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_llama_4gpu" \
BATCH_SIZE=2 \
bash scripts/run_sft_training.sh
```

### Step 3: Monitor Training

Training logs will show:
```
Epoch 1/3 - Step 100/1000 - Loss: 2.314 - LR: 1.0e-05 - Tokens/sec: 4523
Epoch 1/3 - Step 200/1000 - Loss: 1.987 - LR: 9.8e-06 - Tokens/sec: 4618
...
```

Checkpoints are saved to `OUTPUT_DIR/`:
```
checkpoints/sft_qwen_1.5b/
├── global_step_0/
├── global_step_500/
└── global_step_final/
```

### Step 4: Use Your Fine-Tuned Model

Load the checkpoint for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./checkpoints/sft_qwen_1.5b/global_step_final"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate
prompt = "Solve: What is 2 + 2?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Common Configurations

### Small Model, Single GPU
```bash
NPROC_PER_NODE=1 \
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
BATCH_SIZE=8 \
TRAIN_FILE=data/sft/train.parquet \
bash scripts/run_sft_training.sh
```

### Medium Model, 2 GPUs with LoRA
```bash
NPROC_PER_NODE=2 \
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
BATCH_SIZE=4 \
LORA_RANK=16 \
TRAIN_FILE=data/sft/train.parquet \
bash scripts/run_sft_peft.sh
```

### Large Model, 4+ GPUs with LoRA
```bash
NPROC_PER_NODE=4 \
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
BATCH_SIZE=2 \
LORA_RANK=32 \
LEARNING_RATE=2e-5 \
TRAIN_FILE=data/sft/train.parquet \
bash scripts/run_sft_peft.sh
```

## SLURM Cluster

Submit to SLURM job scheduler:

```bash
# Full fine-tuning (4 GPUs)
sbatch slurm/sft_training.slurm

# PEFT training (2 GPUs)
sbatch slurm/sft_peft.slurm
```

Check job status:
```bash
squeue -u $USER
tail -f logs/sft_*.out
```

## Troubleshooting

### "CUDA out of memory"
1. Reduce batch size: `BATCH_SIZE=1`
2. Reduce sequence length: Add `data.max_length=1024` to script
3. Use LoRA: Switch to `run_sft_peft.sh`
4. Use more GPUs: Increase `NPROC_PER_NODE`

### "RuntimeError: No CUDA GPUs are available"
```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch can see GPUs
python -c "import torch; print(torch.cuda.is_available())"
```

### "FileNotFoundError: data/sft/train.parquet"
Make sure you ran data preparation:
```bash
python sft/prepare_data.py "outputs/*.jsonl" -o data/sft
```

### Training is slow
1. Increase batch size if memory allows
2. Use more GPUs
3. Enable flash attention: Add `model.use_flash_attention=true`
4. Check if you're using GPU: Look for "cuda" in logs

## Advanced Options

### Custom Hyperparameters

Override any config parameter:

```bash
bash scripts/run_sft_training.sh \
    optim.lr=2e-5 \
    optim.weight_decay=0.01 \
    data.max_length=4096 \
    trainer.save_freq=100 \
    trainer.logger=console
```

### Resume from Checkpoint

```bash
bash scripts/run_sft_training.sh \
    trainer.resume_from_checkpoint=./checkpoints/sft_qwen/global_step_500
```

### Gradient Accumulation

Effectively increase batch size:

```bash
bash scripts/run_sft_training.sh \
    data.micro_batch_size_per_gpu=2 \
    trainer.gradient_accumulation_steps=4
```

### Mixed Precision Training

Enable bfloat16 for faster training:

```bash
bash scripts/run_sft_training.sh \
    trainer.mixed_precision=bf16
```

## Next Steps

1. **Evaluate your model**: Use the evaluation scripts in `evaluation/`
2. **Try different models**: Experiment with Llama, Qwen, Mistral, etc.
3. **Tune hyperparameters**: Adjust learning rate, batch size, epochs
4. **Scale up**: Use multiple GPUs or nodes for larger models
5. **Iterate**: Generate more data, fine-tune again, evaluate

## Resources

- [Full SFT Documentation](sft/README.md)
- [verl GitHub](https://github.com/volcengine/verl)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
