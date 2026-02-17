# SFT Setup Guide for Qwen2.5 Math Model

This guide provides step-by-step instructions for performing supervised fine-tuning (SFT) on your locally downloaded Qwen2.5 math model.

## Prerequisites

1. **Local Model**: Ensure you have a Qwen2.5 math model downloaded locally
2. **Training Data**: JSONL files with `prompt` and `generated_solution` fields
3. **Dependencies**: verl framework and required packages installed

## Step 1: Prepare Your Training Data

### 1.1 Check Your Data Format

Your JSONL files should have the following structure:
```json
{
  "id": 0,
  "prompt": "Solve the following math problem...",
  "generated_solution": "Step 1: ...",
  "num_tokens": 2048
}
```

Verify your data:
```bash
# Check if you have JSONL files
ls -lh outputs/*.jsonl

# Inspect a sample entry
head -1 outputs/math_test_solutions_*.jsonl | python -m json.tool
```

### 1.2 Convert JSONL to Parquet Format

The verl trainer requires Parquet format. Convert your JSONL files:

```bash
# Convert all JSONL files to Parquet with train/val split
python sft/prepare_data.py "outputs/*.jsonl" \
    --output-dir data/sft \
    --val-split 0.1 \
    --seed 42
```

This creates:
- `data/sft/train.parquet` (90% of data)
- `data/sft/val.parquet` (10% of data)

### 1.3 Verify Data Preparation

```bash
# Check the prepared data
python -c "
import pandas as pd
train_df = pd.read_parquet('data/sft/train.parquet')
val_df = pd.read_parquet('data/sft/val.parquet')
print(f'Train examples: {len(train_df)}')
print(f'Val examples: {len(val_df)}')
print(f'\nColumns: {list(train_df.columns)}')
print(f'\nSample prompt: {train_df[\"prompt\"].iloc[0][:200]}...')
print(f'\nSample response: {train_df[\"generated_solution\"].iloc[0][:200]}...')
"
```

## Step 2: Locate Your Local Model

### 2.1 Find Your Model Path

Your Qwen2.5 math model should be in one of these locations:
- Local directory: `/path/to/qwen2.5-math-model/`
- HuggingFace cache: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-*/`

Find it:
```bash
# Option 1: If you know the directory name
find ~ -type d -name "*qwen*math*" 2>/dev/null | head -5

# Option 2: Check HuggingFace cache
ls -la ~/.cache/huggingface/hub/ | grep -i qwen

# Option 3: If model is in current directory
ls -la | grep -i qwen
```

### 2.2 Verify Model Structure

The model directory should contain:
- `config.json`
- `tokenizer.json` or `tokenizer_config.json`
- Model weights (`.safetensors` or `.bin` files)

```bash
# Check model directory (replace with your path)
MODEL_PATH="/path/to/your/qwen2.5-math-model"
ls -lh $MODEL_PATH
```

## Step 3: Run SFT Training

### 3.1 Basic Training Command

```bash
# Set your model path (use absolute path or relative path)
MODEL_NAME="/path/to/your/qwen2.5-math-model"  # or "Qwen/Qwen2.5-Math-7B-Instruct" if using HF

# Run training with response length logging
NPROC_PER_NODE=1 \
MODEL_NAME="$MODEL_NAME" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
MAX_LENGTH=2048 \
PROJECT_NAME="qwen2.5-math-sft" \
EXPERIMENT_NAME="qwen2.5-math-$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_sft_training.sh \
    trainer.logger='["console"]' \
    trainer.save_freq=500
```

### 3.2 Training with Response Length Logging

Use the custom training script that logs average response length per batch:

```bash
# Use the enhanced training script with response length logging
# This script automatically:
# 1. Pre-computes response length statistics from your data
# 2. Logs average response length every N steps during training
# 3. Saves logs to response_length_log.json

NPROC_PER_NODE=1 \
MODEL_NAME="$MODEL_NAME" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
MAX_LENGTH=2048 \
PROJECT_NAME="qwen2.5-math-sft" \
EXPERIMENT_NAME="qwen2.5-math-$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_sft_training_with_logging.sh \
    trainer.logger='["console"]' \
    trainer.save_freq=500
```

**Alternative: Use Python script directly**

```bash
# More control over logging parameters
python sft/train_with_response_length_logging.py \
    --train-file data/sft/train.parquet \
    --val-file data/sft/val.parquet \
    --model-name "$MODEL_NAME" \
    --output-dir ./checkpoints/sft_qwen2.5_math \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --epochs 3 \
    --nproc-per-node 1 \
    --log-freq 10
```

### 3.3 Multi-GPU Training

For faster training with multiple GPUs:

```bash
NPROC_PER_NODE=4 \
MODEL_NAME="$MODEL_NAME" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math" \
BATCH_SIZE=2 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
bash scripts/run_sft_training_with_logging.sh
```

### 3.4 Training with LoRA/PEFT (Memory Efficient)

For large models or limited GPU memory:

```bash
NPROC_PER_NODE=2 \
MODEL_NAME="$MODEL_NAME" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math_lora" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-4 \
EPOCHS=3 \
LORA_RANK=16 \
LORA_ALPHA=32 \
bash scripts/run_sft_peft.sh
```

## Step 4: Monitor Training

### 4.1 Console Output

The training script will log:
- Training loss
- Learning rate
- Average response length per batch (if using enhanced script)
- Throughput (tokens/sec)
- GPU memory usage

### 4.2 Check Training Progress

```bash
# Watch training logs
tail -f checkpoints/sft_qwen2.5_math/*/train.log 2>/dev/null || echo "Logs will appear during training"

# Check checkpoint directories
ls -lh checkpoints/sft_qwen2.5_math/
```

## Step 5: Locate Final Model Weights

### 5.1 Checkpoint Structure

After training completes, checkpoints are saved in:
```
checkpoints/sft_qwen2.5_math/
├── PROJECT_NAME/
│   └── EXPERIMENT_NAME/
│       ├── global_step_0/
│       ├── global_step_500/
│       ├── global_step_1000/
│       └── global_step_final/
```

### 5.2 Find Final Model

```bash
# List all checkpoints
find checkpoints/sft_qwen2.5_math -type d -name "global_step_*" | sort

# Find the final checkpoint
FINAL_CHECKPOINT=$(find checkpoints/sft_qwen2.5_math -type d -name "global_step_final" | head -1)
echo "Final checkpoint: $FINAL_CHECKPOINT"

# Check checkpoint contents
ls -lh "$FINAL_CHECKPOINT"
```

### 5.3 Verify Model Weights

```bash
# Check if model files exist
FINAL_CHECKPOINT=$(find checkpoints/sft_qwen2.5_math -type d -name "global_step_final" | head -1)
if [ -n "$FINAL_CHECKPOINT" ]; then
    echo "Model checkpoint found at: $FINAL_CHECKPOINT"
    ls -lh "$FINAL_CHECKPOINT"
    
    # Check for model files
    if [ -f "$FINAL_CHECKPOINT/adapter_model.safetensors" ] || [ -f "$FINAL_CHECKPOINT/model.safetensors" ]; then
        echo "✓ Model weights found"
    else
        echo "Checking for other weight formats..."
        ls "$FINAL_CHECKPOINT" | grep -E "\.(safetensors|bin|pt|pth)$"
    fi
else
    echo "Final checkpoint not found. Check training logs."
fi
```

## Step 6: Use Fine-Tuned Model

### 6.1 For Full Fine-Tuning

```bash
# Set the checkpoint path
CHECKPOINT_PATH="./checkpoints/sft_qwen2.5_math/PROJECT_NAME/EXPERIMENT_NAME/global_step_final"

# Use for inference
python scripts/run_vllm_inference.py \
    --model "$CHECKPOINT_PATH" \
    --prompts-file data/math_test_prompts.jsonl \
    --output-file outputs/finetuned_solutions.jsonl
```

### 6.2 For LoRA/PEFT Models

```bash
# LoRA models need the base model + adapter
BASE_MODEL="/path/to/your/qwen2.5-math-model"
ADAPTER_PATH="./checkpoints/sft_qwen2.5_math_lora/.../global_step_final"

# Load with PEFT
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL')
model = PeftModel.from_pretrained(base_model, '$ADAPTER_PATH')
model.save_pretrained('./merged_model')
"
```

## Troubleshooting

### Issue: Model Not Found

```bash
# Verify model path
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
print('Model found!')
"
```

### Issue: Out of Memory

- Reduce batch size: `BATCH_SIZE=2` or `BATCH_SIZE=1`
- Reduce sequence length: `MAX_LENGTH=1024`
- Use LoRA: `bash scripts/run_sft_peft.sh`
- Use gradient checkpointing: Add `model.gradient_checkpointing=true`

### Issue: Data Format Errors

```bash
# Verify data format
python -c "
import pandas as pd
df = pd.read_parquet('data/sft/train.parquet')
print('Columns:', list(df.columns))
print('Required: prompt, generated_solution')
assert 'prompt' in df.columns, 'Missing prompt column'
assert 'generated_solution' in df.columns, 'Missing generated_solution column'
print('✓ Data format correct')
"
```

### Issue: Training Too Slow

- Increase batch size if memory allows
- Use more GPUs: `NPROC_PER_NODE=4`
- Enable mixed precision: Add `trainer.mixed_precision=bf16`

## Quick Reference

### Complete Training Pipeline

```bash
# Step 1: Prepare data
python sft/prepare_data.py "outputs/*.jsonl" -o data/sft --val-split 0.1

# Step 2: Set your local model path
# Option A: Local directory
MODEL_NAME="/path/to/your/qwen2.5-math-model"

# Option B: HuggingFace model name (if cached locally)
# MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"

# Step 3: Run training with response length logging
NPROC_PER_NODE=1 \
MODEL_NAME="$MODEL_NAME" \
TRAIN_FILE="data/sft/train.parquet" \
VAL_FILE="data/sft/val.parquet" \
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math" \
BATCH_SIZE=4 \
LEARNING_RATE=1e-5 \
EPOCHS=3 \
MAX_LENGTH=2048 \
bash scripts/run_sft_training_with_logging.sh

# Step 4: Find final model weights
FINAL_CHECKPOINT=$(find checkpoints/sft_qwen2.5_math -type d -name "global_step_final" | head -1)
if [ -n "$FINAL_CHECKPOINT" ]; then
    echo "✓ Model saved at: $FINAL_CHECKPOINT"
    ls -lh "$FINAL_CHECKPOINT"
else
    echo "Check training logs for checkpoint location"
    ls -lh checkpoints/sft_qwen2.5_math/*/
fi

# Step 5: Check response length logs and graph
echo "Response length statistics:"
cat checkpoints/sft_qwen2.5_math/response_length_stats.json | python -m json.tool
echo ""
echo "Training-time logs:"
cat checkpoints/sft_qwen2.5_math/response_length_log.json | python -m json.tool | head -20
echo ""
echo "Response length graph:"
ls -lh checkpoints/sft_qwen2.5_math/response_length_vs_examples.*
echo ""
echo "To view the graph, open: checkpoints/sft_qwen2.5_math/response_length_vs_examples.png"
```

### Key Files Created

After running the pipeline, you'll have:

```
checkpoints/sft_qwen2.5_math/
├── PROJECT_NAME/
│   └── EXPERIMENT_NAME/
│       ├── global_step_0/          # Initial checkpoint
│       ├── global_step_500/        # Intermediate checkpoints
│       ├── global_step_1000/
│       └── global_step_final/       # Final model weights ⭐
│           ├── config.json
│           ├── model.safetensors (or .bin files)
│           └── tokenizer files
├── response_length_stats.json       # Pre-computed statistics
├── response_length_log.json         # Training-time logs
├── response_length_vs_examples.png  # Matplotlib graph (PNG)
├── response_length_vs_examples.pdf # Matplotlib graph (PDF)
└── training.log                     # Full training log
```

## Response Length Logging and Visualization

The enhanced training scripts automatically log response length information and generate a matplotlib graph:

### What Gets Logged

1. **Pre-training Statistics** (saved to `response_length_stats.json`):
   - Mean response length (tokens)
   - Min/max response lengths
   - Median and standard deviation
   - Total number of examples

2. **During Training** (saved to `response_length_log.json`):
   - Step number
   - Average response length per batch (estimated from data statistics)
   - Loss value (when available)
   - Logged every N steps (default: every 10 steps)

3. **Visualization** (saved to `response_length_vs_examples.png` and `.pdf`):
   - Matplotlib graph showing response length vs number of SFT examples processed
   - Includes mean/median lines and standard deviation bands
   - Subplot showing training loss (if available) or response length distribution

### Log Files and Graphs

After training, check these files in your output directory:

```bash
OUTPUT_DIR="./checkpoints/sft_qwen2.5_math"

# View pre-computed statistics
cat $OUTPUT_DIR/response_length_stats.json | python -m json.tool

# View training-time logs
cat $OUTPUT_DIR/response_length_log.json | python -m json.tool

# View the matplotlib graph
# The graph shows:
# - Top plot: Response length vs examples processed (with mean/median/std dev lines)
# - Bottom plot: Training loss over time OR distribution of response lengths
ls -lh $OUTPUT_DIR/response_length_vs_examples.*

# View full training log
tail -100 $OUTPUT_DIR/training.log
```

### Example Log Output

During training, you'll see messages like:
```
[Response Length Log] Step 10: Avg response length ≈ 1234.5 tokens (from data statistics)
[Response Length Log] Step 20: Avg response length ≈ 1234.5 tokens (from data statistics)
...
Generating response length vs examples processed graph...
✓ Saved plot to: ./checkpoints/sft_qwen2.5_math/response_length_vs_examples.png
✓ Saved plot (PDF) to: ./checkpoints/sft_qwen2.5_math/response_length_vs_examples.pdf
```

### Graph Features

The generated graph includes:
- **Main plot**: Line graph showing how average response length changes as more examples are processed
- **Reference lines**: Overall mean and median response lengths
- **Standard deviation band**: Shaded region showing ±1 std dev
- **Statistics box**: Summary statistics (mean, median, min, max, std dev)
- **Subplot**: Either training loss over time (if available) or distribution histogram

The visualization helps you:
- Monitor if response lengths change during training
- Identify trends in the data
- Debug data issues
- Optimize batch sizes and sequence lengths
- Understand the distribution of response lengths in your training data

## Next Steps

1. **Evaluate fine-tuned model**: Run evaluation on test set
2. **Compare results**: Compare before/after fine-tuning metrics
3. **Iterate**: Generate more data with fine-tuned model and retrain
4. **Deploy**: Use fine-tuned model for production inference
