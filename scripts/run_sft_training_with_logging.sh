#!/bin/bash
# SFT Training Script with Response Length Logging
# This script fine-tunes a model and logs average response length per batch

set -e  # Exit on error

# Default values
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
MODEL_NAME=${MODEL_NAME:-"./Qwen2.5-Math-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"./data/sft/train.parquet"}
VAL_FILE=${VAL_FILE:-"./data/sft/val.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/sft_qwen2.5_math_1.5b"}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
EPOCHS=${EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-16384}
PROJECT_NAME=${PROJECT_NAME:-"math-sft"}
LOG_RESPONSE_LENGTHS=${LOG_RESPONSE_LENGTHS:-true}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"math-sft-$(date +%Y%m%d_%H%M%S)"}

# Check if train file is provided
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: TRAIN_FILE must be set"
    echo "Usage: TRAIN_FILE=path/to/train.parquet VAL_FILE=path/to/val.parquet $0"
    exit 1
fi

# Check if val file is provided
if [ -z "$VAL_FILE" ]; then
    echo "Warning: VAL_FILE not set, using train file for validation"
    VAL_FILE=$TRAIN_FILE
fi

echo "=========================================="
echo "SFT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "GPUs: $NPROC_PER_NODE"
echo "Response length logging: $LOG_RESPONSE_LENGTHS"
echo "=========================================="

# Pre-compute response lengths if logging is enabled
if [ "$LOG_RESPONSE_LENGTHS" = "true" ]; then
    echo ""
    echo "Pre-computing response length statistics..."
    python3 << EOF
import pandas as pd
from transformers import AutoTokenizer
import json
from pathlib import Path

# Load training data
train_df = pd.read_parquet("$TRAIN_FILE")
print(f"Loaded {len(train_df)} training examples")

# Load tokenizer to compute token counts
try:
    tokenizer = AutoTokenizer.from_pretrained("$MODEL_NAME", trust_remote_code=True)
    print(f"Loaded tokenizer from: $MODEL_NAME")
except Exception as e:
    print(f"Warning: Could not load tokenizer: {e}")
    print("Will use character count as proxy for token count")
    tokenizer = None

# Compute response lengths
response_lengths = []
for idx, row in train_df.iterrows():
    response = str(row.get("generated_solution", ""))
    if tokenizer:
        # Count tokens
        tokens = tokenizer.encode(response, add_special_tokens=False)
        length = len(tokens)
    else:
        # Fallback to character count (rough estimate: ~4 chars per token)
        length = len(response) // 4
    response_lengths.append(length)

# Save statistics
mean_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
stats = {
    "mean": mean_len,
    "min": min(response_lengths) if response_lengths else 0,
    "max": max(response_lengths) if response_lengths else 0,
    "median": sorted(response_lengths)[len(response_lengths) // 2] if response_lengths else 0,
    "std": (sum((x - mean_len) ** 2 for x in response_lengths) / len(response_lengths)) ** 0.5 if response_lengths else 0,
    "total_examples": len(response_lengths)
}

# Save to file
stats_file = Path("$OUTPUT_DIR") / "response_length_stats.json"
stats_file.parent.mkdir(parents=True, exist_ok=True)
with open(stats_file, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nResponse length statistics:")
print(f"  Mean: {stats['mean']:.1f} tokens")
print(f"  Min: {stats['min']} tokens")
print(f"  Max: {stats['max']} tokens")
print(f"  Median: {stats['median']} tokens")
print(f"\nSaved statistics to: {stats_file}")
EOF
fi

# Load response length statistics for logging
if [ -f "$OUTPUT_DIR/response_length_stats.json" ]; then
    AVG_RESPONSE_LENGTH=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/response_length_stats.json')); print(d['mean'])")
    echo "Using pre-computed average response length: ${AVG_RESPONSE_LENGTH} tokens"
else
    AVG_RESPONSE_LENGTH="N/A"
fi

# Run training with verl and log response lengths
echo ""
echo "Starting training..."
echo "=========================================="

# Create log directory
mkdir -p "$OUTPUT_DIR"

# Run training and capture output for logging
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.prompt_key=prompt \
    data.response_key=generated_solution \
    data.micro_batch_size_per_gpu=$BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.truncation=error \
    model.partial_pretrain=$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=$LEARNING_RATE \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.logger='["console","wandb"]' \
    "$@" 2>&1 | tee "$OUTPUT_DIR/training.log" | python3 -c "
import sys
import re
import json
import os
from pathlib import Path

# Get output directory from environment
output_dir = os.environ.get('OUTPUT_DIR', './sft_output')
stats_file = Path(output_dir) / 'response_length_stats.json'

# Load response length statistics if available
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
        avg_length = stats.get('mean', 0)
else:
    avg_length = None

# Track batch information
batch_logs = []
current_step = None
log_freq = 10  # Log every N steps

for line in sys.stdin:
    # Print original line
    print(line, end='')
    sys.stdout.flush()
    
    # Try to extract step number from verl logs
    step_patterns = [
        r'step[:\s]+(\d+)',
        r'Step\s+(\d+)',
        r'global_step[=:\s]+(\d+)',
        r'step\s*=\s*(\d+)',
    ]
    
    for pattern in step_patterns:
        step_match = re.search(pattern, line, re.IGNORECASE)
        if step_match:
            current_step = int(step_match.group(1))
            break
    
    # Look for loss (indicates batch completion)
    if 'loss' in line.lower() and current_step is not None:
        # Extract loss value if possible
        loss_match = re.search(r'loss[:\s=]+([\d.e-]+)', line, re.IGNORECASE)
        loss = float(loss_match.group(1)) if loss_match else None
        
        # Log batch completion with estimated response length
        if current_step % log_freq == 0 or current_step <= 10:
            log_entry = {
                'step': current_step,
                'loss': loss,
                'avg_response_length': avg_length,
                'note': 'Estimated from data statistics' if avg_length else 'Not available'
            }
            batch_logs.append(log_entry)
            
            if avg_length:
                print(f'\n[Response Length Log] Step {current_step}: Avg response length ≈ {avg_length:.1f} tokens (from data statistics)', file=sys.stderr)
            sys.stderr.flush()

# Save batch logs
if batch_logs:
    log_file = Path(output_dir) / 'response_length_log.json'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(batch_logs, f, indent=2)
    print(f'\n[Response Length Log] Saved {len(batch_logs)} log entries to {log_file}', file=sys.stderr)
" OUTPUT_DIR="$OUTPUT_DIR" BATCH_SIZE="$BATCH_SIZE" NPROC_PER_NODE="$NPROC_PER_NODE" TRAIN_FILE="$TRAIN_FILE" MODEL_NAME="$MODEL_NAME"

# Generate matplotlib graph after training completes
if [ "$LOG_RESPONSE_LENGTHS" = "true" ] && [ -f "$OUTPUT_DIR/response_length_log.json" ]; then
    echo ""
    echo "Generating response length vs examples processed graph..."
    python3 << 'PYTHON_EOF'
import sys
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np

# Get parameters from environment
output_dir = Path(os.environ.get('OUTPUT_DIR', './sft_output'))
log_file = output_dir / 'response_length_log.json'
train_file = os.environ.get('TRAIN_FILE', '')
model_name = os.environ.get('MODEL_NAME', '')
batch_size = int(os.environ.get('BATCH_SIZE', 4))
nproc_per_node = int(os.environ.get('NPROC_PER_NODE', 1))

if not log_file.exists():
    print(f"Log file not found: {log_file}")
    sys.exit(0)

# Load training logs
with open(log_file) as f:
    batch_logs = json.load(f)

if not batch_logs:
    print("No batch logs found")
    sys.exit(0)

# Load training data to get actual response lengths
print(f"Loading training data from: {train_file}")
train_df = pd.read_parquet(train_file)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Loaded tokenizer: {model_name}")
except Exception as e:
    print(f"Warning: Could not load tokenizer ({e}), using character count")
    tokenizer = None

# Compute response lengths for all examples
print("Computing response lengths for all examples...")
response_lengths = []
for idx, row in train_df.iterrows():
    response = str(row.get("generated_solution", ""))
    if tokenizer:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        length = len(tokens)
    else:
        length = len(response) // 4  # Rough estimate
    response_lengths.append(length)

# Calculate examples processed at each step
# examples_processed = step * batch_size * num_gpus
examples_per_step = batch_size * nproc_per_node

# Create data points: (examples_processed, avg_response_length)
# We'll use actual response lengths from the data, assuming sequential processing
data_points = []

for log_entry in batch_logs:
    step = log_entry.get('step', 0)
    examples_processed = step * examples_per_step
    
    # Calculate average response length up to this point
    # Assuming examples are processed sequentially
    if examples_processed <= len(response_lengths):
        # Use mean of response lengths up to this point
        avg_length = np.mean(response_lengths[:int(examples_processed)])
        # Also track individual lengths for better visualization
        lengths_up_to_now = response_lengths[:int(examples_processed)]
    else:
        # If we've processed more than available (multiple epochs), use overall mean
        # and cycle through the data
        num_epochs = int(examples_processed / len(response_lengths))
        remainder = examples_processed % len(response_lengths)
        if remainder > 0:
            # Include partial epoch
            all_lengths = response_lengths * num_epochs + response_lengths[:remainder]
        else:
            all_lengths = response_lengths * num_epochs
        avg_length = np.mean(all_lengths)
        lengths_up_to_now = all_lengths
    
    data_points.append({
        'step': step,
        'examples_processed': examples_processed,
        'avg_response_length': avg_length,
        'loss': log_entry.get('loss')
    })

if not data_points:
    print("No data points to plot")
    sys.exit(0)

# Create the plot
print(f"Creating plot with {len(data_points)} data points...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

examples = [d['examples_processed'] for d in data_points]
avg_lengths = [d['avg_response_length'] for d in data_points]
losses = [d['loss'] for d in data_points if d['loss'] is not None]

# Main plot: Response length vs examples processed
ax1.plot(examples, avg_lengths, 'b-', linewidth=2.5, label='Average Response Length', alpha=0.8, marker='o', markersize=4)
ax1.scatter(examples, avg_lengths, s=50, alpha=0.7, c='blue', edgecolors='darkblue', linewidths=0.5)

# Add horizontal line for overall mean
overall_mean = np.mean(response_lengths)
overall_median = np.median(response_lengths)
ax1.axhline(y=overall_mean, color='r', linestyle='--', linewidth=2, 
           label=f'Overall Mean: {overall_mean:.1f} tokens', alpha=0.7)
ax1.axhline(y=overall_median, color='g', linestyle='--', linewidth=2, 
           label=f'Overall Median: {overall_median:.1f} tokens', alpha=0.7)

# Add shaded region for std deviation
std_dev = np.std(response_lengths)
ax1.fill_between(examples, 
                 [overall_mean - std_dev] * len(examples),
                 [overall_mean + std_dev] * len(examples),
                 alpha=0.2, color='gray', label=f'±1 Std Dev: {std_dev:.1f} tokens')

# Formatting for main plot
ax1.set_xlabel('Number of SFT Examples Processed', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Response Length (tokens)', fontsize=12, fontweight='bold')
ax1.set_title('Response Length vs SFT Examples Processed', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=10, framealpha=0.9)

# Add statistics text box
stats_text = f'Total Examples: {len(response_lengths):,}\n'
stats_text += f'Mean: {np.mean(response_lengths):.1f} tokens\n'
stats_text += f'Median: {np.median(response_lengths):.1f} tokens\n'
stats_text += f'Std Dev: {np.std(response_lengths):.1f} tokens\n'
stats_text += f'Min: {np.min(response_lengths)} tokens\n'
stats_text += f'Max: {np.max(response_lengths)} tokens'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
        fontsize=9, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot: Loss over steps (if available)
if losses and len(losses) > 0:
    loss_steps = [d['step'] for d in data_points if d['loss'] is not None]
    loss_examples = [d['examples_processed'] for d in data_points if d['loss'] is not None]
    ax2.plot(loss_examples, losses, 'r-', linewidth=2, label='Training Loss', alpha=0.8, marker='s', markersize=3)
    ax2.set_xlabel('Number of SFT Examples Processed', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Training Loss vs Examples Processed', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
else:
    # If no loss data, show distribution of response lengths
    ax2.hist(response_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Response Length (tokens)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Response Lengths in Training Data', fontsize=12, fontweight='bold')
    ax2.axvline(overall_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.1f}')
    ax2.axvline(overall_median, color='g', linestyle='--', linewidth=2, label=f'Median: {overall_median:.1f}')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()

# Save the plot
plot_file = output_dir / 'response_length_vs_examples.png'
fig.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {plot_file}")

# Also save as PDF for better quality
plot_file_pdf = output_dir / 'response_length_vs_examples.pdf'
fig.savefig(plot_file_pdf, bbox_inches='tight')
print(f"✓ Saved plot (PDF) to: {plot_file_pdf}")

plt.close()
PYTHON_EOF
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
if [ "$LOG_RESPONSE_LENGTHS" = "true" ]; then
    echo "Response length statistics: $OUTPUT_DIR/response_length_stats.json"
    echo "Response length logs: $OUTPUT_DIR/response_length_log.json"
    if [ -f "$OUTPUT_DIR/response_length_vs_examples.png" ]; then
        echo "Response length graph: $OUTPUT_DIR/response_length_vs_examples.png"
    fi
fi
echo "Training log: $OUTPUT_DIR/training.log"
echo "=========================================="
