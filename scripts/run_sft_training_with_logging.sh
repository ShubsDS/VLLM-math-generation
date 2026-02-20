#!/bin/bash
# SFT Training Script with Response Length Tracking
#
# Two complementary length-measurement mechanisms:
#   1. During training (wandb): every TEST_FREQ steps, run greedy decoding on
#      GENERATION_PROBE_SIZE val prompts and log val/avg_generated_length.
#      Set GENERATION_PROBE_SIZE=0 to disable.
#   2. After training (JSON + plot): load each epoch HF checkpoint from disk,
#      run inference on PROBE_SIZE val prompts, save response_length_by_checkpoint.json
#      and response_length_vs_training.png.

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Default values
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
MODEL_NAME=${MODEL_NAME:-"./Qwen2.5-Math-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"./data/sft/train.parquet"}
VAL_FILE=${VAL_FILE:-"./data/sft/val.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/bigtemp/fvc9ch/checkpoints/sft_qwen2.5_math_1.5b"}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
EPOCHS=${EPOCHS:-10}
MAX_LENGTH=${MAX_LENGTH:-16384}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PROJECT_NAME=${PROJECT_NAME:-"math-sft"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"math-sft-$(date +%Y%m%d_%H%M%S)"}
# How often (in steps) to run validation + in-training generation probe.
# Default 1 = run every step for maximum data density (slower but smooth curve).
# Increase (e.g. 5) to reduce overhead if each probe step is too slow.
TEST_FREQ=${TEST_FREQ:-1}
# Number of val prompts for in-training greedy-decoding probe (logged to wandb as val/avg_generated_length).
# Smaller = faster per-step overhead. 5-10 is usually enough for a reliable mean.
# Set to 0 to disable in-training generation measurement entirely.
GENERATION_PROBE_SIZE=${GENERATION_PROBE_SIZE:-5}
# Number of val prompts for post-training checkpoint inference (JSON + plot)
PROBE_SIZE=${PROBE_SIZE:-50}
# Save a checkpoint every N epochs (default 5 = checkpoint at epoch 5, 10, 15, ...)
SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS:-2}

echo "=========================================="
echo "SFT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Global train batch size: $TRAIN_BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "GPUs: $NPROC_PER_NODE"
echo "Val/probe frequency (steps): $TEST_FREQ"
echo "In-training generation probe size: $GENERATION_PROBE_SIZE"
echo "Post-training checkpoint probe size: $PROBE_SIZE"
echo "Save checkpoint every N epochs: $SAVE_EVERY_N_EPOCHS"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

# Compute steps per epoch so we can save a checkpoint at the end of each epoch
STEPS_PER_EPOCH=$(python3 -c "
import pandas as pd, math
df = pd.read_parquet('$TRAIN_FILE')
print(math.floor(len(df) / $TRAIN_BATCH_SIZE))
")
echo "Steps per epoch: $STEPS_PER_EPOCH"

SAVE_FREQ=$(( STEPS_PER_EPOCH * SAVE_EVERY_N_EPOCHS ))
echo "Checkpoint every $SAVE_EVERY_N_EPOCHS epochs ($SAVE_FREQ steps)"

# Run training
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.prompt_key=prompt \
    data.response_key=generated_solution \
    data.micro_batch_size_per_gpu=$BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.truncation=error \
    model.partial_pretrain=$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    use_remove_padding=true \
    model.use_liger=true \
    optim.lr=$LEARNING_RATE \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.generation_probe_size=$GENERATION_PROBE_SIZE \
    'trainer.checkpoint.save_contents=["model","hf_model"]' \
    trainer.resume_mode=auto \
    trainer.logger='["console","wandb"]' \
    "$@" 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Measuring response lengths at each checkpoint..."

# Measure response length at each epoch checkpoint by running inference
python3 - <<PYTHON
import json
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

output_dir = Path("$OUTPUT_DIR")
model_name = "$MODEL_NAME"
val_file = "$VAL_FILE"
probe_size = int("$PROBE_SIZE")
max_new_tokens = 2048

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Sample probe questions from val set (just the prompts, not the solutions)
val_df = pd.read_parquet(val_file)
probe_df = val_df.sample(n=min(probe_size, len(val_df)), random_state=42)
probe_prompts = probe_df["prompt"].tolist()

def measure_avg_tokens(model_path: Path) -> float:
    """Load a checkpoint and generate responses for the probe set."""
    print(f"  Loading checkpoint: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda().eval()

    token_counts = []
    with torch.no_grad():
        for prompt in probe_prompts:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).cuda()
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Count only the generated tokens, not the prompt
            n_generated = output_ids.shape[1] - input_ids.shape[1]
            token_counts.append(n_generated)

    del model
    torch.cuda.empty_cache()
    return float(np.mean(token_counts)), token_counts


# Find all hf_model checkpoint directories, sorted by step
checkpoints = sorted(
    output_dir.glob("global_step_*/huggingface"),
    key=lambda p: int(p.parent.name.split("_")[-1]),
)

if not checkpoints:
    print("No hf_model checkpoints found. Make sure trainer.checkpoint.save_contents includes 'hf_model'.")
    sys.exit(0)

print(f"Found {len(checkpoints)} checkpoints: {[str(c.parent.name) for c in checkpoints]}")

# Also measure the base model (step 0 / before training)
results = []

print("\nMeasuring base model (before training)...")
avg, counts = measure_avg_tokens(Path(model_name))
results.append({"step": 0, "label": "base", "avg_tokens": avg, "token_counts": counts})
print(f"  Base model avg tokens: {avg:.1f}")

for ckpt in checkpoints:
    step = int(ckpt.parent.name.split("_")[-1])
    print(f"\nMeasuring checkpoint at step {step}...")
    avg, counts = measure_avg_tokens(ckpt)
    results.append({"step": step, "label": f"step_{step}", "avg_tokens": avg, "token_counts": counts})
    print(f"  Step {step} avg tokens: {avg:.1f}")

# Save raw results
results_file = output_dir / "response_length_by_checkpoint.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw results to: {results_file}")

# Plot
steps = [r["step"] for r in results]
avgs = [r["avg_tokens"] for r in results]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, avgs, marker="o", linewidth=2, markersize=8)
for r in results:
    ax.annotate(f"{r['avg_tokens']:.0f}", (r["step"], r["avg_tokens"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

ax.set_xlabel("Training Step (0 = base model)", fontsize=12)
ax.set_ylabel("Avg Generated Tokens", fontsize=12)
ax.set_title("Student Model Response Length vs SFT Training Progress", fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xticks(steps)
ax.set_xticklabels([r["label"] for r in results], rotation=15, ha="right")

plt.tight_layout()
plot_path = output_dir / "response_length_vs_training.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")
PYTHON

echo ""
echo "=========================================="
echo "Done!"
echo "  Training log:    $OUTPUT_DIR/training.log"
echo "  Length results:  $OUTPUT_DIR/response_length_by_checkpoint.json"
echo "  Plot:            $OUTPUT_DIR/response_length_vs_training.png"
echo "=========================================="
