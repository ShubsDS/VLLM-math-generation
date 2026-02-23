# VLLM MATH Generation (Minimal)

This repository contains four experiment pipelines:

1. `scripts/run_local_pipeline.py`
   - MATH-only inference + evaluation in one command.
   - Supports MATH `train`/`test`, subset filtering, max sample caps.
   - Writes full generations JSONL and one markdown evaluation report.
   - Optional flag writes a correct-only JSONL for SFT data collection.

2. SFT data + training
   - `scripts/convert_jsonl_to_parquet.py`: convert correct-teacher JSONL into `train.parquet` + `val.parquet`.
   - `scripts/run_sft_training.sh`: plain verl SFT with epoch checkpoints saved as HuggingFace format.

3. Trajectory length analysis
   - `scripts/measure_avg_trajectory_length.py`: module + CLI for measuring output-token statistics on a single checkpoint.
   - `scripts/sweep_trajectory_lengths.py`: sweeps every checkpoint in a directory, producing per-checkpoint histograms and a mean-trajectory plot.

4. DAPO training sweep
   - `scripts/prepare_dapo_data.sh`: download DAPO-MATH-17K and AIME-2024 datasets.
   - `scripts/run_dapo_from_checkpoint.sh`: run one DAPO job from a single SFT checkpoint.
   - `scripts/sweep_dapo_checkpoints.sh`: iterate over a list of SFT epoch checkpoints and run DAPO from each, producing comparable W&B runs in a shared project.

## Requirements

- Python 3.12+
- CUDA GPU(s)
- vLLM + datasets + pandas + transformers
- verl available in your environment for SFT training

Install project dependencies with your preferred environment manager, e.g. `uv`.

## 1) Run MATH inference + evaluation

On multi-GPU, set this env variable first:

`export VLLM_WORKER_MULTIPROC_METHOD=spawn`

```bash
python scripts/run_local_pipeline.py \
  --model "Qwen/Qwen2.5-Math-1.5B" \
  --split test \
  --subset all \
  --max-tokens 16384
```

Optional correct-only export:

```bash
python scripts/run_local_pipeline.py \
  --model "Qwen/Qwen2.5-Math-1.5B" \
  --split train \
  --save-correct-jsonl
```

### Outputs

- Full generations JSONL:
  - `outputs/<model_slug>_<split>_<timestamp>.jsonl`
- Markdown evaluation report:
  - `eval_results/<model_slug>_<timestamp>.md`
- Optional correct-only JSONL:
  - `outputs/<model_slug>_<split>_<timestamp>.correct.jsonl`

## 2) Filter correct JSONL by sequence length

verl's SFT dataset raises an error if any sequence exceeds `MAX_LENGTH`. Since
`--max-tokens` in inference caps *output* tokens only, the total sequence
(prompt + output) can exceed the training max length. Filter first:

```bash
python scripts/filter_jsonl_by_length.py \
  --input outputs/<correct_file>.correct.jsonl \
  --max-tokens 16000
```

Output is written alongside the input as `<stem>.filtered16k.correct.jsonl`.

> **Note:** use a threshold somewhat below your training `MAX_LENGTH` (e.g. 16000 for
> a 16384 limit). Token counts in the JSONL are from the inference tokenizer; verl
> re-tokenizes at training time and may produce slightly higher counts due to BPE
> boundary effects at the prompt/response join and added special tokens (EOS etc.).

## 3) Convert correct JSONL to train/val parquet

```bash
python scripts/convert_jsonl_to_parquet.py \
  --input outputs/<correct_file>.jsonl \
  --output-dir data/sft
```

Defaults:
- Validation split: `0.1`
- Shuffle: enabled
- Seed: `42`
- Output files: `data/sft/train.parquet` and `data/sift/val.parquet`

Input JSONL must include:
- `prompt`
- `generated_solution`

## 4) Run plain SFT training

```bash
MODEL_NAME="./Qwen2.5-Math-1.5B" \
TRAIN_FILE="./data/sft/train.parquet" \
VAL_FILE="./data/sft/val.parquet" \
NPROC_PER_NODE=1 \
EPOCHS=3 \
bash scripts/run_sft_training.sh
```

Behavior:
- plain verl FSDP SFT
- console logging by default; set `WANDB_ENABLED=true` to also log to wandb
- checkpoint frequency = every epoch
- checkpoint content = `hf_model` only

## 5) Measure trajectory lengths across checkpoints

After SFT training you often want to know how verbose each checkpoint is — a
shorter average trajectory indicates the model has learned to answer more
concisely; a longer one that it is reasoning more.

### Single checkpoint

```bash
python scripts/measure_avg_trajectory_length.py \
  --model ./Qwen2.5-Math-1.5B \
  --num-samples 200 \
  --max-tokens 4096
```

### Sweep all checkpoints in a directory

```bash
python scripts/sweep_trajectory_lengths.py \
  --checkpoint-dir /bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b \
  --num-samples 200 \
  --max-tokens 4096 \
  --image-dir images/sweep_qwen_1_5b
```

Checkpoints must follow the `{model_name}_{N}` naming convention produced by
`run_sft_training.sh`.  The script:
- Discovers all `*_{N}` sub-directories and sorts them by `N`.
- Uses the `huggingface/` sub-directory inside each checkpoint as the model
  path (where HF weights are saved by verl).
- Loads the dataset **once** and reuses the same shuffled sample for every
  checkpoint, ensuring a fair comparison.
- Sequentially runs vLLM inference per checkpoint, releasing GPU memory
  between runs.

### Outputs

| File | Description |
|------|-------------|
| `<image-dir>/dist_checkpoint_{N}.png` | Output-token histogram for checkpoint N |
| `<image-dir>/mean_trajectory_length.png` | Mean tokens vs. checkpoint number |
| stdout | Summary table: N, mean, std, min, p50, p95, max per checkpoint |

## 6) DAPO training sweep across SFT checkpoints

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) is used here to
continue training from SFT checkpoints, with each checkpoint producing a separate
W&B run so that training reward curves and AIME-2024 accuracy can be overlaid and
compared.

### Step 1 — Download data

```bash
bash scripts/prepare_dapo_data.sh
```

Downloads to `data/dapo/`:
- `dapo-math-17k.parquet` — DAPO-MATH-17K training prompts
- `aime-2024.parquet` — AIME-2024 validation set

Skips files that already exist.

### Step 2 — Run DAPO from a single checkpoint

```bash
CHECKPOINT_PATH=/bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b/qwen2_5_math_1_5b_10/huggingface \
bash scripts/run_dapo_from_checkpoint.sh
```

All settings have defaults. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | *(required)* | Path to HF checkpoint directory |
| `CHECKPOINT_NAME` | `basename CHECKPOINT_PATH` | Label used in W&B run name |
| `WANDB_PROJECT` | `dapo-sft-sweep` | Shared W&B project for all runs |
| `NPROC_PER_NODE` | `2` | Number of GPUs |
| `TOTAL_STEPS` | `200` | Max training steps |
| `TRAIN_FILE` | `data/dapo/dapo-math-17k.parquet` | Training data |
| `TEST_FILE` | `data/dapo/aime-2024.parquet` | Eval data |
| `OUTPUT_BASE_DIR` | `/bigtemp/fvc9ch/checkpoints` | Root for checkpoint output |

Each run is saved to `${OUTPUT_BASE_DIR}/dapo_from_${CHECKPOINT_NAME}/` and logged
to W&B as `dapo_from_${CHECKPOINT_NAME}` within the shared project.

Smoke test (5 steps):

```bash
TOTAL_STEPS=5 NPROC_PER_NODE=2 \
CHECKPOINT_PATH=/bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b/qwen2_5_math_1_5b_10/huggingface \
bash scripts/run_dapo_from_checkpoint.sh
```

### Step 3 — Sweep multiple checkpoints

Pass epoch numbers as positional arguments; the script constructs the checkpoint
path automatically from the standard `sft_qwen2_5_math_1_5b` directory:

```bash
bash scripts/sweep_dapo_checkpoints.sh 1 5 10
```

Runs are sequential to avoid GPU contention. Optional env vars (`WANDB_PROJECT`,
`NPROC_PER_NODE`, `TOTAL_STEPS`, etc.) are forwarded to the inner script:

```bash
WANDB_PROJECT=my-project TOTAL_STEPS=300 \
bash scripts/sweep_dapo_checkpoints.sh 1 3 5 7 10
```

### Algorithm details

- **Advantage estimator**: GRPO
- **Clip ratios**: `low=0.2`, `high=0.28`, `c=10.0` (DAPO decoupled clip)
- **Loss aggregation**: `token-mean`
- **KL**: disabled in both reward and loss
- **Overlong buffer**: enabled (length=4096, penalty\_factor=1.0)
- **Rollout**: vLLM, `n=8` responses per prompt, `gpu_memory_utilization=0.4`
- **Actor**: full fine-tuning with FSDP param + optimizer offloading; no LoRA
- **Eval**: `val_before_train=True`, every 10 steps on AIME-2024, 5 generations logged

### W&B comparison

All runs land in the same project. Because the experiment name encodes the SFT
epoch, the W&B compare view lets you overlay `val/test_score/mean` (AIME-2024
accuracy) and `train/reward` across checkpoints on the same step axis.

## Key CLI references

### `scripts/run_local_pipeline.py`
- `--model` (required): HF model ID or local model path
- `--split`: `train` or `test`
- `--subset`: MATH subset (default `all`)
- `--max-samples`: optional cap
- `--max-tokens`: generation max new tokens
- `--gpu-memory-utilization`: fraction of GPU memory vLLM may use (default `0.95`)
- `--save-correct-jsonl`: optional correct-only export

### `scripts/filter_jsonl_by_length.py`
- `--input` (required): correct-teacher JSONL
- `--max-tokens` (required): drop rows where `prompt_tokens + output_tokens > N`
- `--output` (optional): output path; defaults to `<stem>.filtered<N>k.correct.jsonl`

### `scripts/convert_jsonl_to_parquet.py`
- `--input` (required)
- `--output-dir` (default `data/sft`)
- `--val-ratio` (default `0.1`)
- `--seed` (default `42`)
- `--no-shuffle` (optional)

### `scripts/run_sft_training.sh`
Environment-configurable values include:
- `MODEL_NAME`, `TRAIN_FILE`, `VAL_FILE`, `OUTPUT_DIR`
- `NPROC_PER_NODE`, `BATCH_SIZE`, `TRAIN_BATCH_SIZE`
- `LEARNING_RATE`, `EPOCHS`, `MAX_LENGTH`
- `GRAD_ACCUM_STEPS`: gradient accumulation steps (default `1`); multiplied into `TRAIN_BATCH_SIZE` to simulate a larger effective batch
- `WANDB_ENABLED`: set to `true` to enable wandb logging (default `false`)

### `scripts/sweep_trajectory_lengths.py`
- `--checkpoint-dir` (required): directory containing `{model}_{N}` checkpoint sub-folders
- `--num-samples` (default `500`): MATH problems per checkpoint
- `--split` (default `test`): dataset split
- `--max-tokens` (default `16384`): max output tokens per generation
- `--temperature` (default `0.0`): sampling temperature
- `--gpu-memory-utilization` (default `0.95`)
- `--tensor-parallel-size` (default `1`)
- `--seed` (default `42`): RNG seed for reproducible dataset slicing
- `--image-dir` (default `images`): where plots are saved

### `scripts/measure_avg_trajectory_length.py`
- `--model` (required): HF model ID or local checkpoint path
- `--num-samples` (default `500`)
- `--split` (default `test`)
- `--max-tokens` (default `16384`)
- `--temperature` (default `0.0`)
- `--gpu-memory-utilization` (default `0.95`)
- `--tensor-parallel-size` (default `1`)
- `--seed` (default `42`)

### `scripts/prepare_dapo_data.sh`
No arguments. Downloads `data/dapo/dapo-math-17k.parquet` and
`data/dapo/aime-2024.parquet`; skips files that already exist.

### `scripts/run_dapo_from_checkpoint.sh`
Configured entirely via environment variables (see table in section 6 above).
`CHECKPOINT_PATH` is the only required variable.

### `scripts/sweep_dapo_checkpoints.sh`
- Positional args: epoch numbers (e.g. `1 5 10`)
- Optional env vars forwarded: `WANDB_PROJECT`, `NPROC_PER_NODE`, `TOTAL_STEPS`,
  `TRAIN_FILE`, `TEST_FILE`, `OUTPUT_BASE_DIR`
