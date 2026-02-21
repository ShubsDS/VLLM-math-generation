# VLLM MATH Generation (Minimal)

This repository now contains only two experiment pipelines:

1. `scripts/run_local_pipeline.py`
   - MATH-only inference + evaluation in one command.
   - Supports MATH `train`/`test`, subset filtering, max sample caps.
   - Writes full generations JSONL and one markdown evaluation report.
   - Optional flag writes a correct-only JSONL for SFT data collection.

2. SFT data + training
   - `scripts/convert_jsonl_to_parquet.py`: convert correct-teacher JSONL into `train.parquet` + `val.parquet`.
   - `scripts/run_sft_training.sh`: plain verl SFT with epoch checkpoints saved as HuggingFace format.

## Requirements

- Python 3.12+
- CUDA GPU(s)
- vLLM + datasets + pandas + transformers
- verl available in your environment for SFT training

Install project dependencies with your preferred environment manager, e.g. `uv`.

## 1) Run MATH inference + evaluation

on Mult-iGPU, must set env variable: 

`export VLLM_WORKER_MULTIPROC_METHOD=spawn`

```bash
python scripts/run_local_pipeline.py \
  --model "Qwen/Qwen2.5-Math-7B-Instruct" \
  --split test \
  --subset all \
  --max-tokens 16384
```

Optional correct-only export:

```bash
python scripts/run_local_pipeline.py \
  --model "Qwen/Qwen2.5-Math-7B-Instruct" \
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

## 2) Convert correct JSONL to train/val parquet

```bash
python scripts/convert_jsonl_to_parquet.py \
  --input outputs/<correct_file>.jsonl \
  --output-dir data/sft
```

Defaults:
- Validation split: `0.1`
- Shuffle: enabled
- Seed: `42`
- Output files: `data/sft/train.parquet` and `data/sft/val.parquet`

Input JSONL must include:
- `prompt`
- `generated_solution`

## 3) Run plain SFT training

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
- console + wandb logging
- checkpoint frequency = every epoch
- checkpoint content = `hf_model` only

## Key CLI references

### `scripts/run_local_pipeline.py`
- `--model` (required): HF model ID or local model path
- `--split`: `train` or `test`
- `--subset`: MATH subset (default `all`)
- `--max-samples`: optional cap
- `--max-tokens`: generation max new tokens
- `--save-correct-jsonl`: optional correct-only export

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
