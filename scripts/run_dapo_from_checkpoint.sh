#!/usr/bin/env bash
# Run a single DAPO training job from one SFT checkpoint.
#
# Required:
#   CHECKPOINT_PATH   Path to HF checkpoint directory (no default)
#
# Optional (all have defaults):
#   CHECKPOINT_NAME   Human label for W&B run names (default: basename of CHECKPOINT_PATH)
#   WANDB_PROJECT     W&B project name shared across all runs (default: dapo-sft-sweep)
#   NPROC_PER_NODE    Number of GPUs (default: 2)
#   TOTAL_STEPS       Max training steps (default: 200)
#   TRAIN_FILE        Training parquet (default: <repo>/data/dapo/dapo-math-17k.parquet)
#   TEST_FILE         Eval parquet    (default: <repo>/data/dapo/aime-2024.parquet)
#   OUTPUT_BASE_DIR   Root for checkpoint output (default: /bigtemp/fvc9ch/checkpoints)
#
# Example:
#   TOTAL_STEPS=5 NPROC_PER_NODE=2 \
#   CHECKPOINT_PATH=/bigtemp/fvc9ch/checkpoints/sft_qwen2_5_math_1_5b/qwen2_5_math_1_5b_10/huggingface \
#   bash scripts/run_dapo_from_checkpoint.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="/u/fvc9ch/nlp_research/verl"

# ── Required ───────────────────────────────────────────────────────────────────
if [ -z "${CHECKPOINT_PATH:-}" ]; then
    echo "Error: CHECKPOINT_PATH is required." >&2
    exit 1
fi
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: CHECKPOINT_PATH does not exist: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

# ── Optional with defaults ─────────────────────────────────────────────────────
CHECKPOINT_NAME="${CHECKPOINT_NAME:-$(basename "${CHECKPOINT_PATH}")}"
WANDB_PROJECT="${WANDB_PROJECT:-dapo-sft-sweep}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/dapo/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/data/dapo/aime-2024.parquet}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/bigtemp/fvc9ch/checkpoints}"

# ── Derived ────────────────────────────────────────────────────────────────────
EXP_NAME="dapo_from_${CHECKPOINT_NAME}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}"

# ── Hyperparameters ────────────────────────────────────────────────────────────
max_prompt_length=2048
max_response_length=16384
overlong_buffer_len=4096
train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=16
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 4 ))

# ── Validation ─────────────────────────────────────────────────────────────────
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Error: training data not found: ${TRAIN_FILE}" >&2
    echo "Run: bash scripts/prepare_dapo_data.sh" >&2
    exit 1
fi
if [ ! -f "${TEST_FILE}" ]; then
    echo "Error: eval data not found: ${TEST_FILE}" >&2
    echo "Run: bash scripts/prepare_dapo_data.sh" >&2
    exit 1
fi
if [ ! -d "${VERL_DIR}" ]; then
    echo "Error: verl directory not found: ${VERL_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "DAPO Training Configuration"
echo "=========================================="
echo "Checkpoint:       ${CHECKPOINT_PATH}"
echo "Experiment name:  ${EXP_NAME}"
echo "W&B project:      ${WANDB_PROJECT}"
echo "GPUs:             ${NPROC_PER_NODE}"
echo "Total steps:      ${TOTAL_STEPS}"
echo "Train batch size: ${train_prompt_bsz} prompts × ${n_resp_per_prompt} responses"
echo "Mini batch size:  ${train_prompt_mini_bsz}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Train file:       ${TRAIN_FILE}"
echo "Test file:        ${TEST_FILE}"
echo "=========================================="

# Run from the verl repo root so Hydra resolves its config path correctly.
cd "${VERL_DIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # default 120s; extended to survive slow FSDP checkpoint saves

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.model.path="${CHECKPOINT_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( max_prompt_length + max_response_length )) \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$(( train_prompt_bsz * n_resp_per_prompt / 16 )) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$(( train_prompt_bsz * n_resp_per_prompt / 16 )) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    actor_rollout_ref.nccl_timeout=3600
