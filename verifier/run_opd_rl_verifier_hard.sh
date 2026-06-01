#!/bin/bash
# Hybrid OPD + RL verifier training on DAPO rollout data
#
# The actor (student verifier) sees: problem + solution (unconditional)
# The frozen teacher verifier reads a privileged ref_prompt containing the
# problem, student solution, verdict label, and oracle/reference solution.
#
# This launcher trains with a hybrid objective:
#   total_loss = OPD_COEF * opd_loss + PG_COEF * policy_gradient_loss
#
# The reward is defined by verifier_reward.py:
#   reward = 1.0 iff the verdict is correct and a FEEDBACK field is present.
#
# Prerequisites:
#   1) Run prepare_verifier_data.py to generate RL verifier parquets.
#   2) Run prepare_verifier_distill_data.py to add the ref_prompt column
#      used by the teacher-conditioned OPD prompt.
#
# Usage:
#   ./verifier/run_opd_rl_verifier_hard.sh
#
# Configurable parameters (set as environment variables):
#   TRAIN_FILES       Training data files (default: dapo_distill_train.parquet)
#   VAL_FILES         Validation data files (default: dapo_rl_val.parquet)
#   MODEL             Student model path (default: Qwen/Qwen3-8B)
#   REF_MODEL         Frozen teacher model path (default: same as MODEL)
#   LR                Learning rate (default: 1e-6)
#   ALPHA             OPD alpha: 0=forward, 0.5=JSD, 1=reverse (default: 1.0)
#   OPD_COEF          Weight on OPD distillation loss (default: 1.0)
#   PG_COEF           Weight on RL policy loss (default: 1.0)
#   RL_ALGO           RL advantage estimator (default: grpo)
#   ROLLOUT_N         Number of rollouts per prompt (default: 8)
#   BATCH_SIZE        Train batch size (default: 16)
#   GPU_MEM_UTIL      vLLM GPU memory utilization target (default: 0.7)
#   EXPERIMENT_NAME   Custom experiment name (auto-generated if not set)

set -x

echo "START TIME: $(date)"

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
mkdir -p "${WANDB_DIR}"
ulimit -c 0

# =============================================================================
# CONFIGURABLE PARAMETERS (override via environment variables)
# =============================================================================
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
REF_MODEL=${REF_MODEL:-"${MODEL}"}
LR=${LR:-"1e-6"}
ALPHA=${ALPHA:-"1.0"}
OPD_COEF=${OPD_COEF:-"1.0"}
PG_COEF=${PG_COEF:-"1.0"}
RL_ALGO=${RL_ALGO:-"grpo"}
ROLLOUT_N=${ROLLOUT_N:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$TRAIN_FILES" ]; then
    TRAIN_FILES="['${SCRIPT_DIR}/data/dapo_distill_train.parquet']"
fi

if [ -z "$VAL_FILES" ]; then
    VAL_FILES="['${SCRIPT_DIR}/data/dapo_rl_val.parquet']"
fi

# =============================================================================
# DERIVED SETTINGS
# =============================================================================

if [ -z "$EXPERIMENT_NAME" ]; then
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
    EXPERIMENT_NAME="opd_${RL_ALGO}_verifier_hard_${MODEL_NAME}_alpha${ALPHA}_lr${LR}_opd${OPD_COEF}_pg${PG_COEF}"
fi

REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"${SCRIPT_DIR}/verifier_reward.py"}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-"compute_score"}

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${RL_ALGO} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.ref_use_ref_prompt=True \
    algorithm.ref_prompt_key=ref_prompt \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.35 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=opd_rl \
    actor_rollout_ref.actor.opd.alpha=${ALPHA} \
    actor_rollout_ref.actor.opd.is_clip=2.0 \
    actor_rollout_ref.actor.opd.distillation_topk=100 \
    actor_rollout_ref.actor.opd.full_logit_distillation=True \
    actor_rollout_ref.actor.opd.distillation_add_tail=True \
    actor_rollout_ref.actor.opd.opd_coef=${OPD_COEF} \
    actor_rollout_ref.actor.opd.pg_coef=${PG_COEF} \
    +actor_rollout_ref.ref.model.path=${REF_MODEL} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    custom_reward_function.path=${REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${REWARD_FUNCTION_NAME} \
    reward_model.use_reward_loop=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_verifier_distillation \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.total_epochs=8 \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.log_val_generations=20 \
    trainer.rollout_data_dir=checkpoints/${EXPERIMENT_NAME}/rollout_data \
    trainer.validation_data_dir=checkpoints/${EXPERIMENT_NAME}/validation_data \
    "$@"

echo "END TIME: $(date)"
