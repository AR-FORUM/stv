#!/bin/bash
# GRPO generator training on multi-turn feedback prompts.
#
# The generator learns to follow verifier feedback and produce correct solutions
# after being told its initial attempt was wrong. Prompts are multi-turn:
#   [user: problem, assistant: round-0 solution, user: feedback reprompt]
#
# Continues training from the RL generator checkpoint.
# Modeled after run_grpo_generator_standard_hard_qwen3_8b.sh.

set -x

echo "START TIME: $(date)"

export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
mkdir -p "${WANDB_DIR}"
ulimit -c 0

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
MODEL=${MODEL:-"checkpoints/generators/grpo_generator_standard_hard_Qwen-Qwen3-8B_lr1e-6/global_step_800/huggingface"}
LR=${LR:-"1e-6"}
ROLLOUT_N=${ROLLOUT_N:-8}
BATCH_SIZE=${BATCH_SIZE:-16}

if [ -z "$TRAIN_FILES" ]; then
    echo "ERROR: TRAIN_FILES must be set explicitly." >&2
    exit 1
fi

if [ -z "$VAL_FILES" ]; then
    echo "ERROR: VAL_FILES must be set explicitly." >&2
    exit 1
fi

if [ -z "$EXPERIMENT_NAME" ]; then
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
    EXPERIMENT_NAME="grpo_generator_feedback_hard_${MODEL_NAME}_lr${LR}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_FUNCTION_PATH="${SCRIPT_DIR}/../examples/grpo_trainer/math_reward.py"

# Multi-turn prompts include problem + solution + feedback, so need much more
# prompt space than the standard generator (512).
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.35 \
    actor_rollout_ref.actor.kl_loss_type=k3+ \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=${REWARD_FUNCTION_PATH} \
    reward_model.use_reward_loop=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_generator_feedback \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.total_epochs=32 \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.log_val_generations=20 \
    trainer.rollout_data_dir=checkpoints/${EXPERIMENT_NAME}/rollout_data \
    trainer.validation_data_dir=checkpoints/${EXPERIMENT_NAME}/validation_data \
    "$@"

echo "END TIME: $(date)"
