#!/bin/bash
set -x

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=1
group_size=1
initial_n=1
model_path=/raid3/data/GTPO/qwen2.5-1.5b-instruct

### useful
history_length=5
default_val_data_size=64
prompt_type=gtpo # react, gtpo, gtpo_wo_planning, gtpo_wo_think, gtpo_wo_explore
use_api_as_teacher=True ## whether to run on train, if True, use api as teacher in train time
PROVIDER=moonshot # azure, google, moonshot
MODEL_NAME=kimi-k2-0905-preview # gemini-2.5-pro, gpt-5, kimi-k2-0905-preview
MAX_WORKERS=64 

SEEDS=(0)
GENERALIZATIONS=(2)
ENV_NAMES=("webshop")

for env_name in "${ENV_NAMES[@]}"; do
    for generalization_level in "${GENERALIZATIONS[@]}"; do
        
        # === 需求 1: Webshop 特殊处理 ===
        # 如果 env_name 是 webshop 且 generalization_level 不是 0，则跳过
        if [[ "$env_name" == "webshop" && "$generalization_level" != "0" ]]; then
            echo "Skipping webshop for generalization level $generalization_level (only running gen 0)."
            continue
        fi
        
        val_data_size=$default_val_data_size

        for seed in "${SEEDS[@]}"; do
        
            SAVE_PATH=/raid1/HOME/szhang/jywu/code/Planning/GTPO/data/teacher_apis/${MODEL_NAME}_${env_name}_gen_${generalization_level}_${prompt_type}
            data_dir=$HOME/data/api_agent_${env_name}

            # 数据准备（使用动态变量）
            python3 examples/data_preprocess/prepare.py \
              --mode 'text' \
              --train_data_size $train_data_size \
              --val_data_size $val_data_size \
              --local_dir $data_dir

            # ===== 训练启动（使用动态变量） =====
            python3 -m verl.trainer.main_ppo \
              algorithm.adv_estimator=gtpo \
              data.train_files=$data_dir/text/train.parquet \
              data.val_files=$data_dir/text/test.parquet \
              data.train_batch_size=$train_data_size \
              data.val_batch_size=$val_data_size \
              data.max_prompt_length=6000 \
              data.max_response_length=512 \
              data.filter_overlong_prompts=True \
              data.truncation='error' \
              data.return_raw_chat=True \
              actor_rollout_ref.model.path=$model_path \
              actor_rollout_ref.actor.optim.lr=1e-6 \
              actor_rollout_ref.model.use_remove_padding=True \
              actor_rollout_ref.actor.ppo_mini_batch_size=256 \
              actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
              actor_rollout_ref.actor.use_kl_loss=True \
              actor_rollout_ref.actor.kl_loss_coef=0.01 \
              actor_rollout_ref.actor.kl_loss_type=low_var_kl \
              actor_rollout_ref.model.enable_gradient_checkpointing=True \
              actor_rollout_ref.actor.fsdp_config.param_offload=False \
              actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
              actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
              actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
              actor_rollout_ref.rollout.name=$ENGINE \
              actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
              actor_rollout_ref.rollout.enable_chunked_prefill=False \
              actor_rollout_ref.rollout.enforce_eager=False \
              actor_rollout_ref.rollout.free_cache_engine=False \
              actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
              actor_rollout_ref.rollout.val_kwargs.do_sample=True \
              actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
              actor_rollout_ref.ref.fsdp_config.param_offload=True \
              actor_rollout_ref.actor.use_invalid_action_penalty=True \
              actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
              algorithm.use_kl_in_reward=False \
              +env.use_api_as_teacher=$use_api_as_teacher \
              +env.api.provider=$PROVIDER \
              +env.api.model_name=$MODEL_NAME \
              +env.api.max_workers=$MAX_WORKERS \
              env.use_api=True \
              env.env_name=$env_name \
              env.seed=$seed \
              env.max_steps=30 \
              env.rollout.n=$group_size \
              env.rollout.initial_n=$initial_n \
              env.history_length=$history_length \
              env.alfworld.generalization_level=$generalization_level \
              env.sciworld.generalization_level=$generalization_level \
              env.webshop.generalization_level=$generalization_level \
              env.prompt_type=$prompt_type \
              trainer.critic_warmup=0 \
              trainer.logger=['console'] \
              trainer.project_name='verl_agent_multi_env' \
              trainer.experiment_name="${env_name}_seed_${seed}_gen_${generalization_level}" \
              trainer.n_gpus_per_node=1 \
              trainer.nnodes=1 \
              trainer.save_freq=5 \
              trainer.test_freq=5 \
              trainer.total_epochs=0 \
              trainer.default_local_dir=$SAVE_PATH \
              trainer.rollout_data_dir=$SAVE_PATH \
              trainer.val_before_train=True

            # ===== 结果处理 =====
            cp $SAVE_PATH/eval/0.jsonl $SAVE_PATH/${seed}.jsonl

            python data/convert2json.py \
              --input_path $SAVE_PATH/${seed}.jsonl \
              --output_path data/${env_name}_cs_L${generalization_level}/interaction.json
        done
    done
done
