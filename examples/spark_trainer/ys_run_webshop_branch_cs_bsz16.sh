set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=16
val_data_size=128
group_size=8
initial_n=4

prompt_type=spark
history_length=5
generalization_level=0
branch_count=2

SAVE_PATH=your_path_to_save_spark_webshop_rl_checkpoints
cs_model_path=your_path_to_cold_start_model
THIRD_LARGEST_STEP=$(ls -d "$cs_model_path"/global_step_*/ | grep -E 'global_step_[0-9]+' | sort -Vr | tail -n 3 | head -n 1)
cs_model_path=${THIRD_LARGEST_STEP%/}  # Remove trailing slash

python3 examples/data_preprocess/prepare.py \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=spark \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$cs_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
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
    env.env_name=webshop \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.rollout.initial_n=$initial_n \
    env.use_branch=True\
    env.branch_condition=explore\
    env.branch_count=$branch_count \
    env.history_length=$history_length \
    env.prompt_type=$prompt_type \
    env.webshop.generalization_level=$generalization_level \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='spark_webshop' \
    trainer.experiment_name=spark_webshop_qwen2.5_1.5b_cs_his-5_ini-${initial_n}_branch-${branch_count}_group-${group_size}_explore_bsz16 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=120 \
    trainer.default_local_dir=$SAVE_PATH/spark_webshop_qwen2.5_1.5b_cs_his-5_ini-${initial_n}_branch-${branch_count}_group-${group_size}_explore_bsz16 \
    trainer.rollout_data_dir=$SAVE_PATH/spark_webshop_qwen2.5_1.5b_cs_his-5_ini-${initial_n}_branch-${branch_count}_group-${group_size}_explore_bsz16 \
    trainer.val_before_train=True $@
