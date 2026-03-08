ENV=alfworld
cs_data_name=spark
max_length=2048 # 3000 for sciworld, 2048 for alfworld, 6000 for webshop
micro_batch_size_per_gpu=4 # 4 for alfworld, sciworld; 2 for webshop
model_path=your_path_to_model
local_dir=$HOME/data/spark_cs/$ENV # for datasets storage, you can change to your own path

python3 -m examples.data_preprocess.cold_start_data \
    --local_dir=$local_dir \
    --data_source="data/${ENV}_cs_${cs_data_name}/${ENV}_cold-start.json" \

torchrun  --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=32\
    data.train_files=$local_dir/train.parquet \
    data.val_files=$local_dir/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=$max_length \
    data.truncation='left'\
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu\
    model.partial_pretrain=$model_path \
    trainer.default_hdfs_dir=null \
    trainer.project_name=spark \
    trainer.experiment_name=${ENV}_qwen1.5b_cold-start_$cs_data_name \
    trainer.total_epochs=3 \
    trainer.default_local_dir="your_path_to_local_dir/spark_${ENV}_qwen1.5b_cold-start_$cs_data_name" \
    trainer.logger=['console'] \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true