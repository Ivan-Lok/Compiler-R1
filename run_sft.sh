# export PYTHONPATH=/root/Agent-R1_qwertyuiop/Agent-R1/
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1

# base_model="Qwen/Qwen2.5-1.5B-Instruct"
# project_name="compiler_autotuning_qwen"
# sft_output_dir="$HOME/outputs/compiler_autotuning_sft/pure_llvmcode/llvmcode/"
# sft_experiment_name="pure-llvmcode-$(basename $base_model)"

# # python3 -m examples.data_preprocess.compiler_autotuning_pure_llvmcode \
# #     --llvm_ir_dir=examples/data_preprocess/llvmir_datasets/ \
# #     --data_file=examples/data_preprocess/ga_best_pass_sequences_vs_original_results.csv \
# #     --local_dir=$HOME/data/compiler_autotuning_sft/pure_llvmcode/
# #     # --max_samples=10000

# torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=$HOME/data/compiler_autotuning_sft/pure_llvmcode/train.parquet \
#     data.val_files=$HOME/data/compiler_autotuning_sft/pure_llvmcode/validation.parquet \
#     data.train_batch_size=32 \
#     data.micro_batch_size_per_gpu=8 \
#     data.prompt_key=extra_info \
#     data.response_key=extra_info \
#     optim.lr=1e-6 \
#     +data.prompt_dict_keys=['question'] \
#     +data.response_dict_keys=['answer'] \
#     data.micro_batch_size=8 \
#     data.max_length=15012 \
#     model.partial_pretrain=$base_model \
#     +model.torch_dtype=bfloat16 \
#     +model.attn_implementation=flash_attention_2 \
#     trainer.default_local_dir=$sft_output_dir \
#     trainer.project_name=$project_name \
#     trainer.experiment_name=$sft_experiment_name \
#     "trainer.logger=[console,wandb]" \
#     trainer.default_hdfs_dir=null \
#     trainer.total_epochs=1 \
#     ulysses_sequence_parallel_size=2 \
#     use_remove_padding=true


export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

base_model="Qwen/Qwen2.5-1.5B-Instruct"
project_name="compiler_autotuning_qwen"
sft_output_dir="$HOME/outputs/compiler_autotuning_sft/pure_autophase/"
sft_experiment_name="pure-autophase_sft-$(basename $base_model)"

python3 -m examples.data_preprocess.compiler_autotuning_pure_sft \
    --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
    --data_file=examples/data_preprocess/ga_best_pass_sequences_vs_original_results.csv \
    --local_dir=$HOME/data/compiler_autotuning_sft/pure_autophase/ \
    # --max_samples=8000

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/compiler_autotuning_sft/pure_autophase/train.parquet \
    data.val_files=$HOME/data/compiler_autotuning_sft/pure_autophase/validation.parquet \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=8 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-6 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=8 \
    data.max_length=15012 \
    model.partial_pretrain=$base_model \
    +model.torch_dtype=bfloat16 \
    +model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$sft_output_dir \
    trainer.project_name=$project_name \
    trainer.experiment_name=$sft_experiment_name \
    "trainer.logger=[console,wandb]" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true