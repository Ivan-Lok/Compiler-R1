#!/bin/bash
set -x

# 确保脚本在 Compiler-R1 环境中运行
source /root/anaconda3/etc/profile.d/conda.sh
conda activate Compiler-R1

# 设置环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
# 限制只使用GPU 0和GPU 1
export CUDA_VISIBLE_DEVICES=0,1

# 默认参数
nproc_per_node=2  # 修改为只使用2个GPU
base_model="Qwen/Qwen2.5-1.5B-Instruct"
project_name="compiler_autotuning_qwen"
sft_output_dir="$HOME/outputs/compiler_autotuning_sft"
sft_steps=1000
grpo_steps=500

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --nproc_per_node)
      echo "Warning: nproc_per_node is fixed to 2 (GPU 0 and 1)"
      shift 2
      ;;
    --model)
      base_model="$2"
      shift 2
      ;;
    --project_name)
      project_name="$2"
      shift 2
      ;;
    --sft_output_dir)
      sft_output_dir="$2"
      shift 2
      ;;
    --sft_steps)
      sft_steps="$2"
      shift 2
      ;;
    --grpo_steps)
      grpo_steps="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 确保 Ray 使用 Compiler-R1 环境
export PYTHONPATH=/root/anaconda3/envs/Compiler-R1/bin:$PYTHONPATH

# 设置实验名称
sft_experiment_name="sft-$(basename $base_model)"
grpo_experiment_name="grpo-after-sft-$(basename $base_model)"

# 检查SFT数据集是否存在
if [ -f "$HOME/data/compiler_autotuning_sft/train.parquet" ] && \
   [ -f "$HOME/data/compiler_autotuning_sft/validation.parquet" ]; then
    echo "检测到已存在的SFT数据集"
    read -p "是否重新构建SFT数据集？(y/n): " rebuild_sft
    if [ "$rebuild_sft" = "y" ]; then
        echo "===================================================================="
        echo "==== 第1步: 重新准备SFT数据集 ===="
        echo "===================================================================="
        mkdir -p $HOME/data/compiler_autotuning_sft
        export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/
        python3 -m examples.data_preprocess.compiler_autotuning_sft \
          --use_rag \
          --data_file=examples/data_preprocess/compiler_autotuning_data.csv \
          --local_dir=$HOME/data/compiler_autotuning_sft \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
          --max_samples=3000
    else
        echo "使用现有的SFT数据集继续..."
    fi
else
    echo "===================================================================="
    echo "==== 第1步: 准备SFT数据集 ===="
    echo "===================================================================="
    mkdir -p $HOME/data/compiler_autotuning_sft
    export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/
    python3 -m examples.data_preprocess.compiler_autotuning_sft \
      --use_rag \
      --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
      --data_file=examples/data_preprocess/compiler_autotuning_data.csv \
      --local_dir=$HOME/data/compiler_autotuning_sft \
      --max_samples=3000
fi

# 检查SFT检查点是否存在
latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
if [ ! -z "$latest_checkpoint" ]; then
    echo "检测到已存在的SFT检查点: $latest_checkpoint"
    read -p "是否重新进行SFT训练？(y/n): " retrain_sft
    if [ "$retrain_sft" = "y" ]; then
        echo "===================================================================="
        echo "==== 第2步: 重新进行SFT训练 ===="
        echo "===================================================================="
        # 确保SFT输出目录存在
        mkdir -p $sft_output_dir
        export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/verl
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
          -m verl.trainer.fsdp_sft_trainer \
          data.train_files=$HOME/data/compiler_autotuning_sft/train.parquet \
          data.val_files=$HOME/data/compiler_autotuning_sft/validation.parquet \
          data.train_batch_size=8 \
          data.micro_batch_size_per_gpu=4 \
          data.prompt_key=extra_info \
          data.response_key=extra_info \
          optim.lr=1e-5 \
          +data.prompt_dict_keys=['question'] \
          +data.response_dict_keys=['answer'] \
          data.micro_batch_size=4 \
          data.max_length=8192 \
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

        echo "SFT训练完成，模型保存在 $sft_output_dir"
        # 更新最新的检查点路径
        latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
    else
        echo "使用现有的SFT检查点继续..."
    fi
else
    echo "===================================================================="
    echo "==== 第2步: SFT训练 ===="
    echo "===================================================================="
    # 确保SFT输出目录存在
    mkdir -p $sft_output_dir
    export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/verl
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
      -m verl.trainer.fsdp_sft_trainer \
      data.train_files=$HOME/data/compiler_autotuning_sft/train.parquet \
      data.val_files=$HOME/data/compiler_autotuning_sft/validation.parquet \
      data.train_batch_size=8 \
      data.micro_batch_size_per_gpu=4 \
      data.prompt_key=extra_info \
      data.response_key=extra_info \
      optim.lr=1e-5 \
      +data.prompt_dict_keys=['question'] \
      +data.response_dict_keys=['answer'] \
      data.micro_batch_size=4 \
      data.max_length=8192 \
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

    echo "SFT训练完成，模型保存在 $sft_output_dir"
    # 获取最新的检查点路径
    latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
fi

if [ -z "$latest_checkpoint" ]; then
  echo "错误: 未找到SFT训练的检查点！"
  # 设置一个默认或者存在的检查点路径用于测试
  latest_checkpoint=$base_model
  echo "使用基础模型继续: $latest_checkpoint"
fi

echo "使用SFT检查点: $latest_checkpoint"

echo "===================================================================="
echo "==== 第3步: 使用SFT模型进行GRPO训练 ===="
echo "===================================================================="

# 检查GRPO数据集是否存在
if [ -f "$HOME/data/compiler_autotuning_grpo/train.parquet" ] && \
   [ -f "$HOME/data/compiler_autotuning_grpo/validation_val-cbench.parquet" ]; then
    echo "检测到已存在的GRPO数据集"
    read -p "是否重新构建GRPO数据集？(y/n): " rebuild_grpo
    if [ "$rebuild_grpo" = "y" ]; then
        echo "准备增强版GRPO数据..."
        export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/
        python3 -m examples.data_preprocess.compiler_autotuning \
          --data_file=examples/data_preprocess/train_random200max_LLM_new_filter.csv \
          --local_dir=$HOME/data/compiler_autotuning_grpo \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
          --val_files examples/data_preprocess/val-cbench.csv \
                      examples/data_preprocess/val-blas.csv \
                      examples/data_preprocess/val-chstone.csv \
                      examples/data_preprocess/val-mibench.csv \
                      examples/data_preprocess/val-npb.csv \
                      examples/data_preprocess/val-opencv.csv \
                      examples/data_preprocess/val-tensorflow.csv
    else
        echo "使用现有的GRPO数据集继续..."
    fi
else
    echo "准备增强版GRPO数据..."
    export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/
    python3 -m examples.data_preprocess.compiler_autotuning \
          --data_file=examples/data_preprocess/train_random200max_LLM_new_filter.csv \
          --local_dir=$HOME/data/compiler_autotuning_grpo \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets/ \
          --val_files examples/data_preprocess/val-cbench.csv \
                      examples/data_preprocess/val-blas.csv \
                      examples/data_preprocess/val-chstone.csv \
                      examples/data_preprocess/val-mibench.csv \
                      examples/data_preprocess/val-npb.csv \
                      examples/data_preprocess/val-opencv.csv \
                      examples/data_preprocess/val-tensorflow.csv
fi

# GRPO阶段使用4个GPU (0,1,2,3)
export PYTHONPATH=/root/Agent-R1_phl/Agent-R1/verl/
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 运行GRPO训练，使用SFT训练好的模型
python3 -m agent_r1.src.main_agent \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/compiler_autotuning_grpo/train.parquet \
  "data.val_files=[$HOME/data/compiler_autotuning_grpo/validation_val-cbench.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-blas.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-chstone.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-mibench.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-npb.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-opencv.parquet,$HOME/data/compiler_autotuning_grpo/validation_val-tensorflow.parquet]" \
  data.train_batch_size=128 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.max_start_length=4096 \
  data.max_tool_response_length=4096 \
  \
  actor_rollout_ref.model.path=$latest_checkpoint \
  +actor_rollout_ref.model.torch_dtype=bfloat16 \
  +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n_repeat=5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  \
  trainer.critic_warmup=0 \
  "trainer.logger=[console,wandb]" \
  trainer.project_name=$project_name \
  trainer.experiment_name=$grpo_experiment_name \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  \
  tool.env='optimizer'
  # trainer.total_training_steps=165 \
echo "完成SFT和GRPO训练流程！" 