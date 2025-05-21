export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

# Default parameters
nproc_per_node=2
base_model="Qwen/Qwen2.5-7B-Instruct"
project_name="compiler_autotuning_qwen"
sft_output_dir="./model_save/cold_start_model/7B/"
sft_steps=1000
grpo_steps=500

# Parse command line arguments
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
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Set experiment names
sft_experiment_name="sft-$(basename $base_model)"
grpo_experiment_name="grpo-after-sft-$(basename $base_model)"

# Check if SFT dataset exists
if [ -f "./dataset/cold_start/train.parquet" ] && \
   [ -f "./dataset/cold_start/validation.parquet" ]; then
    echo "Detected existing SFT dataset"
    read -p "Rebuild SFT dataset? (y/n): " rebuild_sft
    if [ "$rebuild_sft" = "y" ]; then
        echo "===================================================================="
        echo "==== Step 1: Rebuild SFT Dataset ===="
        echo "===================================================================="
        mkdir -p $HOME/data/compiler_autotuning_sft
        export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/
        python3 -m examples.data_preprocess.compiler_autotuning_sft \
          --data_file=examples/data_preprocess/Experiment_1_2.csv \
          --local_dir=./dataset/cold_start/ \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
          --max_samples=800
    else
        echo "Using existing SFT dataset..."
    fi
else
    echo "===================================================================="
    echo "==== Step 1: Prepare SFT Dataset ===="
    echo "===================================================================="
    mkdir -p $HOME/data/compiler_autotuning_sft
    export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/
    python3 -m examples.data_preprocess.compiler_autotuning_sft \
      --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
      --data_file=examples/data_preprocess/Experiment_1_2.csv \
      --local_dir=./dataset/cold_start/ \
      --max_samples=800
fi

# Check if SFT checkpoint exists
latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
if [ ! -z "$latest_checkpoint" ]; then
    echo "Detected existing SFT checkpoint: $latest_checkpoint"
    read -p "Retrain SFT? (y/n): " retrain_sft
    if [ "$retrain_sft" = "y" ]; then
        echo "===================================================================="
        echo "==== Step 2: Retrain SFT ===="
        echo "===================================================================="
        # Ensure SFT output directory exists
        mkdir -p $sft_output_dir
        
        torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
          -m verl.trainer.fsdp_sft_trainer \
          data.train_files=./dataset/cold_start/train.parquet \
          data.val_files=./dataset/cold_start/validation.parquet \
          data.train_batch_size=8 \
          data.micro_batch_size_per_gpu=4 \
          data.prompt_key=extra_info \
          data.response_key=extra_info \
          optim.lr=1e-6 \
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

        echo "SFT training completed, model saved at $sft_output_dir"
        # Update latest checkpoint path
        latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
    else
        echo "Using existing SFT checkpoint..."
    fi
else
    echo "===================================================================="
    echo "==== Step 2: SFT Training ===="
    echo "===================================================================="
    # Ensure SFT output directory exists
    mkdir -p $sft_output_dir
    export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/verl
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
      -m verl.trainer.fsdp_sft_trainer \
      data.train_files=./dataset/cold_start/train.parquet \
      data.val_files=./dataset/cold_start/validation.parquet \
      data.train_batch_size=8 \
      data.micro_batch_size_per_gpu=4 \
      data.prompt_key=extra_info \
      data.response_key=extra_info \
      optim.lr=1e-6 \
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

    echo "SFT training completed, model saved at $sft_output_dir"
    # Get latest checkpoint path
    latest_checkpoint=$(ls -dt $sft_output_dir/global_step_* 2>/dev/null | head -n 1)
fi

if [ -z "$latest_checkpoint" ]; then
  echo "Error: No SFT training checkpoint found!"
  # Set a default or existing checkpoint path for testing
  latest_checkpoint=$base_model
  echo "Using base model: $latest_checkpoint"
fi

echo "Using SFT checkpoint: $latest_checkpoint"

echo "===================================================================="
echo "==== Step 3: GRPO Training with SFT Model ===="
echo "===================================================================="

# Check if GRPO dataset exists
if [ -f "./dataset/rl//train.parquet" ] && \
   [ -f "./dataset/rl//validation_val-cbench.parquet" ]; then
    echo "Detected existing GRPO dataset"
    read -p "Rebuild GRPO dataset? (y/n): " rebuild_grpo
    if [ "$rebuild_grpo" = "y" ]; then
        echo "Preparing enhanced GRPO data..."
        export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/
        python3 -m examples.data_preprocess.compiler_autotuning \
          --data_file=examples/data_preprocess/Experiment_1_2.csv \
          --local_dir=./dataset/rl/ \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets \
          --val_files examples/data_preprocess/val-cbench.csv \
                      examples/data_preprocess/val-blas.csv \
                      examples/data_preprocess/val-chstone.csv \
                      examples/data_preprocess/val-mibench.csv \
                      examples/data_preprocess/val-npb.csv \
                      examples/data_preprocess/val-opencv.csv \
                      examples/data_preprocess/val-tensorflow.csv
    else
        echo "Continuing with existing GRPO dataset..."
    fi
else
    echo "Preparing enhanced GRPO data..."
    export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/
    python3 -m examples.data_preprocess.compiler_autotuning \
          --data_file=examples/data_preprocess/Experiment_1_2.csv \
          --local_dir=./dataset/rl/ \
          --llvm_ir_dir=examples/data_preprocess/llvmir_datasets/ \
          --val_files examples/data_preprocess/val-cbench.csv \
                      examples/data_preprocess/val-blas.csv \
                      examples/data_preprocess/val-chstone.csv \
                      examples/data_preprocess/val-mibench.csv \
                      examples/data_preprocess/val-npb.csv \
                      examples/data_preprocess/val-opencv.csv \
                      examples/data_preprocess/val-tensorflow.csv
fi

export PYTHONPATH=/PATH_PLACEHOLDER/NIPS_Material/verl/
python3 -m agent_r1.src.main_agent \
  algorithm.adv_estimator=grpo \
  data.train_files=./dataset/rl//train.parquet \
  "data.val_files=[./dataset/rl//validation_val-cbench.parquet,./dataset/rl//validation_val-blas.parquet,./dataset/rl//validation_val-chstone.parquet,./dataset/rl//validation_val-mibench.parquet,./dataset/rl//validation_val-npb.parquet,./dataset/rl//validation_val-opencv.parquet,./dataset/rl//validation_val-tensorflow.parquet]" \
  data.train_batch_size=64 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n_repeat=5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  \
  trainer.critic_warmup=0 \
  "trainer.logger=[console,wandb]" \
  trainer.project_name=$project_name \
  trainer.experiment_name=$grpo_experiment_name \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  \
  tool.env='optimizer' \
  trainer.total_training_steps=40
echo "Finished" 