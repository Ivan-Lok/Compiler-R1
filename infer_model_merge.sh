export CHECKPOINT_DIR="/PATH_PLACEHOLDER/NIPS_Material/checkpoints/compiler_autotuning_qwen/grpo-after-sft-Qwen2.5-7B-Instruct/global_step_40/actor/"

python3 verl/scripts/model_merger.py --local_dir $CHECKPOINT_DIR