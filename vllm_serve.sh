export CUDA_VISIBLE_DEVICES=0,1
export MODEL_NAME="/root/Agent-R1_qwertyuiop/Agent-R1/checkpoints/compiler_autotuning_qwen/grpo-after-sft-Qwen2.5-3B-Instruct/global_step_40/actor/huggingface"

vllm serve $MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --served-model-name agent --port 8001 --tensor-parallel-size 2