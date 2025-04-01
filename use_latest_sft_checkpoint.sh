#!/bin/bash
set -e

# 确保脚本在 Compiler-R1 环境中运行
source /root/anaconda3/etc/profile.d/conda.sh
conda activate Compiler-R1

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 从原始脚本中提取最新的SFT检查点
sft_model_path="$HOME/outputs/compiler_autotuning_sft"
latest_checkpoint=$(ls -dt $sft_model_path/global_step_* 2>/dev/null | head -n 1)
# 如果未找到检查点，提示用户
if [ -z "$latest_checkpoint" ]; then
  echo "错误: 未找到SFT训练的检查点！"
  echo "请提供SFT模型路径，例如: $HOME/outputs/compiler_autotuning_sft/global_step_1000"
  read -p "请输入SFT模型路径: " latest_checkpoint
fi

echo "使用SFT检查点: $latest_checkpoint"

# 运行测试脚本，直接传递检查点路径
echo "开始运行SFT模型测试..."
python3 simple_test_sft.py --model_path "$latest_checkpoint"
echo "测试完成！查看上面的结果或检查model_response.txt文件" 