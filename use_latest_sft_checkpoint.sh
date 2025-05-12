#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# 1. Conda Environment (Optional - uncomment if needed)
# CONDA_ENV_NAME="Compiler-R1"
# CONDA_BASE_PATH="/root/anaconda3" # Adjust if needed

# 2. GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1 # Set which GPUs to make *visible* to the script

# 3. Paths
SFT_MODEL_PARENT_DIR="$HOME//outputs/compiler_autotuning_sft/pure_autophase/" # ADJUST THIS
MODEL_CHECKPOINT_PATH="" # Set to empty to find latest
TEST_BASE_DIR="/root/Agent-R1_qwertyuiop/Agent-R1/examples/data_preprocess/llvmir_datasets/after_cleaned_test/"
RAW_TOOL_PATH="/root/Agent-R1_qwertyuiop/Agent-R1/agent_r1/tool/tools/comiler_autotuning/raw_tool/"
# *** UPDATE PYTHON SCRIPT NAME ***
PYTHON_SCRIPT_NAME="evaluate_sft_batch.py" # Use the new multiprocessing script

# 4. Evaluation Parameters
DATASETS_TO_TEST=( "cbench-v1" "blas-v0" "mibench-v1" "chstone-v0" "tensorflow-v0" "npb-v0" "opencv-v0" )
NUM_ANSWERS_LIST=(1)
BASE_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" # ADJUST THIS
# *** Adjust NUM_WORKERS based on GPU count and memory ***
# Example: 2 visible GPUs, maybe start with 2 or 4 workers
NUM_WORKERS=16

# 5. Output Filenames
OUTPUT_TABLE_FILE="batch_evaluation_results_mp.txt"
OUTPUT_PLOT_FILE="batch_evaluation_plot_mp.png"

# --- End Configuration ---

# ... (Keep Conda activation and path validation the same) ...
# Activate Conda Environment (Optional - uncomment if needed)
# echo "Activating Conda environment: $CONDA_ENV_NAME..."
# source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
# if ! conda activate "$CONDA_ENV_NAME"; then
#     echo "错误：无法激活 Conda 环境 '$CONDA_ENV_NAME'。请确保环境存在并且 Conda 配置正确。"
#     exit 1
# fi
# echo "Conda environment activated."
# echo "Using Python: $(which python)"

# Determine Model Checkpoint Path
# ... (Checkpoint finding logic remains the same) ...
if [ -z "$MODEL_CHECKPOINT_PATH" ]; then
    echo "正在 $SFT_MODEL_PARENT_DIR 中搜索最新的 'global_step_*' 检查点..."
    latest_checkpoint=$(ls -d "$SFT_MODEL_PARENT_DIR"/global_step_* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$latest_checkpoint" ] || [ ! -d "$latest_checkpoint" ]; then
        echo "错误: 在 $SFT_MODEL_PARENT_DIR 中未找到 'global_step_*' 检查点目录！"
        exit 1
    fi
    MODEL_CHECKPOINT_PATH="$latest_checkpoint"
    echo "自动找到最新的检查点: $MODEL_CHECKPOINT_PATH"
else
    echo "使用指定的检查点路径: $MODEL_CHECKPOINT_PATH"
    if [ ! -e "$MODEL_CHECKPOINT_PATH" ]; then
       echo "错误: 指定的检查点路径不存在: $MODEL_CHECKPOINT_PATH"
       exit 1
    fi
fi

# Validate other paths
# ... (Path validation logic remains the same) ...
# echo "检查所需路径..."
# if [ ! -d "$TEST_BASE_DIR" ]; then
#     echo "错误: 测试数据集基础目录不存在: $TEST_BASE_DIR"
#     exit 1
# fi
# dataset_check_flag=0
# for ds in "${DATASETS_TO_TEST[@]}"; do
#     if [ -d "$TEST_BASE_DIR/$ds" ]; then
#         dataset_check_flag=1
#         break
#     fi
# done
# if [ $dataset_check_flag -eq 0 ]; then
#     echo "错误: 在 $TEST_BASE_DIR 中未找到任何指定的测试数据集目录 (${DATASETS_TO_TEST[*]})"
#     exit 1
# fi
if [ ! -d "$RAW_TOOL_PATH" ]; then
    echo "错误: Raw tool 路径不存在: $RAW_TOOL_PATH"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "错误: Python 脚本未找到: $PYTHON_SCRIPT_NAME"
    exit 1
fi
echo "所有路径检查通过。"

# Convert arrays to space-separated strings for argparse
datasets_arg="${DATASETS_TO_TEST[*]}"
num_answers_arg="${NUM_ANSWERS_LIST[*]}"

# Run the Python Batch Evaluation Script
echo "========================================"
echo "开始运行 SFT 模型批量评估脚本 (多进程)..."
echo "模型检查点: $MODEL_CHECKPOINT_PATH"
echo "测试基础目录: $TEST_BASE_DIR"
echo "测试数据集: $datasets_arg"
echo "测试答案数: $num_answers_arg"
echo "Raw Tool 路径: $RAW_TOOL_PATH"
echo "基础模型 (备用): $BASE_MODEL_NAME"
echo "可见 GPUs: $CUDA_VISIBLE_DEVICES"
echo "工作进程数: $NUM_WORKERS"
echo "输出表格: $OUTPUT_TABLE_FILE"
echo "输出绘图: $OUTPUT_PLOT_FILE"
echo "========================================"

python3 "$PYTHON_SCRIPT_NAME" \
    --model_path "$MODEL_CHECKPOINT_PATH" \
    --test_base_dir "$TEST_BASE_DIR" \
    --raw_tool_path "$RAW_TOOL_PATH" \
    --datasets /root/Agent-R1_qwertyuiop/Agent-R1/examples/data_preprocess/llvmir_datasets/after_cleaned_test \
    --num_answers_list $num_answers_arg \
    --base_model "$BASE_MODEL_NAME" \
    --output_table "$OUTPUT_TABLE_FILE" \
    --output_plot "$OUTPUT_PLOT_FILE" \
    --num_workers "$NUM_WORKERS"
    # Note: --use_gpu is removed, script now relies on CUDA_VISIBLE_DEVICES

# Capture exit status
PYTHON_EXIT_CODE=$?

# ... (Keep exit code handling and optional conda deactivate the same) ...
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "批量评估脚本成功完成。"
    echo "结果表格保存在: $OUTPUT_TABLE_FILE"
    echo "结果绘图保存在: $OUTPUT_PLOT_FILE"
    echo "========================================"
else
    echo "========================================"
    echo "错误：批量评估脚本执行失败，退出码: $PYTHON_EXIT_CODE"
    echo "========================================"
    exit $PYTHON_EXIT_CODE
fi

exit 0