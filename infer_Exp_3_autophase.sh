#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# 1. Conda Environment (Optional - uncomment if needed)
# CONDA_ENV_NAME="Compiler-R1"
# CONDA_BASE_PATH="/root/anaconda3" # Adjust if needed

# 2. GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1 # Set which GPUs to make *visible* to the script

# 3. Paths
SFT_MODEL_PARENT_DIR=""./model_save/pure_autophase_Exp3/"" # ADJUST THIS
MODEL_CHECKPOINT_PATH=" " # Set to empty to find latest
TEST_BASE_DIR="/PATH_PLACEHOLDER/NIPS_Material/examples/data_preprocess/llvmir_datasets/after_cleaned_test/"
RAW_TOOL_PATH="/PATH_PLACEHOLDER/NIPS_Material/agent_r1/tool/tools/comiler_autotuning/raw_tool/"
# *** UPDATE PYTHON SCRIPT NAME ***
PYTHON_SCRIPT_NAME="evaluate_sft_batch_autophase.py" # Use the new multiprocessing script

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
#     echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'. Please ensure the environment exists and Conda is configured correctly."
#     exit 1
# fi
# echo "Conda environment activated."
# echo "Using Python: $(which python)"

# Determine Model Checkpoint Path
# ... (Checkpoint finding logic remains the same) ...
if [ -z "$MODEL_CHECKPOINT_PATH" ]; then
    echo "Searching for latest 'global_step_*' checkpoint in $SFT_MODEL_PARENT_DIR..."
    latest_checkpoint=$(ls -d "$SFT_MODEL_PARENT_DIR"/global_step_* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$latest_checkpoint" ] || [ ! -d "$latest_checkpoint" ]; then
        echo "Error: No 'global_step_*' checkpoint directory found in $SFT_MODEL_PARENT_DIR!"
        exit 1
    fi
    MODEL_CHECKPOINT_PATH="$latest_checkpoint"
    echo "Automatically found latest checkpoint: $MODEL_CHECKPOINT_PATH"
else
    echo "Using specified checkpoint path: $MODEL_CHECKPOINT_PATH"
    if [ ! -e "$MODEL_CHECKPOINT_PATH" ]; then
       echo "Error: Specified checkpoint path does not exist: $MODEL_CHECKPOINT_PATH"
       exit 1
    fi
fi

if [ ! -d "$RAW_TOOL_PATH" ]; then
    echo "Error: Raw tool path does not exist: $RAW_TOOL_PATH"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT_NAME"
    exit 1
fi
echo "All path checks passed."

# Convert arrays to space-separated strings for argparse
datasets_arg="${DATASETS_TO_TEST[*]}"
num_answers_arg="${NUM_ANSWERS_LIST[*]}"

# Run the Python Batch Evaluation Script
echo "========================================"
echo "Starting SFT model batch evaluation script (multiprocessing)..."
echo "Model checkpoint: $MODEL_CHECKPOINT_PATH"
echo "Test base directory: $TEST_BASE_DIR"
echo "Test datasets: $datasets_arg"
echo "Number of test answers: $num_answers_arg"
echo "Raw Tool path: $RAW_TOOL_PATH"
echo "Base model (backup): $BASE_MODEL_NAME"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of workers: $NUM_WORKERS"
echo "Output table: $OUTPUT_TABLE_FILE"
echo "Output plot: $OUTPUT_PLOT_FILE"
echo "========================================"

python3 "$PYTHON_SCRIPT_NAME" \
    --model_path "$MODEL_CHECKPOINT_PATH" \
    --test_base_dir "$TEST_BASE_DIR" \
    --raw_tool_path "$RAW_TOOL_PATH" \
    --datasets "$TEST_BASE_DIR" \
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
    echo "Batch evaluation script completed successfully."
    echo "Results table saved to: $OUTPUT_TABLE_FILE"
    echo "Results plot saved to: $OUTPUT_PLOT_FILE"
    echo "========================================"
else
    echo "========================================"
    echo "Error: Batch evaluation script failed with exit code: $PYTHON_EXIT_CODE"
    echo "========================================"
    exit $PYTHON_EXIT_CODE
fi

exit 0