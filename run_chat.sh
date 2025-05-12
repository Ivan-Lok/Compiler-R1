export CUDA_VISIBLE_DEVICES=0,1

# python3 -m agent_r1.vllm_infer.chat \
#     --env optimizer \
#     --api-key EMPTY \
#     --api-base http://localhost:8000/v1 \
#     --model agent \
#     --temperature 0.7 \
#     --top-p 0.8 \
#     --max-tokens 512 \
#     --repetition-penalty 1.05 \
#     --input-file /root/data/compiler_autotuning_grpo/validation_val-cbench.parquet


#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="agent_r1.vllm_infer.chat"
DATA_DIR="/root/data/compiler_autotuning_grpo/"
OUTPUT_FILE="overoz_summary_with_rate.txt" # Changed output filename
LLVM_IR_DIR="/root/Agent-R1_qwertyuiop/Agent-R1/examples/data_preprocess/llvmir_datasets/"
LLVM_TOOLS_PATH="/root/Agent-R1_qwertyuiop/Agent-R1/agent_r1/tool/tools/comiler_autotuning/raw_tool/"

declare -a DATASETS=(
    "validation_val-cbench.parquet"
    "validation_val-mibench.parquet"
    "validation_val-blas.parquet"
    "validation_val-opencv.parquet"
    "validation_val-chstone.parquet"
    "validation_val-tensorflow.parquet"
    "validation_val-npb.parquet"
)

COMMON_ARGS=(
    --env optimizer
    --api-key EMPTY
    --api-base http://localhost:8001/v1
    --model agent
    --temperature 0.7
    --top-p 0.8
    --max-tokens 10240
    --repetition-penalty 1.05 # 1.1
    --llvm-ir-dir "$LLVM_IR_DIR"
    --llvm-tools-path "$LLVM_TOOLS_PATH"
    # --no-color # Keep Python output clean for parsing
)

# --- Script Logic ---

# Prepare the output file and write the header - Added Success Rate Column
printf "%-40s | %-15s | %-18s\n" "Dataset" "Average OverOz" "Success Rate (%)" > "$OUTPUT_FILE"
printf "%-40s-|-%-15s-|-%-18s\n" "----------------------------------------" "---------------" "------------------" >> "$OUTPUT_FILE"

echo "Starting batch processing..."

# Loop through each dataset filename
for dataset_file in "${DATASETS[@]}"; do
    full_input_path="${DATA_DIR}${dataset_file}"

    echo "-----------------------------------------------------"
    echo "Processing: ${dataset_file}"
    echo "-----------------------------------------------------"

    # Initialize variables for each dataset
    avg_overoz="N/A"
    exclude_count="N/A"
    attempted_count="N/A"
    success_rate="N/A"

    # Check if the input file exists
    if [[ ! -f "$full_input_path" ]]; then
        echo "Error: Input file not found: ${full_input_path}"
        avg_overoz="File_Not_Found"
        # Keep counts/rate as N/A
    else
        # Run the Python script and capture the output
        echo "Running Python script for ${dataset_file}..."
        script_output=$(python3 -m "$PYTHON_SCRIPT" "${COMMON_ARGS[@]}" --input-file "$full_input_path" --no-color 2>&1) # Force no-color for easier parsing
        # echo "${script_output}"
        # --- Extract Data from Output ---

        # 1. Extract Average OverOz Score
        avg_overoz=$(echo "$script_output" | grep -oP 'Average OverOz Score \(for included records\): \K[0-9.-]+')
        if [[ -z "$avg_overoz" ]]; then
            # Check specific non-success messages
             if echo "$script_output" | grep -q "No valid OverOz scores were calculated"; then
                 avg_overoz="No_Scores"
             elif echo "$script_output" | grep -q "Error"; then
                 avg_overoz="Error_Detected"
             else
                 avg_overoz="N/A" # Default if line not found
             fi
            echo "Warning: Could not extract average OverOz score for ${dataset_file}. Set to ${avg_overoz}."
        else
             echo "Extracted Average OverOz: ${avg_overoz}"
        fi

        # 2. Extract Included Count
        exclude_count=$(echo "$script_output" | grep -oP 'Records finally failed \(excluded from avg\): \K[0-9]+')
        if [[ -z "$exclude_count" ]]; then
            echo "Warning: Could not extract 'exclude count' for ${dataset_file}."
            exclude_count="N/A"
        else
            echo "Extracted Excluded Count: ${exclude_count}"
        fi

        # 3. Extract Attempted Count
        attempted_count=$(echo "$script_output" | grep -oP 'Total records attempted: \K[0-9]+')
        if [[ -z "$attempted_count" ]]; then
            echo "Warning: Could not extract 'attempted count' for ${dataset_file}."
            attempted_count="N/A"
        else
             echo "Extracted Attempted Count: ${attempted_count}"
        fi

        # 4. Calculate Success Rate (only if counts are valid numbers)
        if [[ "$exclude_count" =~ ^[0-9]+$ && "$attempted_count" =~ ^[0-9]+$ ]]; then
            if [[ "$attempted_count" -gt 0 ]]; then
                # Use bc for floating point calculation (scale=2 means 2 decimal places)
                success_rate=$(echo "scale=2; (($attempted_count - $exclude_count) * 100) / $attempted_count" | bc)
                echo "Calculated Success Rate: ${success_rate}%"
            elif [[ "$attempted_count" -eq 0 ]]; then
                 success_rate="0.00" # Or N/A if 0 attempted means error
                 echo "Attempted count is 0, setting success rate to 0.00%"
            else
                 success_rate="N/A" # Should not happen if regex matched
            fi
        else
            echo "Cannot calculate success rate due to missing/invalid counts."
            success_rate="N/A"
        fi
    fi

    # Append the result to the output file using printf for alignment
    # Ensure variables are treated as strings for printf
    printf "%-40s | %-15s | %18s\n" "$dataset_file" "$avg_overoz" "${success_rate}%" >> "$OUTPUT_FILE"

done

echo "====================================================="
echo "Batch processing finished."
echo "Results saved to: ${OUTPUT_FILE}"
echo "====================================================="

# Display the final table
cat "$OUTPUT_FILE"


# export CUDA_VISIBLE_DEVICES=0,1

# python3 -m agent_r1.vllm_infer.chat \
#     --env optimizer \
#     --api-key EMPTY \
#     --api-base http://localhost:8000/v1 \
#     --model agent \
#     --temperature 0.7 \
#     --top-p 0.8 \
#     --max-tokens 4096 \
#     --repetition-penalty 1.05