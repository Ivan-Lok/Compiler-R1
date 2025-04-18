#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the compiler autotuning dataset to parquet format
"""

import os
import pandas as pd
import datasets
import argparse
import json
import random
import ast
from tqdm import tqdm
import numpy as np
from verl.utils.hdfs_io import copy, makedirs
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs
from agent_r1.tool.tools.comiler_autotuning.raw_tool.gen_pass_from_number import Actions_LLVM_10_0_0


def read_llvm_ir_file(file_path):
    """
    Read LLVM IR code from a file
    
    Args:
        file_path: Path to the LLVM IR file
        
    Returns:
        LLVM IR code as string
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def get_autophase_features(ll_code):
    """
    获取LLVM IR代码的autophase特征
    
    Args:
        ll_code: LLVM IR代码
        
    Returns:
        autophase特征字典，如果发生错误则返回None
    """
    try:
        # 获取autophase特征
        features = get_autophase_obs(ll_code)
        return features
    except Exception as e:
        print(f"Error getting autophase features: {e}")
        return None


def get_all_passes():
    """
    获取所有可用的LLVM优化passes
    
    Returns:
        包含pass名称和值的列表
    """
    all_passes = []
    for action in Actions_LLVM_10_0_0:
        all_passes.append({
            "name": action.name,
            "pass": action.value
        })
    return all_passes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='compiler_autotuning_data.csv',
                        help='Path to the compiler autotuning data CSV file')
    parser.add_argument('--val_files', nargs='+', default=['cbench-val.csv'],
                        help='List of paths to validation data CSV files')
    parser.add_argument('--llvm_ir_dir', default=None, 
                        help='Directory containing LLVM IR files (optional)')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning',
                        help='Local directory to save the processed data')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to save the processed data')
    parser.add_argument('--test_ratio', type=float, default=0.001,
                        help='Ratio of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Load the main dataset
    print(f"Loading compiler autotuning data from {args.data_file}...")
    
    # Determine the full path to the main CSV file
    if os.path.isabs(args.data_file):
        csv_path = args.data_file
    else:
        # If it's a relative path, check if it's in the current directory
        if os.path.exists(args.data_file):
            csv_path = args.data_file
        # Check if it's in the same directory as this script
        elif os.path.exists(os.path.join(os.path.dirname(__file__), args.data_file)):
            csv_path = os.path.join(os.path.dirname(__file__), args.data_file)
        else:
            raise FileNotFoundError(f"Could not find {args.data_file}")
    
    # Read the main CSV file
    main_df = pd.read_csv(csv_path)
    
    # Limit the number of samples if needed
    if args.max_samples is not None and args.max_samples < len(main_df):
        main_df = main_df.sample(n=args.max_samples, random_state=args.seed)
        print(f"Limited main dataset to {len(main_df)} samples")
    
    # 获取所有可用的优化passes
    all_passes = get_all_passes()
    print(f"Found {len(all_passes)} available optimization passes")
    
    # Process the main dataframe for training and testing
    def process_dataframe(df, llvm_ir_dir=None):
        data_records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
            # Extract filename
            filename = row['Filename']
            overoz = row['OverOz']
            pass_sequence = row['PassSequence']
            
            # Read LLVM IR code if directory is provided
            ll_code = None
            if llvm_ir_dir is not None:
                ll_file_path = os.path.join(llvm_ir_dir, filename)
                ll_code = read_llvm_ir_file(ll_file_path)
            
            if ll_code:
                # 计算初始的autophase特征
                initial_features = get_autophase_features(ll_code)
                
                if initial_features:
                    # Create record
                    record = {
                        'filename': filename,
                        'll_code': ll_code,  # 保留原始代码，用于后续计算overOz
                        'autophase_features': json.dumps(initial_features),
                        'overoz' : overoz,
                        'pass_sequence': pass_sequence
                    }
                    
                    data_records.append(record)
                else:
                    print(f"Warning: Failed to get autophase features for {filename}, skipping")
            else:
                print(f"Warning: Failed to read {filename}, skipping")
        
        return data_records
    
    # Process main dataset (for train and test)
    main_records = process_dataframe(main_df, args.llvm_ir_dir)
    main_dataset = datasets.Dataset.from_pandas(pd.DataFrame(main_records))
    
    # Split the main dataset into train and test
    splits = main_dataset.train_test_split(
        test_size=args.test_ratio,
        seed=args.seed
    )
    train_dataset = splits['train']
    test_dataset = splits['test']
    
    # Process validation datasets (multiple)
    validation_datasets = {}
    
    for val_file in args.val_files:
        # Get a base name for this validation dataset (without extension)
        val_base_name = os.path.splitext(os.path.basename(val_file))[0]
        print(f"Loading validation data from {val_file}...")
        
        # Determine the full path to the validation CSV file
        if os.path.isabs(val_file):
            val_csv_path = val_file
        else:
            # If it's a relative path, check if it's in the current directory
            if os.path.exists(val_file):
                val_csv_path = val_file
            # Check if it's in the same directory as this script
            elif os.path.exists(os.path.join(os.path.dirname(__file__), val_file)):
                val_csv_path = os.path.join(os.path.dirname(__file__), val_file)
            else:
                print(f"Warning: Could not find validation file {val_file}, skipping")
                continue
        
        try:
            # Read the validation CSV file
            val_df = pd.read_csv(val_csv_path)
            print(f"Loaded {len(val_df)} validation samples from {val_file}")
            
            # Process validation dataset
            val_records = process_dataframe(val_df, args.llvm_ir_dir)
            val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(val_records))
            
            # Add to the validation datasets dictionary
            validation_datasets[val_base_name] = val_dataset
        except Exception as e:
            print(f"Error processing validation file {val_file}: {e}")
            continue
    
    # Print dataset split information
    print(f"Dataset split: {len(train_dataset)} train, {len(test_dataset)} test")
    for val_name, val_dataset in validation_datasets.items():
        print(f"Validation dataset '{val_name}': {len(val_dataset)} samples")
    
    # Instruction template
    instruction_following = f"""Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count. You will interact with a analysis tool.

**Follow this EXACT Structure and Formatting Requirements:**

It involves **three distinct phases**: an Initial Baseline Check, exactly 5 Optimization Rounds, and a Final Decision. The total interaction MUST consist of exactly 13 turns in the following sequence:

**Phase 1: Initial Baseline Check (Turns 1-2)**

1.  **Turn 1 (Assistant): Baseline Setup**
    *   Format: `<|im_start|>assistant\\n<think>...</think>\\n<tool_call>...</tool_call>\\n<|im_end|>`
    *   `<think>` Content:
        *   Must start with the marker: `[Initial Baseline Check]`
        *   Must include a plan mentioning the goal is to establish a baseline using `-Oz`. Use Python list literal format for the pass: `['-Oz']`.
    *   `<tool_call>` Content:
        *   Valid JSON: `{{"name": "analyze_autophase", "arguments": {{"filename": filename, "optimization_passes": ["-Oz"]}}}}` (Use **JSON list** for passes).

2.  **Turn 2 (User): Baseline Response**
    *   Format: `<|im_start|>user\\n<tool_response>...</tool_response>\\n<|im_end|>`
    *   (You will receive this response, containing the instruction count after applying `-Oz`).

**Phase 2: Optimization Rounds (Turns 3-12)**

*   This phase consists of **exactly 5 Optimization Rounds**. Each round is 2 turns (Assistant then User).

3.  **Turns 3, 5, 7, 9, 11 (Assistant): Optimization Round N**
    *   Format: `<|im_start|>assistant\\n<think>...</think>\\n<tool_call>...</tool_call>\\n<|im_end|>`
    *   `<think>` Content (**ALL elements below are REQUIRED**):
        *   **Round Marker:** `[Optimization Round N/5]` (where N is 1 to 5).
        *   **State/Recap:**
            *   For Round 1: `Initial State:` mentioning the result (instruction count) from the Baseline Check.
            *   For Rounds 2-5: `Recap:` mentioning the passes *added* in the *previous* round's plan (use **Python list literal**). Must also include `Result:` mentioning the analysis outcome from the previous round (e.g., "Total InstCount decreased by X").
        *   **Counts:**
            *   `Current InstCount (Sequence):` showing the count *before* applying this round's new passes.
            *   For Rounds 2-5: `Baseline InstCount (-Oz):` showing the count achieved in the initial baseline check.
        *   **Plan:** Must include a line like `Plan (Round N): Add passes: ['--newPassA', '--newPassB', ...]` specifying the *new* passes to add in this round (use **Python list literal**). Optimization level flags (e.g., `-Oz`) can be included here alongside specific passes if desired.
        *   **Descriptions:** Provide brief descriptions for ONLY the *new* passes added in the Plan section (e.g., `- --pass: Description.`).
        *   **Cumulative Statement:** Mention the upcoming tool call uses the accumulated sequence.
        *   **EXACT Ending:** The think block MUST end *exactly* with the line: `\\nTool call analyzes the effect of applying the *cumulative* sequence generated so far (compared to previous round's state).` (Include the asterisk formatting).
    *   `<tool_call>` Content:
        *   Valid JSON: `{{"name": "analyze_autophase", "arguments": {{"filename": filename, "optimization_passes": [...]}}}}`.
        *   **Accumulation Rule:** The `"optimization_passes"` **JSON list** MUST contain *all* passes from the `Plan:` sections of Optimization Rounds 1 through N, concatenated in order. The baseline `-Oz` is NOT included in this accumulation unless explicitly added again in an optimization round's plan.

4.  **Turns 4, 6, 8, 10, 12 (User): Optimization Response N**
    *   Format: `<|im_start|>user\\n<tool_response>...</tool_response>\\n<|im_end|>`
    *   (You will receive this response, containing the instruction count after applying the cumulative passes up to round N).

**Phase 3: Final Decision (Turn 13)**

5.  **Turn 13 (Assistant): Final Decision and Answer**
    *   **Position:** MUST appear IMMEDIATELY after the 5th optimization round's user turn (Turn 12).
    *   **Format:** `<|im_start|>assistant\\n<think>...</think>\\n<answer>...</answer>\\n<|im_end|>`
    *   **CRITICAL:** This block contains BOTH `<think>` and `<answer>`. NO `<tool_call>`.
    *   `<think>` Content (**ALL elements below are REQUIRED**):
        *   **Marker:** `[Final Decision]`
        *   **Counts Mention:** State the `Final InstCount (Result of 5-Round Sequence):` (from Turn 12 response) AND the `Baseline InstCount (Result of Initial -Oz):` (from Turn 2 response).
        *   **Comparison:** Include a `Comparison:` line explicitly comparing the two counts and stating which is better (lower).
        *   **Conclusion:** Include a `Conclusion:` line stating the final choice based on the comparison.
    *   `<answer>` Content:
        *   Contains ONLY the final chosen pass sequence.
        *   **Logic:**
            *   If Baseline InstCount <= Final Sequence InstCount, the answer MUST be `['-Oz']`.
            *   If Final Sequence InstCount < Baseline InstCount, the answer MUST be the full accumulated list of passes from the 5th optimization round's `<tool_call>`.
        *   **Format:** Use **Python list literal** string format inside `<answer>` (e.g., `['-Oz']` or `['--pass1', '--pass2', ...]`).
    *   **Absolute End:** NO content or blocks whatsoever after this final assistant turn.

**Formatting Recap (JSON vs. Python List Literal):**
*   `<tool_call>` -> `optimization_passes`: **JSON List** `["--pass1", "-Oz"]`
*   `<think>` (Plan/Recap) -> Pass lists: **Python List Literal** `['--pass1', '-Oz']`
*   `<answer>` -> Final pass list: **Python List Literal** `['--pass1', '-Oz']`

Adhere strictly to the markers, required text, endings, accumulation logic, and final answer determination. Use standard pass descriptions or "General optimization pass." if unsure.
"""

    # 添加所有可用passes的信息到指令中
    # passes_info = "\n\n可用的优化passes:\n"
    # for i, pass_info in enumerate(all_passes):
    #     passes_info += f"{i}. {pass_info['pass']}\n"
    
    # instruction_following += passes_info

    # Process each data item
    def make_map_fn(split, val_source=None):
        def process_fn(example, idx):
            # Basic info
            filename = example.get('filename', '')
            ll_code = example.get('ll_code', '')
            autophase_features = example.get('autophase_features', '{}')
            pass_sequence = example.get('pass_sequence', [])
            overoz = example.get('overoz', [])
            
            # 解析autophase特征
            try:
                features_dict = json.loads(autophase_features)
            except:
                features_dict = {}
                
            # 创建特征表示并获取初始指令计数
            initial_inst_count = features_dict.get('TotalInsts', 'N/A')
            formatted_features = json.dumps(features_dict, indent=2)
            features_text = f"The LLVM IR code is represented by autophase features, the initial autophase features are:\n```json\n{formatted_features}\n```\n\nInitial instruction count: {initial_inst_count}\n"
            
            # Create prompt
            prompt = instruction_following + " \n"
            prompt += f"Filename for tool call reference: {filename}\n\n"
            prompt += features_text
            
            # 添加一个提示，告诉模型如何在tool_call中使用文件名
            prompt += f"\nNote: When calling the analyze_autophase tool, use the exact filename provided above: {filename}"
            
            # Create extra_info with validation source if applicable
            extra_info = {
                'split': split,
                'index': str(idx),
                'pass_sequence': pass_sequence,
                'overoz': overoz
            }
            
            # Add validation source information if provided
            if val_source:
                extra_info['validation_source'] = val_source
                
            # Create data record
            data = {
                "data_source": "compiler_autotuning",
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "compiler_autotuning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": filename 
                },
                "extra_info": extra_info
            }
            return data

        return process_fn

    # Apply the processing function for train and test
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    # Save datasets to parquet files
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    # Process and save each validation dataset
    for val_name, val_dataset in validation_datasets.items():
        processed_val_dataset = val_dataset.map(
            function=make_map_fn('validation', val_source=val_name), 
            with_indices=True
        )
        validation_filename = f'validation_{val_name}.parquet'
        processed_val_dataset.to_parquet(os.path.join(local_dir, validation_filename))
        print(f"Saved validation dataset '{val_name}' to {os.path.join(local_dir, validation_filename)}")
    
    print(f"Saved processed datasets to {local_dir}")
    
    # If HDFS directory is provided, copy the datasets there
    if args.hdfs_dir is not None:
        print(f"Copying datasets to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy completed")