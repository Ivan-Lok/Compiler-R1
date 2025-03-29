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
    parser.add_argument('--val_file', default='cbench-val.csv',
                        help='Path to the validation data CSV file')
    parser.add_argument('--llvm_ir_dir', default=None, 
                        help='Directory containing LLVM IR files (optional)')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning',
                        help='Local directory to save the processed data')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to save the processed data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
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
    
    # Load the validation dataset
    print(f"Loading validation data from {args.val_file}...")
    
    # Determine the full path to the validation CSV file
    if os.path.isabs(args.val_file):
        val_csv_path = args.val_file
    else:
        # If it's a relative path, check if it's in the current directory
        if os.path.exists(args.val_file):
            val_csv_path = args.val_file
        # Check if it's in the same directory as this script
        elif os.path.exists(os.path.join(os.path.dirname(__file__), args.val_file)):
            val_csv_path = os.path.join(os.path.dirname(__file__), args.val_file)
        else:
            raise FileNotFoundError(f"Could not find {args.val_file}")
    
    # Read the validation CSV file
    val_df = pd.read_csv(val_csv_path)
    
    print(f"Loaded {len(main_df)} main samples and {len(val_df)} validation samples")
    
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
    
    # Process validation dataset
    val_records = process_dataframe(val_df, args.llvm_ir_dir)
    val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(val_records))
    
    # Split the main dataset into train and test
    splits = main_dataset.train_test_split(
        test_size=args.test_ratio,
        seed=args.seed
    )
    train_dataset = splits['train']
    test_dataset = splits['test']
    
    # Use the dedicated validation dataset
    validation_dataset = val_dataset
    
    print(f"Dataset split: {len(train_dataset)} train, {len(validation_dataset)} validation, {len(test_dataset)} test")
    
    # Instruction template
    instruction_following = """Act as a compiler optimization expert working to find an optimal pass sequence for LLVM IR. Your goal is to reduce the total instruction count.

The LLVM IR code is represented by autophase features due to its length. These features capture key statistical properties of the code.

Your task is to:
1. Analyze the initial autophase features in <think>.
2. Choose LLVM optimization passes based on the analysis in <think>.
3. Make a tool call to `analyze_autophase` with the selected passes.
4. Review the tool response to understand the impact of your optimization choices.
5. Repeat steps 1-4 to iteratively build an effective optimization pass sequence.
6. Finally, output the complete list of optimization passes in <answer>.

The format for each round of interaction:
<think>
Analyze the features and explain your reasoning for selecting specific optimization passes.
</think>

<tool_call>
{"name": "analyze_autophase", "arguments": {"filename": "example.ll", "optimization_passes": ["--pass1", "--pass2", ...]}}
</tool_call>

<tool_response>
The tool will return analysis of how the features changed after applying your passes.
</tool_response>

The format for the final answer:
<answer>
[Complete list of all optimization passes you've selected]
</answer>

Remember to use the exact filename provided when making tool calls. The final answer must be a JSON-formatted list of all applied optimization passes enclosed in square brackets.
"""

    # 添加所有可用passes的信息到指令中
    # passes_info = "\n\n可用的优化passes:\n"
    # for i, pass_info in enumerate(all_passes):
    #     passes_info += f"{i}. {pass_info['pass']}\n"
    
    # instruction_following += passes_info

    # Process each data item
    def make_map_fn(split):
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
            features_text = f"The LLVM IR code is represented by autophase features:\n```json\n{formatted_features}\n```\n\nInitial instruction count: {initial_inst_count}\n"
            
            # Create prompt
            prompt = instruction_following + "\n\nInitial code information:\n"
            prompt += f"Filename for reference: {filename}\n\n"
            prompt += features_text
            
            # 添加一个提示，告诉模型如何在tool_call中使用文件名
            prompt += f"\nNote: When calling the analyze_autophase tool, use the exact filename provided above: {filename}"

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
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    # 'filename': filename,
                    'pass_sequence': pass_sequence,
                    'overoz' : overoz
                }
            }
            return data

        return process_fn

    # Apply the processing function
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    # Save datasets to parquet files
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Saved processed datasets to {local_dir}")
    
    # If HDFS directory is provided, copy the datasets there
    if args.hdfs_dir is not None:
        print(f"Copying datasets to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy completed")