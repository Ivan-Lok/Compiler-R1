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
    parser.add_argument('--llvm_ir_dir', default=None, 
                        help='Directory containing LLVM IR files (optional)')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning',
                        help='Local directory to save the processed data')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to save the processed data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of data to use for validation')
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

    # Load the dataset
    print(f"Loading compiler autotuning data from {args.data_file}...")
    
    # Determine the full path to the CSV file
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
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Limit the number of samples if needed
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
    
    print(f"Loaded {len(df)} samples")
    
    # 获取所有可用的优化passes
    all_passes = get_all_passes()
    print(f"Found {len(all_passes)} available optimization passes")
    
    # Process the dataframe
    data_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        # Extract filename
        filename = row['Filename']
        
        # Read LLVM IR code if directory is provided
        ll_code = None
        if args.llvm_ir_dir is not None:
            ll_file_path = os.path.join(args.llvm_ir_dir, filename)
            ll_code = read_llvm_ir_file(ll_file_path)
        
        if ll_code:
            # 计算初始的autophase特征
            initial_features = get_autophase_features(ll_code)
            
            if initial_features:
                # Create record
                record = {
                    'filename': filename,
                    'll_code': ll_code,  # 保留原始代码，用于后续计算overOz
                    'autophase_features': json.dumps(initial_features)
                }
                
                data_records.append(record)
            else:
                print(f"Warning: Failed to get autophase features for {filename}, skipping")
        else:
            print(f"Warning: Failed to read {filename}, skipping")
    
    # Create dataset
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data_records))
    
    # Split the dataset
    splits = dataset.train_test_split(
        test_size=args.val_ratio + args.test_ratio,
        seed=args.seed
    )
    train_dataset = splits['train']
    
    val_test_splits = splits['test'].train_test_split(
        test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
        seed=args.seed
    )
    validation_dataset = val_test_splits['train']
    test_dataset = val_test_splits['test']
    
    print(f"Dataset split: {len(train_dataset)} train, {len(validation_dataset)} validation, {len(test_dataset)} test")
    
    # Instruction template
    instruction_following = """优化给定的LLVM IR代码以减少指令数量。

由于LLVM IR代码太长，我们使用autophase特征来代表代码。这些特征捕捉了代码的关键统计特性。

你的任务是：
1. 分析初始autophase特征
2. 在<think>...</think>中思考，选择一个LLVM优化pass
3. 使用gen_autophase工具，将选择的pass应用到代码上，并获取新的autophase特征
4. 分析新的特征，决定下一个要应用的pass
5. 重复步骤2-4，重复至少10轮，构建一个有效的优化pass序列
6. 最后在<answer>中输出完整的优化pass列表

可用的工具:
1. gen_autophase: 生成应用优化passes后的autophase特征
   {"name": "gen_autophase", "arguments": {"filename": "文件名", "optimization_passes": ["--adce", "--inline"]}}

每轮交互的格式：
<think>
分析当前的autophase特征，我认为应该使用XX优化...（选择一个优化pass）
</think>
<tool_call>
{"name": "gen_autophase", "arguments": {"filename": "文件名", "optimization_passes": ["之前选择的pass", "新选择的pass"]}}
</tool_call>

最终回答的格式:
<think>
经过多轮优化后，我生成了以下优化pass序列...
</think>
<answer>
[优化pass的列表]
</answer>

最终答案必须是一个包含所有应用过的优化pass的列表，按应用顺序排列。系统将基于这个答案计算相对于-Oz的指令数减少比例。
注意：
1. 为一个程序找到好的pass序列是需要多探索的，所以根据autophase特征请多尝试不同的优化pass，不要放弃。
"""

    # 添加所有可用passes的信息到指令中
    passes_info = "\n\n可用的优化passes:\n"
    for i, pass_info in enumerate(all_passes):
        passes_info += f"{i}. {pass_info['pass']}\n"
    
    instruction_following += passes_info

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            # Basic info
            filename = example.get('filename', '')
            ll_code = example.get('ll_code', '')
            autophase_features = example.get('autophase_features', '{}')
            pass_sequence = example.get('pass_sequence', [])
            
            # 解析autophase特征
            try:
                features_dict = json.loads(autophase_features)
            except:
                features_dict = {}
                
            # 创建特征表示
            features_text = "Autophase特征:\n"
            for key, value in features_dict.items():
                features_text += f"- {key}: {value}\n"
            
            # Create prompt
            prompt = instruction_following + "\n\n初始代码信息:\n"
            prompt += f"文件名: {filename}\n\n"
            prompt += features_text
            
            # 添加一个提示，告诉模型如何在tool_call中使用文件名
            prompt += f"\n注意：当调用gen_autophase工具时，请使用上面提供的文件名({filename})作为filename参数。"
            
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
                    "ground_truth": ll_code  # 保存原始LLVM IR代码作为ground truth，用于计算overOz
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'filename': filename,
                    'pass_sequence': pass_sequence
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