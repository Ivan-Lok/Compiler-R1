#!/usr/bin/env python
# Copyright 2024 Test for Compiler Autotuning RAG
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
测试编译器自动调优RAG搜索工具
"""

import os
import json
import ast
import random
import argparse
import pandas as pd
from typing import Dict, List, Any

# 导入RAG系统类
from agent_r1.tool.tools.comiler_autotuning.rag_search_tool import RAGSearchTool

def format_pass_sequence(pass_sequence: List[str]) -> str:
    """
    格式化Pass序列，使其更易于阅读
    
    Args:
        pass_sequence: Pass序列列表
        
    Returns:
        格式化的Pass序列字符串
    """
    formatted = []
    for i, pass_name in enumerate(pass_sequence):
        formatted.append(f"{i+1}. {pass_name}")
    
    return "\n".join(formatted)

def test_tool_interface():
    """
    测试工具接口
    """
    print("====== 测试RAG搜索工具 ======")
    
    # 初始化工具
    try:
        rag_tool = RAGSearchTool()
    except Exception as e:
        print(f"初始化RAG搜索工具时出错: {e}")
        return
    
    # 获取CSV文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../../../../examples/data_preprocess/compiler_autotuning_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"找不到CSV文件: {csv_path}")
        return
    
    # 随机选择一个样本作为查询
    df = pd.read_csv(csv_path)
    random_index = random.randint(0, len(df) - 1)
    row = df.iloc[random_index]
    
    # 解析Autophase嵌入
    autophase_embedding = ast.literal_eval(row['Autophase_embedding'])
    ground_truth = ast.literal_eval(row['PassSequence'])
    
    print(f"随机选择的查询样本索引: {random_index}")
    print(f"文件名: {row['Filename']}")
    print(f"Ground Truth Pass序列:")
    print(format_pass_sequence(ground_truth))
    print("\n")
    
    # 执行工具
    args = {
        "autophase_embedding": autophase_embedding,
        "top_k": 3
    }
    
    try:
        result = rag_tool.execute(args)
        result_obj = json.loads(result)
        
        print("搜索结果:")
        if "results" in result_obj:
            for i, result_item in enumerate(result_obj["results"]):
                print(f"结果 #{i+1} (相似度: {result_item['similarity']:.4f}):")
                print(format_pass_sequence(result_item['pass_sequence']))
                print("\n")
        else:
            print(f"错误: {result_obj.get('error', '未知错误')}")
    except Exception as e:
        print(f"执行搜索时出错: {e}")
    
    # 测试奖励计算
    try:
        reward = rag_tool.calculate_reward(args, result)
        print(f"计算的奖励值: {reward:.4f}")
    except Exception as e:
        print(f"计算奖励时出错: {e}")

def test_batch_execute():
    """
    测试批量执行功能
    """
    print("\n====== 测试批量执行 ======")
    
    # 初始化工具
    try:
        rag_tool = RAGSearchTool()
    except Exception as e:
        print(f"初始化RAG搜索工具时出错: {e}")
        return
    
    # 获取CSV文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../../../../examples/data_preprocess/compiler_autotuning_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"找不到CSV文件: {csv_path}")
        return
    
    # 随机选择三个样本作为查询
    df = pd.read_csv(csv_path)
    indices = random.sample(range(len(df)), 3)
    
    args_list = []
    ground_truths = []
    
    for idx in indices:
        row = df.iloc[idx]
        autophase_embedding = ast.literal_eval(row['Autophase_embedding'])
        ground_truth = ast.literal_eval(row['PassSequence'])
        
        args_list.append({
            "autophase_embedding": autophase_embedding,
            "top_k": 1
        })
        ground_truths.append((idx, row['Filename'], ground_truth))
    
    # 执行批量搜索
    try:
        results = rag_tool.batch_execute(args_list)
        
        for i, (result, (idx, filename, ground_truth)) in enumerate(zip(results, ground_truths)):
            print(f"查询 #{i+1} (样本索引: {idx}, 文件名: {filename}):")
            print("Ground Truth:")
            print(format_pass_sequence(ground_truth))
            print("\n搜索结果:")
            
            result_obj = json.loads(result)
            if "results" in result_obj and len(result_obj["results"]) > 0:
                result_item = result_obj["results"][0]
                print(f"相似度: {result_item['similarity']:.4f}")
                print(format_pass_sequence(result_item['pass_sequence']))
            else:
                print(f"错误: {result_obj.get('error', '未知错误')}")
            
            print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"执行批量搜索时出错: {e}")

def test_dict_input():
    """
    测试字典格式输入的功能
    """
    print("\n====== 测试字典格式输入 ======")
    
    # 初始化工具
    try:
        rag_tool = RAGSearchTool()
    except Exception as e:
        print(f"初始化RAG搜索工具时出错: {e}")
        return
    
    # 获取CSV文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../../../../examples/data_preprocess/compiler_autotuning_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"找不到CSV文件: {csv_path}")
        return
    
    # 随机选择一个样本作为查询
    df = pd.read_csv(csv_path)
    random_index = random.randint(0, len(df) - 1)
    row = df.iloc[random_index]
    
    # 解析Autophase嵌入为列表
    autophase_list = ast.literal_eval(row['Autophase_embedding'])
    ground_truth = ast.literal_eval(row['PassSequence'])
    
    # 将列表转换为字典
    feature_names = [
        "TotalInsts", "IntInsts", "FloatInsts", "BranchInsts", "VectorInsts",
        "CallInsts", "PHIInsts", "AllocaInsts", "LoadInsts", "StoreInsts",
        "GetElemPtrInsts", "CmpInsts", "SelectInsts", "BBs", "CFGEdges",
        "CallsitesInvokes", "GlobalVars", "LocalVars", "DirectCalls", "IndirectCalls",
        "IntArgs", "FloatArgs", "Instructions", "BinaryOps", "UnaryOps",
        "SelectionsRets", "Insns", "Switches", "IntConstOperands", "FloatConstOperands",
        "ExternalSymOperands", "ConstantOperands", "MemIntrinsicInsts", "MemoryOperatedInsts",
        "MemoryAddressingInsts", "ExternalFunctions", "DynamicMemoryOps", "NonLocalLoads",
        "NonLocalStores", "UnknownInsts", "TextSize", "DataSize", "BSSSize",
        "FunctionCount", "DeadInsts", "NonTerminator", "DerivedVars", "TotalUses",
        "UniqueName", "UniqueConstants", "BasicBlocks", "FunctionMeanSize", "FunMaxSize",
        "FunMeanUses", "FunMaxUses", "IntConstZero", "FloatConstZero"
    ]
    
    autophase_dict = {}
    for i, feature_name in enumerate(feature_names):
        if i < len(autophase_list):
            autophase_dict[feature_name] = autophase_list[i]
    
    print(f"随机选择的查询样本索引: {random_index}")
    print(f"文件名: {row['Filename']}")
    print(f"Ground Truth Pass序列:")
    print(format_pass_sequence(ground_truth))
    
    print("\n字典格式的Autophase特征(前10个特征):")
    for i, (key, value) in enumerate(list(autophase_dict.items())[:10]):
        print(f"{key}: {value}")
    print("...")
    
    # 执行工具 - 使用字典格式
    print("\n使用字典格式执行RAG搜索:")
    dict_args = {
        "autophase_embedding": autophase_dict,
        "top_k": 3
    }
    
    try:
        dict_result = rag_tool.execute(dict_args)
        dict_result_obj = json.loads(dict_result)
        
        if "results" in dict_result_obj:
            for i, result_item in enumerate(dict_result_obj["results"]):
                print(f"结果 #{i+1} (相似度: {result_item['similarity']:.4f}):")
                print(format_pass_sequence(result_item['pass_sequence']))
                print()
        else:
            print(f"错误: {dict_result_obj.get('error', '未知错误')}")
    except Exception as e:
        print(f"执行搜索时出错: {e}")
    
    # 执行工具 - 使用列表格式进行比较
    print("\n使用列表格式执行RAG搜索(用于比较):")
    list_args = {
        "autophase_embedding": autophase_list,
        "top_k": 3
    }
    
    try:
        list_result = rag_tool.execute(list_args)
        list_result_obj = json.loads(list_result)
        
        if "results" in list_result_obj:
            for i, result_item in enumerate(list_result_obj["results"]):
                print(f"结果 #{i+1} (相似度: {result_item['similarity']:.4f}):")
                print(format_pass_sequence(result_item['pass_sequence']))
                print()
        else:
            print(f"错误: {list_result_obj.get('error', '未知错误')}")
    except Exception as e:
        print(f"执行搜索时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="测试编译器自动调优RAG搜索工具")
    parser.add_argument("--test_type", choices=["single", "batch", "dict", "all"], default="all",
                      help="要运行的测试类型")
    
    args = parser.parse_args()
    
    if args.test_type == "single" or args.test_type == "all":
        test_tool_interface()
    
    if args.test_type == "batch" or args.test_type == "all":
        test_batch_execute()
        
    if args.test_type == "dict" or args.test_type == "all":
        test_dict_input()

if __name__ == "__main__":
    main() 