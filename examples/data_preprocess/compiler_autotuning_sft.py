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
将编译器自动调优数据集预处理为SFT格式
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
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode
from agent_r1.tool.tools.comiler_autotuning.raw_tool.gen_pass_from_number import Actions_LLVM_10_0_0

# LLVM优化pass的功能描述，帮助生成思考过程
PASS_DESCRIPTIONS = {
    "--add-discriminators": "Add discriminators for better debug info.",
    "--adce": "Aggressively eliminate dead code.",
    "--aggressive-instcombine": "Aggressive instruction combining.",
    "--alignment-from-assumptions": "Optimize memory alignment based on assumptions.",
    "--always-inline": "Inline all always_inline functions.",
    "--argpromotion": "Promote arguments from byref to byval.",
    "--attributor": "Propagate attributes across modules.",
    "--barrier": "Place barriers before code generation.",
    "--bdce": "Bit-level dead code elimination.",
    "--break-crit-edges": "Break critical edges to simplify CFG.",
    "--simplifycfg": "Simplify the control flow graph.",
    "--callsite-splitting": "Split indirect call sites based on constants.",
    "--called-value-propagation": "Propagate called values at indirect call sites.",
    "--canonicalize-aliases": "Canonicalize aliases for better analysis.",
    "--consthoist": "Hoist constants to higher scopes.",
    "--constmerge": "Merge duplicate constants.",
    "--constprop": "Simple constant propagation.",
    "--coro-cleanup": "Remove coroutine scheduling remnants.",
    "--coro-early": "Early coroutine transformation.",
    "--coro-elide": "Remove unnecessary coroutine constructs.",
    "--coro-split": "Split coroutines into multiple functions.",
    "--correlated-propagation": "Propagate correlated value info.",
    "--cross-dso-cfi": "Cross-DSO control flow integrity.",
    "--deadargelim": "Remove unused function arguments.",
    "--dce": "Dead code elimination.",
    "--die": "Dead instruction elimination.",
    "--dse": "Dead store elimination.",
    "--reg2mem": "Convert registers to stack memory references.",
    "--div-rem-pairs": "Optimize division and remainder pairs.",
    "--early-cse-memssa": "Early CSE based on memory SSA.",
    "--early-cse": "Early common subexpression elimination.",
    "--elim-avail-extern": "Convert available external globals to definitions.",
    "--ee-instrument": "Instrument exception handling for stack space.",
    "--flattencfg": "Flatten the control flow graph.",
    "--float2int": "Optimize floating-point to integer computations.",
    "--forceattrs": "Force setting function attributes.",
    "--inline": "Inline function code at call sites.",
    "--insert-gcov-profiling": "Insert GCOV-compatible instrumentation.",
    "--gvn-hoist": "Hoist redundant expressions.",
    "--gvn": "Global value numbering.",
    "--globaldce": "Global dead code elimination.",
    "--globalopt": "Global variable optimization.",
    "--globalsplit": "Split global variables into fragments.",
    "--guard-widening": "Widen guard conditions.",
    "--hotcoldsplit": "Split hot and cold paths.",
    "--ipconstprop": "Interprocedural constant propagation.",
    "--ipsccp": "Interprocedural sparse conditional constant propagation.",
    "--indvars": "Canonicalize loop induction variables.",
    "--irce": "Inductive range check elimination.",
    "--infer-address-spaces": "Infer address spaces.",
    "--inferattrs": "Infer attributes for unknown functions.",
    "--inject-tli-mappings": "Inject target library info mappings.",
    "--instsimplify": "Remove redundant instructions.",
    "--instcombine": "Combine instructions into simpler forms.",
    "--instnamer": "Assign names to unnamed instructions.",
    "--jump-threading": "Thread conditional jumps.",
    "--lcssa": "Convert loops to loop-closed SSA form.",
    "--licm": "Move loop-invariant code out of loops.",
    "--libcalls-shrinkwrap": "Optimize library call wrappers.",
    "--load-store-vectorizer": "Vectorize adjacent loads and stores.",
    "--loop-data-prefetch": "Prefetch data in loops.",
    "--loop-deletion": "Delete useless loops.",
    "--loop-distribute": "Distribute loops for parallelism.",
    "--loop-fusion": "Fuse loops to reduce overhead.",
    "--loop-guard-widening": "Widen loop guard conditions.",
    "--loop-idiom": "Recognize and replace common idioms in loops.",
    "--loop-instsimplify": "Simplify instructions in loops.",
    "--loop-interchange": "Interchange nested loops.",
    "--loop-load-elim": "Eliminate redundant loads in loops.",
    "--loop-predication": "Convert branches in loops to selects.",
    "--loop-reroll": "Reroll unrolled loops.",
    "--loop-rotate": "Rotate loops for better execution.",
    "--loop-simplifycfg": "Simplify loop control flow graph.",
    "--loop-simplify": "Canonicalize loop form.",
    "--loop-sink": "Sink instructions in loops.",
    "--loop-reduce": "Loop strength reduction.",
    "--loop-unroll-and-jam": "Unroll and jam nested loops.",
    "--loop-unroll": "Unroll loops.",
    "--loop-unswitch": "Extract conditions from loops.",
    "--loop-vectorize": "Vectorize loops.",
    "--loop-versioning-licm": "Create loop versions for LICM.",
    "--loop-versioning": "Create multiple loop versions.",
    "--loweratomic": "Lower atomic instructions.",
    "--lower-constant-intrinsics": "Lower constant intrinsics.",
    "--lower-expect": "Lower llvm.expect intrinsics.",
    "--lower-guard-intrinsic": "Lower guard intrinsics.",
    "--lowerinvoke": "Lower invoke and unwind instructions.",
    "--lower-matrix-intrinsics": "Lower matrix operation intrinsics.",
    "--lowerswitch": "Lower switch instructions.",
    "--lower-widenable-condition": "Lower widenable conditions.",
    "--memcpyopt": "Optimize memory copy operations.",
    "--mergefunc": "Merge duplicate functions.",
    "--mergeicmps": "Merge consecutive compare instructions.",
    "--mldst-motion": "Move memory load/store operations.",
    "--sancov": "Instrument sanitizer coverage.",
    "--name-anon-globals": "Name anonymous global variables.",
    "--nary-reassociate": "Reassociate n-ary expressions.",
    "--newgvn": "New global value numbering.",
    "--pgo-memop-opt": "Profile-guided memory operation optimization.",
    "--partial-inliner": "Partially inline hot paths.",
    "--partially-inline-libcalls": "Partially inline library calls.",
    "--post-inline-ee-instrument": "Instrument exception handling after inlining.",
    "--functionattrs": "Infer function attributes.",
    "--mem2reg": "Convert memory references to registers.",
    "--prune-eh": "Remove unreachable exception handling code.",
    "--reassociate": "Reassociate expressions.",
    "--redundant-dbg-inst-elim": "Remove redundant debug instructions.",
    "--rpo-functionattrs": "Infer function attributes in reverse postorder.",
    "--rewrite-statepoints-for-gc": "Rewrite statepoints for garbage collection.",
    "--sccp": "Sparse conditional constant propagation.",
    "--slp-vectorizer": "Superword-level parallelism vectorization.",
    "--sroa": "Scalar replacement of aggregates.",
    "--scalarizer": "Convert vector operations to scalar operations.",
    "--separate-const-offset-from-gep": "Separate constant offsets from GEP instructions.",
    "--simple-loop-unswitch": "Simplified loop unswitching.",
    "--sink": "Sink instructions to their use points.",
    "--speculative-execution": "Speculatively execute instructions.",
    "--slsr": "Straight-line strength reduction.",
    "--strip-dead-prototypes": "Remove unused function prototypes.",
    "--strip-debug-declare": "Remove debug declarations.",
    "--strip-nondebug": "Remove all non-debug information.",
    "--strip": "Remove all symbol information.",
    "--tailcallelim": "Eliminate tail calls.",
    "--mergereturn": "Merge multiple return points into one."
}

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

def analyze_feature_changes(prev_features_dict, new_features_dict):
    """
    分析特征变化，为下一轮优化提供依据
    
    Args:
        prev_features_dict: 上一轮的特征（字典格式）
        new_features_dict: 新的特征（字典格式）
        
    Returns:
        特征变化分析
    """
    # 检查输入格式
    if not isinstance(prev_features_dict, dict) or not isinstance(new_features_dict, dict):
        return "无法比较特征变化：输入格式不正确"
    
    analysis = []
    try:
        # 找出所有变化的特征
        changed_features = []
        
        # 检查两个字典中共有的键
        common_keys = set(prev_features_dict.keys()) & set(new_features_dict.keys())
        for key in common_keys:
            if prev_features_dict[key] != new_features_dict[key]:
                change = new_features_dict[key] - prev_features_dict[key]
                direction = "increase" if change > 0 else "decrease"
                changed_features.append((key, abs(change), direction, change))
        
        # 按特征名称排序
        changed_features.sort(key=lambda x: x[0])
        
        # 生成分析文本：直接输出所有变化的特征
        for feature, change_abs, direction, change in changed_features:
            analysis.append(f"{feature}: {prev_features_dict[feature]} -> {new_features_dict[feature]} ({direction} {change_abs})")
            
        # 添加TotalInsts的变化情况
        if "TotalInsts" in common_keys:
            total_insts_change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
            if total_insts_change > 0:
                analysis.append(f"Total InstCount increased by {total_insts_change}")
            elif total_insts_change < 0:
                analysis.append(f"Total InstCount decreased by {abs(total_insts_change)}")
            else:
                analysis.append("Total InstCount unchanged")

        # 特殊情况：无显著变化
        if not analysis:
            # 检查总指令数
            if "TotalInsts" in common_keys:
                change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
                if change != 0:
                    direction = "increase" if change > 0 else "decrease"
                    analysis.append(f"Total InstCount {direction} by {abs(change)}")
                else:
                    analysis.append("Feature changes are not obvious")
            else:
                analysis.append("Feature changes are not obvious")
                
    except Exception as e:
        analysis.append(f"Feature analysis error: {e}")
    
    return ", ".join(analysis) if analysis else "Feature changes are not obvious"

# def generate_thinking_process(filename, initial_autophase, pass_sequence):
#     """
#     生成多轮思考过程，每几个优化pass为一轮并调用工具
    
#     Args:
#         filename: 文件名
#         initial_autophase: 初始特征
#         pass_sequence: 优化pass序列
        
#     Returns:
#         完整的思考和工具调用过程
#     """
#     if not pass_sequence:
#         return ""
        
#     result = ""
#     current_autophase = initial_autophase
#     ll_file_path = os.path.join(os.path.join(os.path.dirname(__file__), 
#                                               './llvmir_datasets/'), filename)
#     ll_code = read_llvm_ir_file(ll_file_path)
    
#     if len(pass_sequence) > 30:
#         pass_sequence = pass_sequence[:30]

#     # 确定每轮使用的pass数量
#     passes_per_round = 6  # 每轮使用6个pass
#     if len(pass_sequence) <= 5:  # 如果总数很少，每轮使用1个
#         passes_per_round = 1
#     elif len(pass_sequence) <= 10:  # 如果总数适中，每轮使用2个
#         passes_per_round = 2
#     elif len(pass_sequence) <= 15:  # 如果总数适中，每轮使用3个
#         passes_per_round = 3
#     elif len(pass_sequence) <= 20:  # 如果总数适中，每轮使用4个
#         passes_per_round = 4
#     elif len(pass_sequence) <= 25:  # 如果总数适中，每轮使用5个
#         passes_per_round = 5
    
#     # 计算总轮数(向上取整)
#     total_rounds = (len(pass_sequence) + passes_per_round - 1) // passes_per_round
    
#     # 为每轮生成思考和工具调用
#     for round_idx in range(total_rounds):
#         # 确定当前轮次的起始和结束索引
#         start_idx = round_idx * passes_per_round
#         end_idx = min(start_idx + passes_per_round, len(pass_sequence))
        
#         # 当前轮次的passes
#         current_round_passes = pass_sequence[start_idx:end_idx]
#         all_passes_so_far = pass_sequence[:end_idx]
        
#         # 生成思考内容
#         thinking = f"<think>\n"
        
#         if round_idx == 0:
#             thinking += f"Based on the initial autophase features, I'll start with: {current_round_passes}\n"
#             # 添加对初始pass的解释
#             for pass_name in current_round_passes:
#                 if pass_name in PASS_DESCRIPTIONS:
#                     thinking += f"- {pass_name}: {PASS_DESCRIPTIONS[pass_name]}\n"
#         else:
#             thinking += f"Let me analyze how the features have changed after applying previous passes and select more passes.\n"
#             thinking += f"Next passes to apply: {current_round_passes}\n"
#             # 添加对新选择pass的解释
#             for pass_name in current_round_passes:
#                 if pass_name in PASS_DESCRIPTIONS:
#                     thinking += f"- {pass_name}: {PASS_DESCRIPTIONS[pass_name]}\n"
        
#         thinking += "</think>\n"
        
#         # 生成工具调用 - 使用合并后的工具
#         tool_call = f"<tool_call>\n"
#         tool_call += f'{{"name": "analyze_autophase", "arguments": {{"filename": "{filename}", "optimization_passes": {json.dumps(all_passes_so_far)}}}}}\n'
#         tool_call += "</tool_call>\n\n"
        
#         try:
#             # 更新当前autophase并生成工具响应
#             llvm_tools_path = os.path.join(os.path.dirname(__file__), 
#                                         '../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')
#             # 获取原始特征
#             original_features = get_autophase_features(ll_code)
            
#             # 应用优化并获取优化后特征
#             optimized_code = GenerateOptimizedLLCode(ll_code, all_passes_so_far, llvm_tools_path=llvm_tools_path)
#             optimized_features = get_autophase_features(optimized_code)
            
#             # 分析特征变化
#             feature_analysis = analyze_feature_changes(original_features, optimized_features)
            
#             # 构建工具响应
#             tool_response = f"<tool_response>\n"
#             tool_response += json.dumps({
#                 "feature_analysis": feature_analysis,
#                 "status": "success"
#             }, indent=2)
#             tool_response += "\n</tool_response>\n\n"
            
#             # 更新当前代码和特征
#             ll_code = optimized_code
#             current_autophase = optimized_features
#         except Exception as e:
#             # 处理优化过程中可能出现的错误
#             tool_response = f"<tool_response>\n"
#             tool_response += json.dumps({
#                 "feature_analysis": f"Error applying optimization passes: {str(e)}",
#                 "status": "error"
#             }, indent=2)
#             tool_response += "\n</tool_response>\n\n"
        
#         # 添加到结果
#         result += thinking + tool_call + tool_response
    
#     # 生成最终答案
#     answer = "<answer>\n"
#     answer += f"{pass_sequence}\n"
#     answer += "</answer>"

#     return result + answer

def generate_thinking_process(filename, initial_autophase, pass_sequence):
    """
    Generates a multi-round thinking process with dynamic feedback.

    Args:
        filename: LLVM IR filename.
        initial_autophase: Initial autophase features dictionary.
        pass_sequence: The known good optimization pass sequence (list of strings).

    Returns:
        A string containing the full SFT sample (thinking, calls, responses, answer).
    """
    result = ""
    # --- Initial Setup ---
    if not filename or initial_autophase is None or not isinstance(pass_sequence, list):
         print("Error: Invalid input to generate_thinking_process.")
         return "<error>Invalid input provided.</error>"

    # Ensure pass_sequence contains strings
    pass_sequence = [str(p) for p in pass_sequence if p]
    if not pass_sequence:
        return "<error>Empty pass sequence provided.</error>"

    ll_file_path = os.path.join(os.path.dirname(__file__), './llvmir_datasets/', filename)
    original_ll_code = read_llvm_ir_file(ll_file_path)
    if original_ll_code is None:
         return f"<error>Failed to read original LLVM IR file: {ll_file_path}</error>"

    # Use a dummy path if running standalone without the full structure
    llvm_tools_path = os.path.join(os.path.dirname(__file__), '../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')

    # Sequence Truncation (optional, consider if needed)
    MAX_PASSES = 50
    if len(pass_sequence) > MAX_PASSES:
        print(f"Warning: Pass sequence truncated from {len(pass_sequence)} to {MAX_PASSES}")
        pass_sequence = pass_sequence[:MAX_PASSES]

    # --- Dynamic Passes Per Round ---
    n_passes = len(pass_sequence)
    if n_passes <= 5: passes_per_round = 1
    elif n_passes <= 10: passes_per_round = 2
    elif n_passes <= 15: passes_per_round = 3
    elif n_passes <= 20: passes_per_round = 4
    elif n_passes <= 25: passes_per_round = 5
    elif n_passes <= 30: passes_per_round = 6
    elif n_passes <= 35: passes_per_round = 7
    elif n_passes <= 40: passes_per_round = 8
    elif n_passes <= 45: passes_per_round = 9
    elif n_passes <= 50: passes_per_round = 10
    elif n_passes <= 55: passes_per_round = 11
    elif n_passes <= 60: passes_per_round = 12
    elif n_passes <= 65: passes_per_round = 13
    elif n_passes <= 70: passes_per_round = 14
    elif n_passes <= 75: passes_per_round = 15
    elif n_passes <= 80: passes_per_round = 16
    elif n_passes <= 85: passes_per_round = 17
    elif n_passes <= 90: passes_per_round = 18
    elif n_passes <= 95: passes_per_round = 19
    elif n_passes <= 100: passes_per_round = 20
    else: passes_per_round = 5 # Default for longer sequences

    total_rounds = (n_passes + passes_per_round - 1) // passes_per_round

    # --- State Variables for Loop ---
    previous_analysis_text = "Initial state, no analysis yet."
    # current_features holds the features *before* the current round's passes are applied
    current_features = initial_autophase

    # --- Loop Through Rounds ---
    for round_idx in range(total_rounds):
        start_idx = round_idx * passes_per_round
        end_idx = min(start_idx + passes_per_round, n_passes)
        current_round_passes = pass_sequence[start_idx:end_idx]
        # all_passes_so_far represents the state *after* this round is completed
        all_passes_so_far = pass_sequence[:end_idx]

        # --- 1. Generate Thinking ---
        thinking = "<think>\n"
        if round_idx == 0:
            thinking += f"Starting optimization based on initial features.\n"
            init_insts = initial_autophase.get('TotalInsts', 'N/A')
            thinking += f"Initial instruction count: {init_insts}.\n"
            thinking += f"Applying first batch of passes: {current_round_passes}\n"
            for pass_name in current_round_passes:
                thinking += f"- {pass_name}: {PASS_DESCRIPTIONS.get(pass_name, 'No description available.')}\n"
        else:
            thinking += f"Reviewing previous round's results before applying next passes.\n"
            # Add dynamic feedback based on previous_analysis_text
            # thinking += f"Previous analysis: {previous_analysis_text}\n" # Include the raw analysis text
            # Add interpretive sentences based on the analysis
            if "Total InstCount decreased" in previous_analysis_text:
                 thinking += f"Feedback: The previous passes were effective, reducing the total instruction count. Continuing optimization.\n"
            elif "Total InstCount increased" in previous_analysis_text:
                 thinking += f"Feedback: The previous passes increased the total instruction count, possibly due to code expansion strategies like inlining or unrolling. Evaluating overall impact.\n"
            elif "Total InstCount unchanged" in previous_analysis_text:
                 thinking += f"Feedback: The total instruction count remained largely unchanged. Proceeding with the next set of passes.\n"
            elif "Error" in previous_analysis_text or "无法" in previous_analysis_text:
                 thinking += f"Feedback: There was an issue analyzing the previous step. Proceeding cautiously.\n"
            else: # Default if no clear signal
                 thinking += f"Feedback: Evaluating the effect of previous passes. Proceeding with the next set.\n"

            thinking += f"Upon above analysis, Next passes i will apply: {current_round_passes}\n"
            for pass_name in current_round_passes:
                thinking += f"- {pass_name}: {PASS_DESCRIPTIONS.get(pass_name, 'No description available.')}\n"
        thinking += "</think>\n"

        # --- 2. Generate Tool Call ---
        # Tool call uses ALL passes applied up to the end of this round
        tool_call = f"<tool_call>\n"
        tool_call += f'''{{"name": "analyze_autophase","arguments": {{"filename": "{filename}","optimization_passes": {json.dumps(all_passes_so_far)}}}}}'''
        tool_call += "\n</tool_call>\n\n"

        # --- 3. Simulate Tool Execution & Get Response ---
        tool_response_content = {"status": "error", "feature_analysis": "Simulation failed"}
        feature_analysis_this_round = "Error during simulation"
        try:
            # Apply ALL passes so far to the ORIGINAL code to get the state *after* this round
            optimized_code_this_round = GenerateOptimizedLLCode(original_ll_code, all_passes_so_far, llvm_tools_path=llvm_tools_path)
            if optimized_code_this_round is None:
                raise ValueError("GenerateOptimizedLLCode returned None")

            optimized_features_this_round = get_autophase_features(optimized_code_this_round)
            if optimized_features_this_round is None:
                 raise ValueError("get_autophase_features returned None")

            # Analyze changes: compare features *before* this round (current_features)
            # with features *after* this round (optimized_features_this_round)
            feature_analysis_this_round = analyze_feature_changes(current_features, optimized_features_this_round)

            tool_response_content["status"] = "success"
            tool_response_content["feature_analysis"] = feature_analysis_this_round
            # Optionally include the new instruction count if available
            new_inst_count = optimized_features_this_round.get("TotalInsts")
            if new_inst_count is not None:
                 tool_response_content["current_total_insts"] = new_inst_count

            # --- Update State for NEXT iteration's thinking phase ---
            previous_analysis_text = feature_analysis_this_round # Store analysis of this round
            current_features = optimized_features_this_round # Update features for next comparison

        except Exception as e:
            print(f"Error during round {round_idx + 1} simulation for {filename}: {e}")
            error_message = f"Error applying or analyzing passes ({all_passes_so_far}): {str(e)}"
            tool_response_content["feature_analysis"] = error_message
            previous_analysis_text = error_message # Update analysis text with error for next round context
            # Keep current_features as they were from the last successful state

        # Format the tool response
        tool_response = f"<tool_response>\n"
        tool_response += json.dumps(tool_response_content, indent=2)
        tool_response += "\n</tool_response>\n\n"

        # Append round results to the final SFT sample string
        result += thinking + tool_call + tool_response

        # Stop if error occurred? For SFT generation, maybe continue to show error handling.
        if tool_response_content["status"] == "error" or "Invalid" in tool_response:
            print(f"Stopping generation for {filename} due to error tool calling.")
            # Optionally add a final error message before the answer
            result += "<error>Optimization process interrupted due to error.</error>\n\n"
            break # Exit the loop

    # --- 4. Final Answer ---
    # The final answer is the complete "good" sequence that was processed.
    answer = "<answer>\n"
    answer += json.dumps(pass_sequence) # Output the sequence as a JSON list string
    answer += "\n</answer>"

    return result + answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='compiler_autotuning_data.csv',
                        help='Path to the compiler autotuning data CSV file')
    parser.add_argument('--llvm_ir_dir', default=None, 
                        help='Directory containing LLVM IR files (optional)')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning_sft',
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

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 加载数据集
    print(f"Loading compiler autotuning data from {args.data_file}...")
    
    # 确定CSV文件的完整路径
    if os.path.isabs(args.data_file):
        csv_path = args.data_file
    else:
        # 如果是相对路径，检查它是否在当前目录
        if os.path.exists(args.data_file):
            csv_path = args.data_file
        # 检查它是否在与此脚本相同的目录
        elif os.path.exists(os.path.join(os.path.dirname(__file__), args.data_file)):
            csv_path = os.path.join(os.path.dirname(__file__), args.data_file)
        else:
            raise FileNotFoundError(f"Could not find {args.data_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 限制样本数量（如果需要）
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
    
    print(f"Loaded {len(df)} samples")
    
    # 处理数据帧
    data_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        # 提取文件名
        filename = row['Filename']
        
        # 解析Autophase_embedding和PassSequence
        try:
            ll_code = None
            if args.llvm_ir_dir is not None:
                ll_file_path = os.path.join(args.llvm_ir_dir, filename)
                ll_code = read_llvm_ir_file(ll_file_path)

            # 获取Autophase特征和pass序列
            autophase_embedding = get_autophase_features(ll_code)
            initial_inst_count = autophase_embedding.get('TotalInsts', 'N/A')
            formatted_features = json.dumps(autophase_embedding, indent=2)
            pass_sequence = ast.literal_eval(row['PassSequence'])
            over_oz = float(row['OverOz'])
            
            # 为SFT构造问题和答案
            question = f"""Act as a compiler optimization expert simulating the process of finding an optimal pass sequence for LLVM IR. Your goal is to reduce the total instruction count.

The LLVM IR code is represented by autophase features:
```json
{formatted_features}
```

Initial instruction count: {initial_inst_count}

Your task is to simulate the process of finding a good optimization sequence using <think>, <tool_call>, and <tool_response> steps. The goal is to minimize the final instruction count.

1. Analyze the initial features in `<think>`.
2. Choose a batch of LLVM optimization passes based on the analysis and previous results (if any) in `<think>`.
3. Make a `<tool_call>` to `analyze_autophase` with the cumulative pass sequence applied so far.
4. Use the feature analysis from `<tool_response>` to inform the next `<think>` step.
5. Repeat steps 1-4 iteratively following the provided successful trajectory.
6. Finally, output the complete target pass sequence (the one used in the final tool call of the provided successful trajectory) in `<answer>`.

Filename for reference: {filename}
"""
            
            # 生成思考过程和工具调用
            full_process = generate_thinking_process(filename, autophase_embedding, pass_sequence)
            
            # 创建记录
            record = {
                'question': question,
                'answer': full_process,
                'filename': filename,
                'autophase_embedding': autophase_embedding,
                'pass_sequence': pass_sequence,
                'over_oz': over_oz
            }

            # if len(record['question'] + record['answer']) >= 8192:
            #     print(len(record['question'] + record['answer']))
            
            data_records.append(record)
        except Exception as e:
            print(f"Error processing row for {filename}: {e}")
            continue
    
    # 创建数据集
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data_records))
    
    # 拆分数据集
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
    
    # 构造SFT格式的数据
    def process_for_sft(example):
        return {
            "extra_info": {
                "question": example["question"],
                "answer": example["answer"]
            },
            "data_source": "compiler_autotuning",
            "ability": "compiler_autotuning"
        }
    
    train_dataset = train_dataset.map(process_for_sft)
    validation_dataset = validation_dataset.map(process_for_sft)
    test_dataset = test_dataset.map(process_for_sft)
    
    # 保存数据集
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Saved processed datasets to {local_dir}")
    
    # 如果提供了HDFS目录，将数据集复制到那里
    if args.hdfs_dir is not None:
        print(f"Copying datasets to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy completed")

if __name__ == '__main__':
    main() 