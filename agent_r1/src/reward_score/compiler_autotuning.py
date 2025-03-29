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

import re
import os
import json
from typing import List, Union, Optional, Dict, Any, Tuple
import torch
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_instrcount, get_overOz
from agent_r1.tool.tools.comiler_autotuning.gen_autophase_tool import GenAutophaseTool
from agent_r1.tool.tools.comiler_autotuning.list_passes_tool import ListPassesTool

# 全局工具实例
gen_autophase_tool = GenAutophaseTool()
list_passes_tool = ListPassesTool()

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


def extract_answer(solution_str: str) -> str:
    """Extract the answer from the solution string.
    
    Args:
        solution_str: Solution text
        
    Returns:
        Extracted answer, or None if not found
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def extract_think_content(assistant_block: str) -> Optional[str]:
    """从助手块中提取think内容。
    
    Args:
        assistant_block: 助手回复块
        
    Returns:
        think内容，如果没有找到则返回None
    """
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, assistant_block, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def extract_passes_from_think(think_content: str) -> Optional[List[str]]:
    """从think内容中提取格式化的优化pass列表。
    
    Args:
        think_content: think内容
        
    Returns:
        提取的优化pass列表，如果没有找到则返回None
    """
    # 查找格式为["--xxx", "--yyy"]或['--xxx', '--yyy']的pass列表
    pass_list_pattern = r'\[((?:"--[a-zA-Z0-9-]+"|\'--[a-zA-Z0-9-]+\')(?:\s*,\s*(?:"--[a-zA-Z0-9-]+"|\'--[a-zA-Z0-9-]+\'))*)\]'
    match = re.search(pass_list_pattern, think_content)
    
    if match:
        # 提取列表内容
        passes_str = match.group(1)
        # 分割并清理引号
        passes = []
        for p in re.findall(r'(?:"(--[a-zA-Z0-9-]+)"|\'(--[a-zA-Z0-9-]+)\')', passes_str):
            # 每个匹配项是一个元组，取非空的那个
            pass_item = p[0] if p[0] else p[1]
            passes.append(pass_item)
        return passes
    return None

# def extract_pass_from_think(think_content: str) -> Optional[List[str]]:
#     """从think内容中提取优化pass。
    
#     Args:
#         think_content: think内容
        
#     Returns:
#         提取的优化pass列表，如果没有找到则返回None
#     """
#     # 尝试匹配"--"开头的优化pass
#     pass_patterns = [
#         r'使用(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
#         r'选择(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
#         r'应用(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
#         r'(?:优化pass|pass|优化选项)[：:]\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
#         r'["\'](-{2}[a-zA-Z0-9-]+)["\']',
#         r'(-{2}[a-zA-Z0-9-]+)'
#     ]
    
#     for pattern in pass_patterns:
#         match = re.search(pattern, think_content, re.IGNORECASE)
#         if match:
#             return [match.group(1)]
    
#     # 如果上面的模式都没匹配到，尝试查找所有--开头的字符串
#     all_passes = re.findall(r'(-{2}[a-zA-Z0-9-]+)', think_content)
#     if all_passes:
#         return all_passes
            
#     return None

def extract_tool_call_data(assistant_block: str) -> Optional[Dict[str, Any]]:
    """提取工具调用数据。
    
    Args:
        assistant_block: 助手回复块
        
    Returns:
        工具调用数据字典，如果未找到则返回None
    """
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(tool_call_pattern, assistant_block, re.DOTALL)
    
    if not match:
        return None
        
    tool_call_content = match.group(1).strip()
    
    # 尝试解析JSON格式的工具调用
    try:
        tool_data = json.loads(tool_call_content)
        return tool_data
    except json.JSONDecodeError:
        # 如果不是JSON格式，尝试正则表达式匹配
        name_pattern = r'name\s*[=:]\s*["\']?([^"\']+)["\']?'
        args_pattern = r'arguments\s*[=:]\s*({.*?})'
        
        name_match = re.search(name_pattern, tool_call_content)
        args_match = re.search(args_pattern, tool_call_content, re.DOTALL)
        
        if name_match:
            tool_data = {"name": name_match.group(1)}
            
            if args_match:
                try:
                    tool_data["arguments"] = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    # 尝试提取优化passes列表
                    passes_pattern = r'optimization_passes\s*[=:]\s*(\[.*?\])'
                    passes_match = re.search(passes_pattern, args_match.group(1), re.DOTALL)
                    if passes_match:
                        try:
                            tool_data["arguments"] = {
                                "optimization_passes": json.loads(passes_match.group(1))
                            }
                        except json.JSONDecodeError:
                            pass
            
            return tool_data
    
    return None

def extract_passes_from_tool_call(tool_data: Dict[str, Any]) -> List[str]:
    """从工具调用数据中提取优化passes。
    
    Args:
        tool_data: 工具调用数据
        
    Returns:
        优化passes列表
    """
    if not tool_data or tool_data.get('name') != 'gen_autophase':
        return []
    
    arguments = tool_data.get('arguments', {})
    if not arguments:
        return []
    
    passes = arguments.get('optimization_passes', [])
    if isinstance(passes, list):
        return passes
    
    return []

def parse_optimization_sequence(sequence_str: str) -> List[str]:
    """Parse the optimization sequence string into a list.
    
    Args:
        sequence_str: Optimization sequence string from <answer> tag
        
    Returns:
        List of optimization options
    """
    if not sequence_str:
        return []
        
    try:
        # Try to parse as JSON directly
        return json.loads(sequence_str)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON array pattern
        json_array_pattern = r'\[(.*?)\]'
        match = re.search(json_array_pattern, sequence_str, re.DOTALL)
        if match:
            try:
                # Try to parse the extracted content
                return json.loads(f"[{match.group(1)}]")
            except json.JSONDecodeError:
                pass
                
        # If JSON parsing fails, try to extract individual optimization passes
        passes = re.findall(r'--[a-zA-Z0-9-]+', sequence_str)
        if passes:
            return passes
            
    return []

def compute_score_format(solution_str: str) -> float:
    """Compute the format reward score based on SFT format.
    
    Args:
        solution_str: Solution text
        
    Returns:
        Format reward score (0.0-10.0)
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Extract assistant blocks
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            return 0.0
        
        total_reward = 0.0
        total_think_blocks = 0
        total_tool_call_blocks = 0
        total_tool_response_blocks = 0
        
        # Check conversation format according to SFT structure
        for i, block in enumerate(assistant_blocks):
            # Count <think> blocks
            think_blocks = re.findall(r'<think>(.*?)</think>', block, re.DOTALL)
            total_think_blocks += len(think_blocks)
            
            # Give points for well-formed <think> blocks
            for think in think_blocks:
                # Check if the think block has meaningful content
                if len(think.strip()) > 50:  # Arbitrary threshold for meaningful content
                    total_reward += 0.2
                
                # Check if think block contains analysis of features
                if re.search(r'(feature|instruction count|TotalInsts)', think):
                    total_reward += 0.3
                
                # Check if think block mentions passes to be applied
                if re.search(r'(pass|--[a-zA-Z0-9-]+)', think):
                    total_reward += 0.3
            
            # Count <tool_call> blocks
            tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', block, re.DOTALL)
            total_tool_call_blocks += len(tool_call_blocks)
            
            # Give points for well-formed <tool_call> blocks
            for tool_call in tool_call_blocks:
                # Check if the tool call has "analyze_autophase" name
                if re.search(r'"name"\s*:\s*"analyze_autophase"', tool_call):
                    total_reward += 0.2
                
                # Check if the tool call has proper arguments
                if re.search(r'"arguments"\s*:\s*{.*"filename".*"optimization_passes"', tool_call, re.DOTALL):
                    total_reward += 0.3
                
                # Check if optimization_passes is a well-formed array
                if re.search(r'"optimization_passes"\s*:\s*\[.*\]', tool_call, re.DOTALL):
                    total_reward += 0.2
            
            # Count <tool_response> blocks
            tool_response_blocks = re.findall(r'<tool_response>(.*?)</tool_response>', block, re.DOTALL)
            total_tool_response_blocks += len(tool_response_blocks)
            
            # Give points for well-formed <tool_response> blocks
            for tool_response in tool_response_blocks:
                # Check if the tool response has a status field
                if re.search(r'"status"\s*:\s*"(success|error)"', tool_response):
                    total_reward += 0.2
                
                # Check if the tool response has a feature_analysis field
                if re.search(r'"feature_analysis"\s*:', tool_response):
                    total_reward += 0.3
        
        # Check final answer block
        if assistant_blocks:
            last_block = assistant_blocks[-1]
            
            # Check if last block has <answer> tag
            answer_match = re.search(r'<answer>(.*?)</answer>', last_block, re.DOTALL)
            if answer_match:
                total_reward += 1.0
                
                answer_content = answer_match.group(1).strip()
                
                # Check if the answer is a valid JSON array of passes
                try:
                    json.loads(answer_content)
                    total_reward += 1.0
                except json.JSONDecodeError:
                    # If it's not valid JSON but has the structure of an array
                    if answer_content.startswith('[') and answer_content.endswith(']'):
                        total_reward += 0.5
                    
                    # Check if it at least contains passes
                    if re.search(r'--[a-zA-Z0-9-]+', answer_content):
                        total_reward += 0.3
        
        # Determine if the overall pattern matches the expected SFT structure
        expected_pattern = True
        
        # Must have at least one think-tool-response cycle
        if total_think_blocks == 0 or total_tool_call_blocks == 0 or total_tool_response_blocks == 0:
            expected_pattern = False
        
        # The number of blocks should be roughly balanced
        if abs(total_think_blocks - total_tool_call_blocks) > 1 or abs(total_think_blocks - total_tool_response_blocks) > 1:
            expected_pattern = False
        
        if expected_pattern:
            # Reward following the expected pattern
            total_reward += 2.0
        
        # Cap the total reward
        return min(total_reward, 10.0)
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0

def trace_agent_optimization_process(solution_str: str, ll_code: str) -> Tuple[List[str], float]:
    """Extract the final pass sequence from the answer tag and calculate overOz.
    
    Args:
        solution_str: Solution text
        ll_code: LLVM IR code
        
    Returns:
        (optimization pass list, overOz value)
    """
    if solution_str is None or ll_code is None:
        return [], 0.0
    
    try:
        # Extract all assistant blocks
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return [], 0.0
        
        # Get the last assistant block and extract answer
        last_block = assistant_blocks[-1]
        answer_content = extract_answer(last_block)
        
        if not answer_content:
            return [], 0.0
        
        # Parse the optimization sequence from the answer
        final_pass_list = parse_optimization_sequence(answer_content)
        
        if not final_pass_list:
            return [], 0.0
        
        # Calculate overOz using the extracted passes
        llvm_tools_path = os.path.join(os.path.dirname(__file__), 
                                     '../../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')
        try:
            overoz = get_overOz(ll_code, final_pass_list, llvm_tools_path=llvm_tools_path)
            return final_pass_list, float(overoz)
        except Exception as e:
            print(f"[DEBUG] Error calculating overOz: {e}")
            return final_pass_list, 0.0
            
    except Exception as e:
        print(f"[DEBUG] Error in trace_agent_optimization_process: {e}")
        return [], 0.0

def compute_score_answer(solution_str: Optional[str], ground_truth: Optional[Union[str, List[str]]]) -> float:
    """Compute the answer reward score based on the overOz value.
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth (filename)
        
    Returns:
        Answer reward score (-10.0 to 15.0)
    """
    if solution_str is None or ground_truth is None:
        return -10.0
    
    try:
        # Get the filename from ground_truth
        filename = ground_truth if isinstance(ground_truth, str) else ground_truth[0]
        ll_file_path = os.path.join(os.path.dirname(__file__) + "/../../../examples/data_preprocess/llvmir_datasets/", filename)
        ll_code = read_llvm_ir_file(ll_file_path)
        
        if not ll_code:
            return -10.0
        
        # Extract optimization passes and calculate overOz
        pass_list, overoz = trace_agent_optimization_process(solution_str, ll_code)
        print(f"[DEBUG] pass_list: {pass_list}, overoz: {overoz}")
        
        # No valid passes found
        if not pass_list:
            return -10.0
        
        # Define reward based on overOz value
        if overoz > 0:
            # Positive overOz (better than -Oz optimization)
            if overoz > 0.1:
                # Significantly better than -Oz
                reward = min(20.0 * overoz, 15.0)
            else:
                # Slightly better than -Oz
                reward = 10.0 * overoz
        else:
            # Negative overOz (worse than -Oz optimization)
            reward = max(overoz * 8.0, -6.0)
        
        return reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return -10.0

def compute_score_format_answer(solution_str: str, ground_truth: Union[str, List[str]]) -> float:
    """Compute the total reward score combining format and answer scores.
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth (filename)
        
    Returns:
        Total reward score (-10.0 to 15.0)
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        # Calculate individual scores
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)
        
        print(f"[DEBUG] Format reward: {format_reward}, Answer reward: {answer_reward}")
        
        # Calculate total reward based on both scores
        if answer_reward > 0:
            # For positive answer rewards, put more weight on the answer (actual optimization)
            if answer_reward > 8.0:
                # For exceptional optimization results, make answer even more important
                total_reward = 0.1 * format_reward + 0.9 * answer_reward
            else:
                # For good optimization results
                total_reward = 0.2 * format_reward + 0.8 * answer_reward
        else:
            # For negative answer rewards, penalize heavily but still consider format
            total_reward = 0.05 * format_reward + 0.95 * answer_reward
        
        # Ensure reward is within acceptable bounds
        total_reward = min(max(total_reward, -10.0), 15.0)
        
        return total_reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0