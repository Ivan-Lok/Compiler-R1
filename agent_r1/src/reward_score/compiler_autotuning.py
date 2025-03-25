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

def extract_pass_from_think(think_content: str) -> Optional[List[str]]:
    """从think内容中提取优化pass。
    
    Args:
        think_content: think内容
        
    Returns:
        提取的优化pass列表，如果没有找到则返回None
    """
    # 尝试匹配"--"开头的优化pass
    pass_patterns = [
        r'使用(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
        r'选择(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
        r'应用(?:优化pass|pass|优化选项)\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
        r'(?:优化pass|pass|优化选项)[：:]\s*["\']?(-{2}[a-zA-Z0-9-]+)["\']?',
        r'["\'](-{2}[a-zA-Z0-9-]+)["\']',
        r'(-{2}[a-zA-Z0-9-]+)'
    ]
    
    for pattern in pass_patterns:
        match = re.search(pattern, think_content, re.IGNORECASE)
        if match:
            return [match.group(1)]
    
    # 如果上面的模式都没匹配到，尝试查找所有--开头的字符串
    all_passes = re.findall(r'(-{2}[a-zA-Z0-9-]+)', think_content)
    if all_passes:
        return all_passes
            
    return None

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
        sequence_str: Optimization sequence string
        
    Returns:
        List of optimization options
    """
    try:
        # Attempt to parse JSON string
        if sequence_str.startswith('[') and sequence_str.endswith(']'):
            try:
                return json.loads(sequence_str)
            except json.JSONDecodeError:
                pass
        
        # 尝试解析Markdown代码块内的内容
        code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        code_match = re.search(code_block_pattern, sequence_str, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试解析以半角逗号分隔的列表
        if ',' in sequence_str:
            items = sequence_str.strip('[]').replace("'", "").replace('"', '').split(',')
            return [item.strip() for item in items if item.strip()]
            
        # 尝试解析以全角逗号分隔的列表
        if '，' in sequence_str:
            items = sequence_str.strip('[]').replace("'", "").replace('"', '').split('，')
            return [item.strip() for item in items if item.strip()]
            
        # 如果以上方法都失败，尝试匹配所有--开头的选项
        if '--' in sequence_str:
            options = re.findall(r'(--[a-zA-Z0-9-]+)', sequence_str)
            if options:
                return options
        
        return []
    except Exception as e:
        print(f"[DEBUG] Error parsing optimization sequence: {e}")
        return []

def compute_score_format(solution_str: str) -> float:
    """Compute the format reward score.
    
    Args:
        solution_str: Solution text
        
    Returns:
        Format reward score (0.0-1.0)
    """
    if solution_str is None:
        return 0.0
    
    try:
        # Check basic structure
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        
        if not assistant_blocks:
            return 0.0
        
        format_reward = 0.0
        
        # Check each assistant block (except the last one)
        has_think_with_pass = False
        has_gen_autophase_tool_call = False
        
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            # 检查是否有包含优化pass的think和有效的工具调用
            if '<think>' in assistant_block and '</think>' in assistant_block:
                think_content = extract_think_content(assistant_block)
                if think_content:
                    pass_list = extract_pass_from_think(think_content)
                    if pass_list:
                        has_think_with_pass = True
                        format_reward += 0.2
            
            if '<tool_call>' in assistant_block and '</tool_call>' in assistant_block:
                tool_data = extract_tool_call_data(assistant_block)
                if tool_data and tool_data.get('name') == 'gen_autophase':
                    has_gen_autophase_tool_call = True
                    format_reward += 0.2
            
            # 检查格式：<think>...</think>\n<tool_call>...</tool_call>
            if (assistant_block.count('<think>') == 1 and 
                assistant_block.count('</think>') == 1 and 
                assistant_block.count('<tool_call>') == 1 and 
                assistant_block.count('</tool_call>') == 1):
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', 
                                      assistant_block, re.DOTALL)
                if think_match:
                    format_reward += 0.1
        
        # 检查最后一个回复是否包含答案
        if assistant_blocks:
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', 
                                         last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.5
                answer_content = extract_answer(last_assistant_block)
                if answer_content and len(parse_optimization_sequence(answer_content)) > 0:
                    format_reward += 0.5
        
        if has_think_with_pass and has_gen_autophase_tool_call:
            format_reward += 0.5
                
        return min(format_reward, 2.0)
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0

def trace_agent_optimization_process(solution_str: str, ll_code: str) -> Tuple[List[str], float]:
    """追踪代理的优化过程，提取最终的pass序列，并计算overOz。
    
    Args:
        solution_str: 解决方案文本
        ll_code: 原始LLVM IR代码
        
    Returns:
        (优化pass列表, overOz值)
    """
    if solution_str is None or ll_code is None:
        return [], 0.0
    
    try:
        # 提取所有助手块
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks:
            return [], 0.0
        
        # 尝试从最后一个block中直接提取答案
        final_pass_list = []
        last_block = assistant_blocks[-1]
        answer = extract_answer(last_block)
        
        if answer:
            final_pass_list = parse_optimization_sequence(answer)
            if final_pass_list:
                # 计算overOz
                llvm_tools_path = os.path.join(os.path.dirname(__file__), 
                                              '../../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')
                try:
                    overoz = get_overOz(ll_code, final_pass_list, llvm_tools_path=llvm_tools_path)
                    return final_pass_list, float(overoz)
                except Exception as e:
                    print(f"[DEBUG] Error calculating overOz from answer: {e}")
        
        # 如果从答案中无法直接获取，尝试从对话中重建优化过程
        accumulated_passes = []
        
        for block in assistant_blocks[:-1]:  # 跳过最后一个块
            # 从工具调用中提取passes
            tool_data = extract_tool_call_data(block)
            if tool_data and tool_data.get('name') == 'gen_autophase':
                passes = extract_passes_from_tool_call(tool_data)
                if passes:
                    # 如果passes是累积的（包含之前的passes），需要提取新添加的
                    if accumulated_passes:
                        new_passes = [p for p in passes if p not in accumulated_passes]
                        accumulated_passes = passes  # 更新累积passes
                    else:
                        accumulated_passes = passes
        
        # 如果我们重建了优化过程，使用累积的passes计算overOz
        if accumulated_passes:
            llvm_tools_path = os.path.join(os.path.dirname(__file__), 
                                         '../../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')
            try:
                overoz = get_overOz(ll_code, accumulated_passes, llvm_tools_path=llvm_tools_path)
                return accumulated_passes, float(overoz)
            except Exception as e:
                print(f"[DEBUG] Error calculating overOz from accumulated passes: {e}")
        
        return final_pass_list or accumulated_passes, 0.0
        
    except Exception as e:
        print(f"[DEBUG] Error in trace_agent_optimization_process: {e}")
        return [], 0.0

def compute_score_answer(solution_str: Optional[str], ground_truth: Optional[Union[str, List[str]]]) -> float:
    """Compute the answer reward score.
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth (ll_code)
        
    Returns:
        Answer reward score
    """
    if solution_str is None or ground_truth is None:
        return -10.0
    
    try:
        # 提取最终答案和计算overOz
        ll_code = ground_truth if isinstance(ground_truth, str) else ground_truth[0]
        pass_list, overoz = trace_agent_optimization_process(solution_str, ll_code)
        print(f"[DEBUG] pass_list: {pass_list}, overoz: {overoz}")

        if pass_list:
            if overoz is not None:
                reward = overoz * 5.0
                return reward
            else:
                reward = -10.0
                return reward
        else:
            reward = -10.0
            return reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return -10.0

def compute_score_format_answer(solution_str: str, ground_truth: Union[str, List[str]]) -> float:
    """Compute the total reward score (format reward + answer reward).
    
    Args:
        solution_str: Solution text
        ground_truth: Ground truth (ll_code)
        
    Returns:
        Total reward score
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        # 总奖励 = 格式奖励 + 答案奖励
        total_reward = 0.2 * format_reward + 0.8 * answer_reward
        
        print(f"[DEBUG] Format reward: {format_reward}, Answer reward: {answer_reward}, Total: {total_reward}")
        
        return total_reward
            
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0