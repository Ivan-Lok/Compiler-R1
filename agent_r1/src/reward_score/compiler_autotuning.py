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
import ast
import datetime # Added for the dummy save function
from typing import List, Union, Optional, Dict, Any, Tuple
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_overOz

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
        passes = re.findall(r'--?[a-zA-Z0-9-]+', sequence_str)
        if passes:
            return passes
            
    return []

# Helper function to extract conversation blocks (assuming this exists and works)
def extract_conversation_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extracts conversation blocks delimited by <|im_start|> and <|im_end|>.
    Handles potential variations in spacing and role names.
    """
    blocks = []
    # Updated regex to capture role robustly and handle optional newline after <|im_start|>
    pattern = re.compile(r"<\|im_start\|>\s*(\w+)\s*\n?(.*?)<\|im_end\|>", re.DOTALL)
    matches = pattern.finditer(text)
    for match in matches:
        role = match.group(1).strip().lower()
        # Strip leading/trailing whitespace, but preserve internal newlines
        content = match.group(2).strip()
        if role in ["assistant", "user", "system"]: # Allow 'system' role as well
             blocks.append({"role": role, "content": content})
        # Silently ignore blocks with unknown roles for robustness
    return blocks

# Dummy _save_debug_log function for demonstration
def _save_debug_log(log_lines: List[str], solution: str, func_name: str):
    """Saves the debug log and solution to a timestamped file."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Sanitize filename slightly (replace potential path chars if func_name is weird)
        safe_func_name = func_name.replace("/", "_").replace("\\", "_")
        filename = f"debug_log_{safe_func_name}_{timestamp}.txt"

        log_content = "\n".join(log_lines)
        full_content = f"{log_content}\n\n--- Solution String Processed ---\n{solution}\n--- End Solution String ---"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)
        # print(f"Debug log saved to {filename}") # Optional: print confirmation
    except Exception as e:
        print(f"Error saving debug log: {e}")


def compute_score_format(text):
    """
    检查输入文本是否满足以下格式要求:
    1. 必须包含 <|im_start|>assistant ... <|im_end|> 格式
    2. 必须包含 <answer> ... </answer> 标签或正确的工具调用<tool_call>...</tool_call>
    3. <answer> 标签内必须包含有效的优化选项序列(列表或单个选项)
    4. 如果包含工具调用，必须使用正确的工具且参数格式正确
    
    支持以下<answer>格式:
    - 列表格式: ["--option1", "--option2", ...]
    - Python列表: [--option1, --option2, ...]
    - 单行格式: "--option1 --option2 ..."
    - 多行格式: 每行一个选项
    
    支持的工具调用:
    - rag_search: 用于检索相似的优化序列
    - gen_autophase: 用于生成自动相位特征
    - optimize_llcode: 用于优化LLVM IR代码
    - count_instructions: 用于计算优化前后的指令数
    
    Args:
        text: 需要检查格式的输入字符串。

    Returns:
        格式评分: 10分(完全符合格式要求), 5分(基本符合但有小问题), 0分(部分符合), -10分(不符合)
    """
    debug_logs = []
    debug_logs.append("开始检查格式...")
    
    # 检查是否包含助手回复块
    assistant_pattern = r'<\|im_start\|>assistant\s*(.*?)<\|im_end\|>'
    assistant_match = re.search(assistant_pattern, text, re.DOTALL)
    if not assistant_match:
        debug_logs.append("格式错误: 不包含助手回复块")
        # _save_debug_log(debug_logs, text, "compute_score_format")
        return -10  # 不包含助手回复块
    
    assistant_content = assistant_match.group(1)
    
    # 检查是否包含tool_call标签
    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    tool_call_match = re.search(tool_call_pattern, assistant_content, re.DOTALL)
    
    # 检查是否包含answer标签
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, assistant_content, re.DOTALL)
    
    # 如果两者都不存在，则格式不正确
    if not tool_call_match and not answer_match:
        debug_logs.append("格式错误: 既没有<tool_call>也没有<answer>标签")
        # _save_debug_log(debug_logs, text, "compute_score_format")
        return -5  # 不包含任何有效标签
    
    # 评分初始值
    format_score = 0
    
    # 检查工具调用格式（如果存在）
    if tool_call_match:
        debug_logs.append("检测到<tool_call>标签")
        tool_call_content = tool_call_match.group(1).strip()
        
        # 检查工具调用JSON格式
        try:
            tool_data = json.loads(tool_call_content)
            tool_name = tool_data.get("name", "")
            debug_logs.append(f"工具名称: {tool_name}")
            
            # 检查是否使用了支持的工具
            supported_tools = ["rag_search", "gen_autophase", "optimize_llcode", "count_instructions"]
            if tool_name in supported_tools:
                format_score += 5
                debug_logs.append(f"使用了支持的工具: {tool_name}")
                
                # 检查参数格式
                arguments = tool_data.get("arguments", {})
                if isinstance(arguments, dict):
                    # 根据不同工具检查参数
                    if tool_name == "rag_search":
                        if "autophase_embedding" in arguments:
                            format_score += 5
                            debug_logs.append("rag_search工具参数格式正确")
                        else:
                            debug_logs.append("rag_search工具缺少必要参数: autophase_embedding")
                    
                    elif tool_name == "gen_autophase":
                        if "optimization_passes" in arguments:
                            format_score += 5
                            debug_logs.append("gen_autophase工具参数格式正确")
                        else:
                            debug_logs.append("gen_autophase工具缺少必要参数: optimization_passes")
                    
                    elif tool_name == "optimize_llcode":
                        if "ll_code" in arguments and "passes" in arguments:
                            format_score += 5
                            debug_logs.append("optimize_llcode工具参数格式正确")
                        else:
                            debug_logs.append("optimize_llcode工具缺少必要参数: ll_code或passes")
                            
                    elif tool_name == "count_instructions":
                        if "filename" in arguments and "optimization_passes" in arguments:
                            format_score += 5
                            debug_logs.append("count_instructions工具参数格式正确")
                        else:
                            debug_logs.append("count_instructions工具缺少必要参数: filename或optimization_passes")
                else:
                    debug_logs.append("工具参数格式错误: 不是字典类型")
            else:
                debug_logs.append(f"使用了不支持的工具: {tool_name}")
        except json.JSONDecodeError:
            debug_logs.append("工具调用JSON格式解析失败")
    
    # 检查答案格式（如果存在）
    if answer_match:
        debug_logs.append("检测到<answer>标签")
        answer_content = answer_match.group(1).strip()
        
        if not answer_content:
            debug_logs.append("answer标签内容为空")
            # _save_debug_log(debug_logs, text, "compute_score_format")
            return -5  # answer标签内容为空
        
        # 检查优化选项格式
        # 1. 检查JSON列表格式 ["--option1", "--option2", ...]
        json_list_pattern = r'\[\s*(?:"[^"]*"|\'[^\']*\')(?:\s*,\s*(?:"[^"]*"|\'[^\']*\'))?\s*\]'
        # 2. 检查Python列表表示 [--option1, --option2, ...]
        python_list_pattern = r'\[\s*(?:--[a-zA-Z0-9-]+)(?:\s*,\s*(?:--[a-zA-Z0-9-]+))?\s*\]'
        # 3. 检查单行格式 --option1 --option2 ...
        single_line_pattern = r'(?:--[a-zA-Z0-9-]+)(?:\s+--[a-zA-Z0-9-]+)*'
        # 4. 检查多行格式 (每行一个选项)
        multi_line_pattern = r'(?:--[a-zA-Z0-9-]+\s*)+' 
        
        # 检查优化选项
        has_valid_format = (
            re.fullmatch(json_list_pattern, answer_content) is not None or
            re.fullmatch(python_list_pattern, answer_content) is not None or
            re.fullmatch(single_line_pattern, answer_content) is not None or
            re.fullmatch(multi_line_pattern, answer_content) is not None or
            "--" in answer_content.lower()  # 宽松检查，至少包含一个优化选项标识
        )
        
        if has_valid_format:
            # 检查是否至少包含一个以--开头的选项
            if re.search(r'--[a-zA-Z0-9-]+', answer_content):
                format_score = 10  # 完全符合answer格式要求
                debug_logs.append("answer标签格式正确，包含有效的优化选项")
            else:
                format_score = 5  # 基本格式正确但没有明确的优化选项
                debug_logs.append("answer标签格式基本正确，但没有明确的优化选项")
        
        # 格式不符合要求但可能是 -Oz 选项
        elif "-Oz" in answer_content:
            format_score = 10  # 特殊情况: -Oz 选项
            debug_logs.append("answer标签包含特殊选项: -Oz")
        else:
            debug_logs.append("answer标签格式不符合要求")
    
    # 根据评分结果返回最终格式评分
    if format_score >= 10:
        final_score = 10
    elif format_score >= 5:
        final_score = 5
    elif format_score > 0:
        final_score = 0
    else:
        final_score = -10
    
    debug_logs.append(f"最终格式评分: {final_score}")
    # if final_score < 5:
    #     # _save_debug_log(debug_logs, text, "compute_score_format")
    
    return final_score

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
        Answer reward score
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        # Get the filename from ground_truth
        filename = ground_truth if isinstance(ground_truth, str) else ground_truth[0]
        ll_file_path = os.path.join(os.path.dirname(__file__) + "/../../../examples/data_preprocess/llvmir_datasets/", filename)
        ll_code = read_llvm_ir_file(ll_file_path)
        
        if not ll_code:
            return 0.0
        
        # Extract optimization passes and calculate overOz
        pass_list, overoz = trace_agent_optimization_process(solution_str, ll_code)
        print(f"[DEBUG] pass_list: {pass_list}, overori: {overoz}")
        
        # No valid passes found
        if not pass_list:
            return 0.0
        
        reward = overoz * 30
        
        # if reward < 0:
        #     return 0.0

        return reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0

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
        
        # print(f"[DEBUG] Format reward: {format_reward}, Answer reward: {answer_reward}")
        
        total_reward = 0.05 * format_reward + 0.95 * answer_reward
        # Ensure reward is within acceptable bounds
        # total_reward = min(max(total_reward, -10.0), 15.0)
        
        return total_reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0

# a = '''
# <|im_start|>assistant
# <answer>
# ['-Oz']
# </answer>
# <|im_end|> 
# '''

# print(compute_score_answer(a, "train/opencv-v0/opencv-v0_1.ll"))