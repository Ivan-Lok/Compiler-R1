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
    """
    计算基于严格格式要求的奖励分数。

    主要检查点：
    1.  是否严格包含5轮 <think>/<tool_call>/<tool_response>。
    2.  每个 <think> 块是否包含正确的 `[Round x/5]` 标记。
    3.  是否在第5轮之后紧跟着 <answer> 标签。
    4.  <answer> 标签后是否有多余的轮次或标签。
    5.  (可选) 内部轮次的其他一致性。

    Args:
        solution_str: 模型生成的完整响应文本。

    Returns:
        格式奖励分数 (范围可能从 -15.0 到 10.0)。
        - 严重违反格式（如超过5轮, 错误Round标记）将导致大的负分。
        - 完全符合格式将得到高分。
    """
    print(f"--- Computing Score for Solution ---\n{solution_str[:500]}...\n--- END ---") # 打印前缀用于调试

    # --- 基本设置 ---
    MAX_SCORE = 10.0
    PENALTY_EXCEED_ROUNDS = -15.0 # 超过5轮的严厉惩罚
    PENALTY_MISSING_ANSWER = -2.0
    PENALTY_JUNK_AFTER_ANSWER = -5.0
    PENALTY_MULTIPLE_ANSWERS = -5.0
    PENALTY_MISSING_ROUND_MARKER = -1.5 # 每个缺失标记的惩罚
    PENALTY_WRONG_TOTAL_ROUNDS_MARKER = -1.5 # 每个标记 /y != 5 的惩罚
    PENALTY_WRONG_ROUND_INDEX_MARKER = -1.0 # 每个标记 x != i+1 的惩罚

    REWARD_BASE_CORRECT_ROUNDS = 1.0 # 基础分：轮数结构正确
    REWARD_SINGLE_ANSWER = 2.0     # 奖励：有且只有一个答案标签
    REWARD_CLEAN_AFTER_ANSWER = 0.5  # 奖励：答案后面干净
    REWARD_PER_CORRECT_INTERNAL_ROUND = 1.0 # 每轮内部格式基本正确的奖励 (包括Round标记)
    # 总内部奖励 = 5 * 1.0 = 5.0
    # 最高理论分数 = 1.0 + 2.0 + 0.5 + 5.0 = 8.5 (可以调整使其接近10.0)
    # 调整一下让最高接近10:
    REWARD_BASE_CORRECT_ROUNDS = 2.0
    REWARD_SINGLE_ANSWER = 3.0
    # 最高理论 = 2.0 + 3.0 + 0.5 + 5.0 = 10.5 (将被min(MAX_SCORE)限制)


    if not solution_str:
        print("[DEBUG] Solution is empty or None.")
        return 0.0

    # 1. 提取助手回答内容
    assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)(<\|im_end\|>|$)', solution_str, re.DOTALL)
    if not assistant_match:
        print("[DEBUG] Could not find assistant block.")
        return 0.0
    assistant_content = assistant_match.group(1).strip()

    # 2. 查找所有核心标签块
    think_blocks = re.findall(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
    tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', assistant_content, re.DOTALL)
    tool_response_blocks = re.findall(r'<tool_response>(.*?)</tool_response>', assistant_content, re.DOTALL)
    answer_blocks = re.findall(r'<answer>(.*?)</answer>', assistant_content, re.DOTALL)

    num_think = len(think_blocks)
    num_tool_call = len(tool_call_blocks)
    num_tool_response = len(tool_response_blocks)
    num_answer = len(answer_blocks)

    print(f"[DEBUG] Found Blocks: Think={num_think}, ToolCall={num_tool_call}, ToolResponse={num_tool_response}, Answer={num_answer}")

    # 3. 核心规则检查：轮数必须精确为5
    # --- 惩罚：超过5轮 ---
    if num_think > 5 or num_tool_call > 5 or num_tool_response > 5:
        print(f"[DEBUG] FAIL: Exceeded 5 rounds. Applying LARGE PENALTY.")
        return PENALTY_EXCEED_ROUNDS
    # --- 奖励基础：恰好5轮 ---
    elif num_think == 5 and num_tool_call == 5 and num_tool_response == 5:
        print("[DEBUG] PASS: Exactly 5 rounds structure detected.")
        total_reward = REWARD_BASE_CORRECT_ROUNDS
    # --- 不奖励/零分：少于5轮 ---
    else:
        print(f"[DEBUG] FAIL: Incorrect number of rounds ({num_think}/{num_tool_call}/{num_tool_response}), less than 5. Score is 0.")
        return 0.0

    # --- 如果我们达到这里，意味着结构轮数 = 5 ---

    # 4. 检查内部轮次的一致性和格式，特别是 [Round x/5]
    internal_consistency_score = 0.0
    all_internal_rounds_ok = True # 假设所有轮次内部都OK

    for i in range(5): # 遍历 0 到 4
        think_content = think_blocks[i]
        # tool_call_content = tool_call_blocks[i] # 可选，用于其他检查
        # tool_response_content = tool_response_blocks[i] # 可选

        round_num = i + 1
        current_round_score = REWARD_PER_CORRECT_INTERNAL_ROUND # 该轮的满分
        round_format_ok = True

        # --- 检查 [Round x/5] marker ---
        round_marker_match = re.search(r'\[Round\s+(\d+)/(\d+)\]', think_content)

        if not round_marker_match:
            print(f"[DEBUG] Round {round_num} FAIL: Missing [Round x/y] marker in <think>. Penalty: {PENALTY_MISSING_ROUND_MARKER}")
            total_reward += PENALTY_MISSING_ROUND_MARKER
            round_format_ok = False
        else:
            try:
                x = int(round_marker_match.group(1))
                y = int(round_marker_match.group(2))

                # 检查 y 是否为 5
                if y != 5:
                    print(f"[DEBUG] Round {round_num} FAIL: Incorrect total rounds marker [Round {x}/{y}] (should be /5). Penalty: {PENALTY_WRONG_TOTAL_ROUNDS_MARKER}")
                    total_reward += PENALTY_WRONG_TOTAL_ROUNDS_MARKER
                    round_format_ok = False

                # 检查 x 是否等于当前轮数 (仅在 y=5 时检查才有意义，但分开检查逻辑更清晰)
                if x != round_num:
                     print(f"[DEBUG] Round {round_num} FAIL: Incorrect round index marker [Round {x}/{y}] (should be {round_num}/5). Penalty: {PENALTY_WRONG_ROUND_INDEX_MARKER}")
                     total_reward += PENALTY_WRONG_ROUND_INDEX_MARKER
                     # 允许 y 不为 5 和 x 不对 同时扣分
                     round_format_ok = False

                if round_format_ok: # 只有当标记完全正确时才打印PASS
                     print(f"[DEBUG] Round {round_num} PASS: Correct [Round {x}/5] marker found.")

            except ValueError:
                 print(f"[DEBUG] Round {round_num} FAIL: Could not parse numbers in [Round x/y] marker. Penalty: {PENALTY_MISSING_ROUND_MARKER}") # 当作缺失处理
                 total_reward += PENALTY_MISSING_ROUND_MARKER
                 round_format_ok = False

        # (可选) 添加其他内部检查，例如 tool_call 内容
        # if '"optimization_passes":' not in tool_call_content:
        #     print(f"[DEBUG] Round {round_num} WARN: Missing 'optimization_passes' in <tool_call>.")
        #     # 可以选择性添加小额惩罚或不给满分
        #     round_format_ok = False

        if round_format_ok:
            internal_consistency_score += REWARD_PER_CORRECT_INTERNAL_ROUND
        else:
            all_internal_rounds_ok = False # 标记至少有一轮内部格式错误

    print(f"[DEBUG] Internal Consistency Score contribution: {internal_consistency_score} ({internal_consistency_score / REWARD_PER_CORRECT_INTERNAL_ROUND:.0f}/5 rounds internally ok)")
    total_reward += internal_consistency_score # 将内部轮次分数加到总分

    # 5. 检查 <answer> 标签 (只有在恰好5轮结构时才进行)
    if num_answer == 1:
        print("[DEBUG] PASS: Exactly one <answer> tag found.")
        total_reward += REWARD_SINGLE_ANSWER

        # 检查 <answer> 之后是否有多余内容
        answer_match = re.search(r'</answer>', assistant_content, re.DOTALL)
        if answer_match:
             answer_end_pos = answer_match.end()
             text_after_answer = assistant_content[answer_end_pos:]
             if re.search(r'<think>|<tool_call>|<tool_response>|\[Round\s*\d+/\d+\]', text_after_answer, re.DOTALL):
                 print(f"[DEBUG] FAIL: Invalid content found after </answer> tag. Penalty: {PENALTY_JUNK_AFTER_ANSWER}")
                 total_reward += PENALTY_JUNK_AFTER_ANSWER
             else:
                 print("[DEBUG] PASS: No invalid content found after </answer> tag.")
                 total_reward += REWARD_CLEAN_AFTER_ANSWER
        else:
             print("[DEBUG] WARN: Could not locate </answer> tag end position properly.")

    elif num_answer > 1:
        print(f"[DEBUG] FAIL: Multiple <answer> tags ({num_answer}) found. Penalty: {PENALTY_MULTIPLE_ANSWERS}")
        total_reward += PENALTY_MULTIPLE_ANSWERS
    else: # num_answer == 0
        print(f"[DEBUG] FAIL: Missing <answer> tag (expected with 5 rounds). Penalty: {PENALTY_MISSING_ANSWER}")
        total_reward += PENALTY_MISSING_ANSWER

    # 6. 最终分数处理：确保在合理范围内
    final_score = min(total_reward, MAX_SCORE)
    # 最低分取决于最差情况的惩罚总和，这里是 PENALTY_EXCEED_ROUNDS
    final_score = max(final_score, PENALTY_EXCEED_ROUNDS) # 确保不低于最大单项惩罚

    print(f"[DEBUG] Final Calculated Score: {final_score}")
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

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Example 1: Perfect Format
    perfect_solution = """<|im_start|>assistant
<think>
[Round 1/5] Analyzing features... Choosing initial passes: ['--instcombine', '--early-cse']. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{
  "filename": "test.ll",
  "optimization_passes": ["--instcombine", "--early-cse"]
}
</tool_call>
<tool_response>
{"features": {...}, "inst_count": 90}
</tool_response>
<think>
[Round 2/5] Inst count reduced. Adding '--sroa'. Passes: ['--instcombine', '--early-cse', '--sroa']. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{
  "filename": "test.ll",
  "optimization_passes": ["--instcombine", "--early-cse", "--sroa"]
}
</tool_call>
<tool_response>
{"features": {...}, "inst_count": 85}
</tool_response>
<think>
[Round 3/5] Good progress. Trying '--gvn'. Passes: ['--instcombine', '--early-cse', '--sroa', '--gvn']. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{
  "filename": "test.ll",
  "optimization_passes": ["--instcombine", "--early-cse", "--sroa", "--gvn"]
}
</tool_call>
<tool_response>
{"features": {...}, "inst_count": 80}
</tool_response>
<think>
[Round 4/5] Adding '--licm'. Passes: ['--instcombine', '--early-cse', '--sroa', '--gvn', '--licm']. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{
  "filename": "test.ll",
  "optimization_passes": ["--instcombine", "--early-cse", "--sroa", "--gvn", "--licm"]
}
</tool_call>
<tool_response>
{"features": {...}, "inst_count": 78}
</tool_response>
<think>
[Round 5/5] Final pass '--instsimplify'. Passes: ['--instcombine', '--early-cse', '--sroa', '--gvn', '--licm', '--instsimplify']. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{
  "filename": "test.ll",
  "optimization_passes": ["--instcombine", "--early-cse", "--sroa", "--gvn", "--licm", '--instsimplify']
}
</tool_call>
<tool_response>
{"features": {...}, "inst_count": 75}
</tool_response>
<answer>["--instcombine", "--early-cse", "--sroa", "--gvn", "--licm", "--instsimplify"]</answer><|im_end|>
"""
    print("\n--- Testing Perfect Solution ---")
    score1 = compute_score_format(perfect_solution)
    print(f"Score for perfect solution: {score1}") # Expected: Close to 10.0

    # Example 2: Exceeds 5 rounds
    too_many_rounds_solution = perfect_solution.replace("<answer>", """</tool_response>
<think>
[Round 6/5] Oh wait, let me try one more thing... '--deadargelim'. Passes: [...]. Tool call uses ALL passes applied up to the end of this round.
</think>
<tool_call>
{...}
</tool_call>
<tool_response>
{...}
</tool_response>
<answer>... an answer ...</answer>""") + "<|im_end|>" # Ensure end tag if missing

    print("\n--- Testing Too Many Rounds Solution ---")
    score2 = compute_score_format(too_many_rounds_solution)
    print(f"Score for too many rounds: {score2}") # Expected: -10.0

    # Example 3: Missing Answer
    missing_answer_solution = perfect_solution.split("<answer>")[0] + "<|im_end|>" # Remove answer and end tag if needed
    print("\n--- Testing Missing Answer Solution ---")
    score3 = compute_score_format(missing_answer_solution)
    print(f"Score for missing answer: {score3}") # Expected: Lower positive score (e.g., around 4.0-7.0 depending on internal checks)

    # Example 4: Junk after Answer
    junk_after_answer_solution = perfect_solution.replace("</answer>", "</answer>\n<think>Oops, forgot this.</think>")
    print("\n--- Testing Junk After Answer Solution ---")
    score4 = compute_score_format(junk_after_answer_solution)
    print(f"Score for junk after answer: {score4}") # Expected: Positive score, but lower than perfect due to penalty (e.g., 4.0-7.0)

    # Example 5: Less than 5 rounds
    too_few_rounds_solution = """<|im_start|>assistant
<think>[Round 1/5] ... </think><tool_call>{...}</tool_call><tool_response>{...}</tool_response>
<think>[Round 2/5] ... </think><tool_call>{...}</tool_call><tool_response>{...}</tool_response>
<answer>...</answer><|im_end|>"""
    print("\n--- Testing Too Few Rounds Solution ---")
    score5 = compute_score_format(too_few_rounds_solution)
    print(f"Score for too few rounds: {score5}") # Expected: 0.0