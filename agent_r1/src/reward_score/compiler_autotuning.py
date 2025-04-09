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
import hashlib
from typing import List, Union, Optional, Dict, Any, Tuple
import time
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
        passes = re.findall(r'--[a-zA-Z0-9-]+', sequence_str)
        if passes:
            return passes
            
    return []

# --- Helper Function to Save Debug Log ---
def _save_debug_log(debug_log: List[str], solution_str: str, function_name: str):
    """Saves the debug log list to a file."""
    try:
        # Determine log directory relative to the script file
        # Handle potential NameError if __file__ is not defined (e.g., in notebooks)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, "logs")
        except NameError:
            log_dir = "logs" # Default to a local 'logs' directory

        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # Create a simple hash for potentially shorter filenames
        # Use a slice of the input string for hashing if it's very long
        input_sample = solution_str[:1000] if len(solution_str) > 1000 else solution_str
        input_hash = hashlib.md5(input_sample.encode()).hexdigest()[:8]
        # Include function name in the log filename
        log_file = os.path.join(log_dir, f"debug_log_{function_name}_{timestamp}_{input_hash}.txt")

        # Ensure log lines are strings before joining
        str_debug_log = [str(line) for line in debug_log]

        with open(log_file, "w", encoding='utf-8') as f:
            f.write("\n".join(str_debug_log))
        # print(f"Debug log saved to: {log_file}") # Optional: print log file path
    except Exception as e:
        # Print log to console if saving fails
        print(f"[ERROR] Failed to save debug log to file: {e}")
        print("-" * 20 + " Log Content Start " + "-" * 20)
        # Ensure log lines are strings before printing
        str_debug_log_print = [str(line) for line in debug_log]
        print("\n".join(str_debug_log_print))
        print("-" * 20 + " Log Content End " + "-" * 20)

# Assume extract_conversation_blocks is defined as before (V5/V6 version)
# --- Helper Function to Extract Blocks (保持 V5/V6 版本) ---
def extract_conversation_blocks(solution_str: str) -> List[Dict[str, str]]:
    """Splits the solution string into sequential blocks with roles."""
    blocks = []
    pattern = re.compile(r'<\|im_start\|>(system|user|assistant)\n(.*?)(?=<\|im_start\|>|\Z)', re.DOTALL)
    cleaned_solution_str = solution_str.strip()
    if cleaned_solution_str.endswith("<|im_end|>"):
        cleaned_solution_str = cleaned_solution_str[:-len("<|im_end|>")].strip()

    last_end = 0
    for match in pattern.finditer(cleaned_solution_str):
        role = match.group(1)
        content = match.group(2).strip().removesuffix("<|im_end|>").strip()
        blocks.append({"role": role, "content": content})
        last_end = match.end()

    if last_end < len(cleaned_solution_str):
         trailing_content = cleaned_solution_str[last_end:].strip()
         if trailing_content and trailing_content != "<|im_end|>":
              # Use print for immediate feedback on parsing issues if needed
              # print(f"[WARN] extract_conversation_blocks: Possible trailing content detected: '{trailing_content[:100]}...'")
              pass # Or add to debug_log if preferred
    return blocks

# --- Main Scoring Function (V6.4 - Verbose Logging) ---
def compute_score_format(solution_str: str) -> float:
    """
    计算基于严格格式要求的奖励分数 (V6.4 - Verbose Logging)。
    1. Logs points awarded AND points *not* awarded for failed checks.
    2. No <answer>...</answer> -> 0 score.
    3. > 5 rounds -> Penalty.
    4. Checks are additive unless penalty applies. Score >= 0.
    5. Accumulation check uses tool_call passes.
    Args:
        solution_str: 模型生成的完整响应文本。

    Returns:
        格式奖励分数。
    """
    # Initialize debug log at the very beginning
    debug_log = ["[-- compute_score_format (V6.5 Verbose Logging) Log --]"]
    function_name_for_log = compute_score_format.__name__ # Store function name

    # --- Constants ---
    MAX_EXPECTED_ROUNDS = 5
    THINK_BLOCK_REQUIRED_ENDING = "\nTool call uses ALL passes applied up to the end of this round."
    # --- Point Values ---
    # (Point values remain the same as V6.4)
    POINTS_PER_VALID_ROUND_TRIPLET = 1.5; POINTS_BONUS_EXACTLY_5_ROUNDS = 2.5
    POINTS_PENALTY_EXCESSIVE_ROUNDS = 10.0
    POINTS_ANSWER_CORRECT_LOCATION_AND_CONTENT = 4.0; POINTS_ANSWER_PYTHON_LIST_FORMAT = 2.0
    POINTS_TOOL_CALL_JSON_PARSE = 0.2; POINTS_TOOL_CALL_NAME_CORRECT = 0.3
    POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID = 0.5; POINTS_TOOL_CALL_PASSES_LIST_FORMAT = 0.3
    POINTS_THINK_ROUND_MARKER_CORRECT = 0.4; POINTS_THINK_PLAN_LINE_PRESENT = 0.2
    POINTS_THINK_PASS_DESCRIPTIONS = 0.3; POINTS_THINK_PREV_ANALYSIS = 0.1
    POINTS_THINK_FEEDBACK = 0.1; POINTS_THINK_ENDS_CORRECTLY = 0.3
    POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT = 1.0; POINTS_ANSWER_MATCHES_FINAL_TOOL_CALL = 2.0
    # --- End Point Values ---

    # --- Initial Checks ---
    if not solution_str:
        debug_log.append("    [FAIL] Solution is empty or None. Score = 0.0")
        # Add solution string to log before saving (even if empty)
        debug_log.append("\n--- Solution String ---")
        debug_log.append(solution_str if solution_str else "<Empty String>")
        debug_log.append("--- End Solution String ---")
        # _save_debug_log(debug_log, solution_str if solution_str else "", function_name_for_log) # Call helper
        return 0.0

    # --- GATEKEEPER: Check for complete <answer>...</answer> tag ---
    if not re.search(r'<answer>.*?</answer>', solution_str, re.IGNORECASE | re.DOTALL):
        debug_log.append("    [FAIL] No complete '<answer>...</answer>' structure found. Score = 0.0")
        # Add solution string to log before saving
        debug_log.append("\n--- Solution String ---")
        debug_log.append(solution_str)
        debug_log.append("--- End Solution String ---")
        # _save_debug_log(debug_log, solution_str, function_name_for_log) # Call helper
        return 0.0
    # If we reach here, the answer tag exists
    debug_log.append("    [INFO] Complete '<answer>...</answer>' structure found. Proceeding.")

    # --- Initialization ---
    total_score = 0.0
    applied_penalties = []

    # 1. Extract Blocks
    conversation_blocks = extract_conversation_blocks(solution_str)
    if not conversation_blocks:
        debug_log.append("    [FAIL] Could not extract conversation blocks.")
        # Optionally save log here
        return 0.0
    total_blocks_extracted = len(conversation_blocks)
    debug_log.append(f"    [INFO] Extracted {total_blocks_extracted} conversation blocks.")

    # 2. Assemble Rounds
    rounds_data: List[Dict[str, Optional[Any]]] = []
    i = 0
    while i < len(conversation_blocks):
        block = conversation_blocks[i]; role = block["role"]
        if role == "assistant":
            content = block["content"]
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if think_match and tool_call_match:
                if i + 1 < total_blocks_extracted and conversation_blocks[i+1]["role"] == "user":
                    response_match = re.search(r'<tool_response>(.*?)</tool_response>', conversation_blocks[i+1]["content"], re.DOTALL)
                    if response_match:
                        rounds_data.append({
                            "think": think_match.group(1).strip(), "tool_call": tool_call_match.group(1).strip(),
                            "tool_response": response_match.group(1).strip(), "assistant_block_index": i, "user_block_index": i + 1
                        })
                        i += 1
        i += 1
    num_rounds_found = len(rounds_data)
    debug_log.append(f"    [INFO] Assembled {num_rounds_found} valid round triplets.")

    # --- Scoring Section 1: Round Structure ---
    score_s1 = 0.0
    s1_log = ["[-- Section 1: Round Structure Checks --]"]
    fifth_round_user_block_index: int = -1
    expected_answer_block_index: int = -1
    expected_final_block_count : int = -1
    structure_after_5_rounds_is_potentially_perfect = False

    score_s1 += min(num_rounds_found, MAX_EXPECTED_ROUNDS) * POINTS_PER_VALID_ROUND_TRIPLET
    s1_log.append(f"    [+ {min(num_rounds_found, MAX_EXPECTED_ROUNDS) * POINTS_PER_VALID_ROUND_TRIPLET:.2f}] Points for {min(num_rounds_found, MAX_EXPECTED_ROUNDS)} valid rounds.")

    if num_rounds_found > MAX_EXPECTED_ROUNDS:
        penalty = POINTS_PENALTY_EXCESSIVE_ROUNDS; total_score -= penalty; applied_penalties.append(f"Excessive Rounds (-{penalty:.2f})")
        s1_log.append(f"    [- {penalty:.2f}] Penalty for {num_rounds_found} rounds (>{MAX_EXPECTED_ROUNDS}).")

    if num_rounds_found == MAX_EXPECTED_ROUNDS:
        score_s1 += POINTS_BONUS_EXACTLY_5_ROUNDS
        s1_log.append(f"    [+ {POINTS_BONUS_EXACTLY_5_ROUNDS:.2f}] Bonus for exactly {MAX_EXPECTED_ROUNDS} rounds.")

        fifth_round_user_block_index = rounds_data[MAX_EXPECTED_ROUNDS - 1].get("user_block_index", -1)
        expected_answer_block_index = fifth_round_user_block_index + 1; expected_final_block_count = expected_answer_block_index + 1
        s1_log.append(f"    [INFO] Checking final structure: Expecting answer at idx {expected_answer_block_index}, total blocks {expected_final_block_count}.")
        final_block_ok, role_ok, content_ok, no_trailing_blocks = False, False, False, False
        if expected_answer_block_index < total_blocks_extracted:
            final_block_candidate = conversation_blocks[expected_answer_block_index]; final_block_ok = True
            if final_block_candidate['role'] == 'assistant':
                role_ok = True; final_block_content = final_block_candidate['content']
                has_answer = re.search(r'<answer>.*?</answer>', final_block_content, re.IGNORECASE | re.DOTALL)
                has_think = '<think>' in final_block_content; has_tool_call = '<tool_call>' in final_block_content
                if has_answer and not has_think and not has_tool_call: content_ok = True
                else: s1_log.append(f"    [INFO] Final block content check failed (HasAns: {bool(has_answer)}, HasThink: {has_think}, HasToolCall: {has_tool_call}).")
            else: s1_log.append(f"    [INFO] Final block role check failed (Role: '{final_block_candidate['role']}').")
        else: s1_log.append(f"    [INFO] Final block existence check failed (Expected idx {expected_answer_block_index}, Total {total_blocks_extracted}).")
        if total_blocks_extracted == expected_final_block_count: no_trailing_blocks = True
        else: s1_log.append(f"    [INFO] Trailing blocks check failed (Expected {expected_final_block_count}, Found {total_blocks_extracted}).")

        if final_block_ok and role_ok and content_ok and no_trailing_blocks:
            structure_after_5_rounds_is_potentially_perfect = True; s1_log.append(f"    [INFO] Final structure check: PASSED all conditions.")
        else: s1_log.append(f"    [INFO] Final structure check: FAILED one or more conditions.")
    elif num_rounds_found < MAX_EXPECTED_ROUNDS :
        s1_log.append(f"    [+ 0.00] Bonus not awarded (found {num_rounds_found} rounds, expected {MAX_EXPECTED_ROUNDS}).")
    # No explicit log needed for > 5 rounds bonus, as penalty already applied

    debug_log.extend(s1_log); total_score += score_s1
    debug_log.append(f"    [INFO] Section 1 Score (Positive Points): {score_s1:.2f}")

    # --- Scoring Section 2: Answer Tag Details ---
    score_s2 = 0.0
    s2_log = ["[-- Section 2: Answer Tag Details --]"]
    answer_content_raw: Optional[str] = None; final_answer_pass_list: Optional[List[str]] = None
    can_check_answer = False
    if num_rounds_found == MAX_EXPECTED_ROUNDS and expected_answer_block_index < total_blocks_extracted:
        final_block = conversation_blocks[expected_answer_block_index]
        if final_block['role'] == 'assistant':
            answer_match = re.search(r'<answer>(.*?)</answer>', final_block['content'], re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer_content_raw = answer_match.group(1).strip(); s2_log.append(f"    [INFO] Found <answer> tag content in final block (idx {expected_answer_block_index}).")
                can_check_answer = True
            else: s2_log.append(f"    [INFO] No <answer> tag found in final assistant block.")
        else: s2_log.append(f"    [INFO] Final block not 'assistant'.")
    else: s2_log.append(f"    [INFO] Skipping answer checks (Rounds != 5 or structure too short).")

    if can_check_answer:
        # Check 1: Location/Content/Structure Correctness
        if structure_after_5_rounds_is_potentially_perfect:
            score_s2 += POINTS_ANSWER_CORRECT_LOCATION_AND_CONTENT; s2_log.append(f"    [+ {POINTS_ANSWER_CORRECT_LOCATION_AND_CONTENT:.2f}] Answer location/content/structure correct.")
        else: s2_log.append(f"    [+ 0.00] Answer location/content/structure incorrect (Reason logged in S1).")

        # Check 2: Answer content format (Python Literal List)
        try:
            parsed_answer = ast.literal_eval(answer_content_raw) # type: ignore
            if isinstance(parsed_answer, list) and all(isinstance(item, str) and item.startswith('-') for item in parsed_answer):
                score_s2 += POINTS_ANSWER_PYTHON_LIST_FORMAT; final_answer_pass_list = parsed_answer
                s2_log.append(f"    [+ {POINTS_ANSWER_PYTHON_LIST_FORMAT:.2f}] Answer content is valid Python literal list.")
            else: s2_log.append(f"    [+ 0.00] Answer content format incorrect (Evaluated, but not list of '-...' strings).")
        except (ValueError, SyntaxError, TypeError) as e:
             s2_log.append(f"    [+ 0.00] Answer content format incorrect (Not a valid Python literal: {e}).")
        except Exception as e: s2_log.append(f"    [+ 0.00] Unexpected error checking answer format: {e}")
    else:
        s2_log.append(f"    [+ 0.00] Cannot check Answer Correct Location/Content (Prerequisites failed).")
        s2_log.append(f"    [+ 0.00] Cannot check Answer Python List Format (Prerequisites failed).")


    debug_log.extend(s2_log); total_score += score_s2
    debug_log.append(f"    [INFO] Section 2 Score: {score_s2:.2f}")

    # --- Scoring Sections 3, 4, 5 (Rounds Logic) ---
    score_s3 = 0.0; score_s4 = 0.0; score_s5 = 0.0
    s3_log = ["[-- Section 3: Tool Call Correctness (per round) --]"]
    s4_log = ["[-- Section 4: Think Content Details (per round) --]"]
    s5_log = ["[-- Section 5: Pass List Consistency --]"]
    previous_tool_call_passes: List[str] = []
    last_tool_call_passes: Optional[List[str]] = None

    llvmir_dataset_path: Optional[str] = None; path_check_possible: bool = False
    try: # Setup path check (same as before)
        script_dir = os.path.dirname(os.path.abspath(__file__)); base_path = os.path.abspath(os.path.join(script_dir, "../../.."))
        potential_path = os.path.join(base_path, "examples/data_preprocess/llvmir_datasets")
        if os.path.isdir(potential_path): llvmir_dataset_path = potential_path; path_check_possible = True
    except NameError: pass
    except Exception as e: debug_log.append(f"    [WARN] Error finding dataset path: {e}. File checks skipped.")

    num_rounds_to_process = min(num_rounds_found, MAX_EXPECTED_ROUNDS)

    for i in range(num_rounds_to_process):
        round_num = i + 1; round_data = rounds_data[i]
        s3_log.append(f"  Round {round_num}:")
        s4_log.append(f"  Round {round_num}:")
        s5_log.append(f"  Round {round_num}:")
        think_content = round_data.get("think"); tool_call_content = round_data.get("tool_call")

        # --- Section 4 (Think) ---
        round_score_s4 = 0.0; think_round_passes: List[str] = []
        if think_content:
            # Check Round Marker
            marker_found = False
            if re.search(r'\[Round\s+' + str(round_num) + r'/' + str(MAX_EXPECTED_ROUNDS) + r'\]', think_content):
                round_score_s4 += POINTS_THINK_ROUND_MARKER_CORRECT; s4_log.append(f"    [+ {POINTS_THINK_ROUND_MARKER_CORRECT:.2f}] Round marker.")
                marker_found = True
            else: s4_log.append(f"    [+ 0.00] Round marker incorrect or missing.")

            # Check Plan Line
            plan_line_found = False
            plan_match = re.search(r"- Plan \(Round " + str(round_num) + r"\):.*?following passes:(\[.*?\])", think_content, re.DOTALL)
            if plan_match:
                round_score_s4 += POINTS_THINK_PLAN_LINE_PRESENT; s4_log.append(f"    [+ {POINTS_THINK_PLAN_LINE_PRESENT:.2f}] Plan line.")
                plan_line_found = True
                try: # Attempt to parse passes from plan line
                    passes_str = plan_match.group(1).strip(); parsed_plan_passes = ast.literal_eval(passes_str)
                    if isinstance(parsed_plan_passes, list) and all(isinstance(p, str) and p.startswith('-') for p in parsed_plan_passes): think_round_passes = parsed_plan_passes
                except (ValueError, SyntaxError, TypeError): pass # Ignore parsing errors here, focus on presence
            else: s4_log.append(f"    [+ 0.00] Plan line missing or incorrect format.")

            # Check Pass Descriptions
            descriptions_found = False
            if re.search(r'^\s*-\s+--\w+:\s+.*', think_content, re.MULTILINE):
                 round_score_s4 += POINTS_THINK_PASS_DESCRIPTIONS; s4_log.append(f"    [+ {POINTS_THINK_PASS_DESCRIPTIONS:.2f}] Descriptions.")
                 descriptions_found = True
            else: s4_log.append(f"    [+ 0.00] Pass descriptions missing or incorrect format.")

            # Check Previous Analysis / Initial State
            prev_analysis_found = False
            if (round_num == 1 and "Initial State:" in think_content) or \
               (round_num > 1 and ("Recap:" in think_content or "Previous analysis:" in think_content)):
                 round_score_s4 += POINTS_THINK_PREV_ANALYSIS; s4_log.append(f"    [+ {POINTS_THINK_PREV_ANALYSIS:.2f}] Prev analysis/Recap/Initial.")
                 prev_analysis_found = True
            else: s4_log.append(f"    [+ 0.00] Prev analysis/Recap/Initial line missing.")

            # Check Feedback / Result Line
            feedback_found = False
            if "Result (after Round" in think_content or "Initial InstCount:" in think_content:
                round_score_s4 += POINTS_THINK_FEEDBACK; s4_log.append(f"    [+ {POINTS_THINK_FEEDBACK:.2f}] Result/Feedback line.")
                feedback_found = True
            else: s4_log.append(f"    [+ 0.00] Result/Feedback line missing.")

            # Check Exact Ending
            ends_correctly = False
            cleaned_think_content = think_content.rstrip()
            if cleaned_think_content.endswith(THINK_BLOCK_REQUIRED_ENDING):
                round_score_s4 += POINTS_THINK_ENDS_CORRECTLY; s4_log.append(f"    [+ {POINTS_THINK_ENDS_CORRECTLY:.2f}] Think ends correctly.")
                ends_correctly = True
            else: s4_log.append(f"    [+ 0.00] Think ending incorrect.")

            score_s4 += round_score_s4
        else: s4_log.append("    [INFO] Think content missing. All Section 4 checks [+0.00].")

        # --- Section 3 (Tool Call) ---
        round_score_s3 = 0.0; current_tool_call_passes_list: Optional[List[str]] = None
        if tool_call_content:
            json_parsed = False; name_correct = False; filename_exists = False; passes_format_correct = False
            try:
                tool_call_data = json.loads(tool_call_content); round_score_s3 += POINTS_TOOL_CALL_JSON_PARSE; s3_log.append(f"    [+ {POINTS_TOOL_CALL_JSON_PARSE:.2f}] JSON parsed.")
                json_parsed = True

                if tool_call_data.get("name") == "analyze_autophase": round_score_s3 += POINTS_TOOL_CALL_NAME_CORRECT; s3_log.append(f"    [+ {POINTS_TOOL_CALL_NAME_CORRECT:.2f}] Tool name."); name_correct = True
                else: s3_log.append(f"    [+ 0.00] Tool name incorrect.")

                args = tool_call_data.get("arguments")
                if isinstance(args, dict):
                    filename_value = args.get("filename")
                    if filename_value and isinstance(filename_value, str):
                        if path_check_possible and llvmir_dataset_path:
                            norm_filename = os.path.normpath(filename_value)
                            if ".." not in norm_filename.split(os.path.sep):
                                ll_file_path = os.path.join(llvmir_dataset_path, norm_filename)
                                if os.path.exists(ll_file_path) and os.path.isfile(ll_file_path): round_score_s3 += POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID; s3_log.append(f"    [+ {POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID:.2f}] Filename exists."); filename_exists = True
                                else: s3_log.append(f"    [+ 0.00] Filename does not exist: {ll_file_path}")
                            else: s3_log.append(f"    [+ 0.00] Invalid filename path (contains '..').")
                        else: s3_log.append(f"    [INFO] File existence check skipped. No points awarded or deducted for this.") # No points if cannot check
                    else: s3_log.append(f"    [+ 0.00] Filename missing or not string in arguments.")

                    passes_arg = args.get("optimization_passes")
                    if isinstance(passes_arg, list) and all(isinstance(p, str) and p.startswith('-') for p in passes_arg):
                         round_score_s3 += POINTS_TOOL_CALL_PASSES_LIST_FORMAT; current_tool_call_passes_list = passes_arg; last_tool_call_passes = current_tool_call_passes_list; s3_log.append(f"    [+ {POINTS_TOOL_CALL_PASSES_LIST_FORMAT:.2f}] Passes format."); passes_format_correct = True
                    else: s3_log.append(f"    [+ 0.00] Passes key missing, not list, or items have wrong format.")
                else: s3_log.append(f"    [+ 0.00] Arguments key missing or not dict.")
            except json.JSONDecodeError: s3_log.append(f"    [+ 0.00] JSON parse failed.")
            except Exception as e: s3_log.append(f"    [+ 0.00] Error processing tool call: {e}")
            score_s3 += round_score_s3
        else: s3_log.append("    [INFO] Tool call content missing. All Section 3 checks [+0.00].")

        # --- Section 5 (Accumulation Check) ---
        round_score_s5 = 0.0
        if current_tool_call_passes_list is not None:
             expected_passes_this_round = previous_tool_call_passes + think_round_passes
             if current_tool_call_passes_list == expected_passes_this_round:
                 round_score_s5 += POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT
                 s5_log.append(f"    [+ {POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT:.2f}] Accumulation match.")
             else: s5_log.append(f"    [+ 0.00] Tool call pass accumulation incorrect.")
        else: s5_log.append(f"    [+ 0.00] Cannot check accumulation (current tool call passes invalid or missing).")
        score_s5 += round_score_s5

        # Update previous passes for next iteration
        if current_tool_call_passes_list is not None:
            previous_tool_call_passes = current_tool_call_passes_list


    # --- Final Section 5 Check: Answer vs Last Tool Call ---
    s5_log.append(f"  Final Answer vs Last Tool Call Check:")
    final_check_score_s5 = 0.0
    if final_answer_pass_list is not None and last_tool_call_passes is not None and num_rounds_found == MAX_EXPECTED_ROUNDS:
         if final_answer_pass_list == last_tool_call_passes:
             final_check_score_s5 += POINTS_ANSWER_MATCHES_FINAL_TOOL_CALL; s5_log.append(f"    [+ {POINTS_ANSWER_MATCHES_FINAL_TOOL_CALL:.2f}] Answer matches final call.")
         else: s5_log.append(f"    [+ 0.00] Answer does not match final call passes.")
    elif num_rounds_found != MAX_EXPECTED_ROUNDS:
        s5_log.append(f"    [+ 0.00] Cannot check answer vs final call (Rounds != 5).")
    elif final_answer_pass_list is None:
         s5_log.append(f"    [+ 0.00] Cannot check answer vs final call (Answer parse failed).")
    elif last_tool_call_passes is None:
         s5_log.append(f"    [+ 0.00] Cannot check answer vs final call (Final tool call passes missing/invalid).")
    score_s5 += final_check_score_s5


    # --- Add final scores ---
    debug_log.extend(s3_log); total_score += score_s3
    debug_log.append(f"  [INFO] Section 3 Score: {score_s3:.2f}")
    debug_log.extend(s4_log); total_score += score_s4
    debug_log.append(f"  [INFO] Section 4 Score: {score_s4:.2f}")
    debug_log.extend(s5_log); total_score += score_s5
    debug_log.append(f"  [INFO] Section 5 Score: {score_s5:.2f}")

    # --- Final Score ---
    final_score = max(0.0, round(total_score, 2))
    debug_log.append(f"\n--- Final Calculated Score (V6.4): {final_score} ---")
    if applied_penalties: debug_log.append(f"--- Penalties Applied: {', '.join(applied_penalties)} ---")

    # --- Logging ---
    # Add solution string to the end of the log for context when reading file
    debug_log.append("\n--- Solution String Processed ---")
    debug_log.append(solution_str)
    debug_log.append("--- End Solution String ---")
    # _save_debug_log(debug_log, solution_str, function_name_for_log) # Call helper

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
            return -10.0
        
        # # Define reward based on overOz value
        # if overoz > 0:
        #     # Positive overOz (better than -Oz optimization)
        #     if overoz > 0.1:
        #         # Significantly better than -Oz
        #         reward = min(20.0 * overoz, 15.0)
        #     else:
        #         # Slightly better than -Oz
        #         reward = 10.0 * overoz
        # else:
        #     # Negative overOz (worse than -Oz optimization)
        #     reward = max(overoz * 8.0, 0)
        reward = overoz * 30
        
        return max(reward, -10.0)
        
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
        return -10.0
    
    try:
        # Calculate individual scores
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)
        
        print(f"[DEBUG] Format reward: {format_reward}, Answer reward: {answer_reward}")
        
        # # Calculate total reward based on both scores
        # if answer_reward > 0:
        #     # For positive answer rewards, put more weight on the answer (actual optimization)
        #     if answer_reward > 8.0:
        #         # For exceptional optimization results, make answer even more important
        #         total_reward = 0.1 * format_reward + 0.9 * answer_reward
        #     else:
        #         # For good optimization results
        #         total_reward = 0.2 * format_reward + 0.8 * answer_reward
        # else:
        #     # For negative answer rewards, penalize heavily but still consider format
        #     total_reward = 0.05 * format_reward + 0.95 * answer_reward
        
        total_reward = 0.05 * format_reward + 0.95 * answer_reward
        # Ensure reward is within acceptable bounds
        # total_reward = min(max(total_reward, -10.0), 15.0)
        
        return total_reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -10.0


# perfect_input_string = """
# <|im_start|>assistant
# <think>
# [Round 1/5]
# - Initial State: Starting optimization analysis.
# - Initial InstCount: 153.
# - Goal: Minimize instruction count.
# - Plan (Round 1): Applying the following initial passes:['--ipsccp', '--gvn-hoist', '--elim-avail-extern', '--inline', '--jump-threading', '--loop-reroll']
#   - --ipsccp: Interprocedural sparse conditional constant propagation.
#   - --gvn-hoist: Hoist redundant expressions.
#   - --elim-avail-extern: Convert available external globals to definitions.
#   - --inline: Inline function code at call sites.
#   - --jump-threading: Thread conditional jumps.
#   - --loop-reroll: Reroll unrolled loops.
# - Cumulative: Tool call will use these 6 initial passes.

# Tool call uses ALL passes applied up to the end of this round.
# </think>
# <tool_call>
# {"name":"analyze_autophase","arguments":{"filename":"train/poj104-v1/poj104-v1_47_1081.ll","optimization_passes":["--ipsccp","--gvn-hoist","--elim-avail-extern","--inline","--jump-threading","--loop-reroll"]}}
# </tool_call>
# <|im_end|>
# <|im_start|>user
# <tool_response>
# {
#   "status": "success",
#   "feature_analysis": "TotalInsts: 153 -> 65 (decrease 88), TotalMemInst: 105 -> 39 (decrease 66), testUnary: 66 -> 26 (decrease 40), NumLoadInst: 32 -> 10 (decrease 22), const32Bit: 31 -> 13 (decrease 18), Total InstCount decreased by 88",
#   "current_total_insts": 65
# }
# </tool_response>
# <|im_end|>
# <|im_start|>assistant
# <think>
# [Round 2/5]
# - Recap: Round 1 added passes: ["--ipsccp", "--gvn-hoist", "--elim-avail-extern", "--inline", "--jump-threading", "--loop-reroll"].
# - Result (after Round 1): TotalInsts: 153 -> 65 (decrease 88), TotalMemInst: 105 -> 39 (decrease 66), testUnary: 66 -> 26 (decrease 40), NumLoadInst: 32 -> 10 (decrease 22), const32Bit: 31 -> 13 (decrease 18), Total InstCount decreased by 88
# - Current InstCount: 65.
# - Goal: Continue minimizing instruction count.
# - Plan (Round 2): Positive results observed. Continuing with the planned sequence to further optimize. Adding the following passes:['--instsimplify', '--loop-instsimplify', '--mem2reg', '--prune-eh', '--lower-expect', '--reassociate']
#   - --instsimplify: Remove redundant instructions.
#   - --loop-instsimplify: Simplify instructions in loops.
#   - --mem2reg: Convert memory references to registers.
#   - --prune-eh: Remove unreachable exception handling code.
#   - --lower-expect: Lower llvm.expect intrinsics.
#   - --reassociate: Reassociate expressions.
# - Cumulative: Tool call will use all 12 passes accumulated so far.

# Tool call uses ALL passes applied up to the end of this round.
# </think>
# <tool_call>
# {"name":"analyze_autophase","arguments":{"filename":"train/poj104-v1/poj104-v1_47_1081.ll","optimization_passes":["--ipsccp","--gvn-hoist","--elim-avail-extern","--inline","--jump-threading","--loop-reroll","--instsimplify","--loop-instsimplify","--mem2reg","--prune-eh","--lower-expect","--reassociate"]}}
# </tool_call>
# <|im_end|>
# <|im_start|>user
# <tool_response>
# {
#   "status": "success",
#   "feature_analysis": "TotalInsts: 65 -> 46 (decrease 19), TotalMemInst: 39 -> 22 (decrease 17), testUnary: 26 -> 14 (decrease 12), NumLoadInst: 10 -> 5 (decrease 5), NumStoreInst: 5 -> 0 (decrease 5), Total InstCount decreased by 19",
#   "current_total_insts": 46
# }
# </tool_response>
# <|im_end|>
# <|im_start|>assistant
# <think>
# [Round 3/5]
# - Recap: Round 2 added passes: ["--instsimplify", "--loop-instsimplify", "--mem2reg", "--prune-eh", "--lower-expect", "--reassociate"].
# - Result (after Round 2): TotalInsts: 65 -> 46 (decrease 19), TotalMemInst: 39 -> 22 (decrease 17), testUnary: 26 -> 14 (decrease 12), NumLoadInst: 10 -> 5 (decrease 5), NumStoreInst: 5 -> 0 (decrease 5), Total InstCount decreased by 19
# - Current InstCount: 46.
# - Goal: Continue minimizing instruction count.
# - Plan (Round 3): Positive results observed. Continuing with the planned sequence to further optimize. Adding the following passes:['--mem2reg', '--adce', '--jump-threading', '--adce', '--load-store-vectorizer', '--die']
#   - --mem2reg: Convert memory references to registers.
#   - --adce: Aggressively eliminate dead code.
#   - --jump-threading: Thread conditional jumps.
#   - --adce: Aggressively eliminate dead code.
#   - --load-store-vectorizer: Vectorize adjacent loads and stores.
#   - --die: Dead instruction elimination.
# - Cumulative: Tool call will use all 18 passes accumulated so far.

# Tool call uses ALL passes applied up to the end of this round.
# </think>
# <tool_call>
# {"name":"analyze_autophase","arguments":{"filename":"train/poj104-v1/poj104-v1_47_1081.ll","optimization_passes":["--ipsccp","--gvn-hoist","--elim-avail-extern","--inline","--jump-threading","--loop-reroll","--instsimplify","--loop-instsimplify","--mem2reg","--prune-eh","--lower-expect","--reassociate","--mem2reg","--adce","--jump-threading","--adce","--load-store-vectorizer","--die"]}}
# </tool_call>
# <|im_end|>
# <|im_start|>user
# <tool_response>
# {
#   "status": "success",
#   "feature_analysis": "Total InstCount unchanged",
#   "current_total_insts": 46
# }
# </tool_response>
# <|im_end|>
# <|im_start|>assistant
# <think>
# [Round 4/5]
# - Recap: Round 3 added passes: ["--mem2reg", "--adce", "--jump-threading", "--adce", "--load-store-vectorizer", "--die"].
# - Result (after Round 3): Total InstCount unchanged
# - Current InstCount: 46.
# - Goal: Continue minimizing instruction count.
# - Plan (Round 4): No significant change observed. Applying the next set of passes. Adding the following passes:['--mergefunc', '--newgvn', '--indvars', '--early-cse-memssa', '--elim-avail-extern', '--die']
#   - --mergefunc: Merge duplicate functions.
#   - --newgvn: New global value numbering.
#   - --indvars: Canonicalize loop induction variables.
#   - --early-cse-memssa: Early CSE based on memory SSA.
#   - --elim-avail-extern: Convert available external globals to definitions.
#   - --die: Dead instruction elimination.
# - Cumulative: Tool call will use all 24 passes accumulated so far.

# Tool call uses ALL passes applied up to the end of this round.
# </think>
# <tool_call>
# {"name":"analyze_autophase","arguments":{"filename":"train/poj104-v1/poj104-v1_47_1081.ll","optimization_passes":["--ipsccp","--gvn-hoist","--elim-avail-extern","--inline","--jump-threading","--loop-reroll","--instsimplify","--loop-instsimplify","--mem2reg","--prune-eh","--lower-expect","--reassociate","--mem2reg","--adce","--jump-threading","--adce","--load-store-vectorizer","--die","--mergefunc","--newgvn","--indvars","--early-cse-memssa","--elim-avail-extern","--die"]}}
# </tool_call>
# <|im_end|>
# <|im_start|>user
# <tool_response>
# {
#   "status": "success",
#   "feature_analysis": "const64Bit: 7 -> 11 (increase 4), const32Bit: 9 -> 5 (decrease 4), TotalInsts: 46 -> 42 (decrease 4), testUnary: 14 -> 11 (decrease 3), NumBitCastInst: 4 -> 2 (decrease 2), Total InstCount decreased by 4",
#   "current_total_insts": 42
# }
# </tool_response>
# <|im_end|>
# <|im_start|>assistant
# <think>
# [Round 5/5]
# - Recap: Round 4 added passes: ["--mergefunc", "--newgvn", "--indvars", "--early-cse-memssa", "--elim-avail-extern", "--die"].
# - Result (after Round 4): const64Bit: 7 -> 11 (increase 4), const32Bit: 9 -> 5 (decrease 4), TotalInsts: 46 -> 42 (decrease 4), testUnary: 14 -> 11 (decrease 3), NumBitCastInst: 4 -> 2 (decrease 2), Total InstCount decreased by 4
# - Current InstCount: 42.
# - Goal: Continue minimizing instruction count.
# - Plan (Round 5): Positive results observed. Continuing with the planned sequence to further optimize. Adding the following passes:['--sccp', '--dse', '--lower-expect', '--simplifycfg', '--bdce', '--lower-constant-intrinsics']
#   - --sccp: Sparse conditional constant propagation.
#   - --dse: Dead store elimination.
#   - --lower-expect: Lower llvm.expect intrinsics.
#   - --simplifycfg: Simplify the control flow graph.
#   - --bdce: Bit-level dead code elimination.
#   - --lower-constant-intrinsics: Lower constant intrinsics.
# - Cumulative: Tool call will use all 30 passes accumulated so far.

# Tool call uses ALL passes applied up to the end of this round.
# </think>
# <tool_call>
# {"name":"analyze_autophase","arguments":{"filename":"train/poj104-v1/poj104-v1_47_1081.ll","optimization_passes":["--ipsccp","--gvn-hoist","--elim-avail-extern","--inline","--jump-threading","--loop-reroll","--instsimplify","--loop-instsimplify","--mem2reg","--prune-eh","--lower-expect","--reassociate","--mem2reg","--adce","--jump-threading","--adce","--load-store-vectorizer","--die","--mergefunc","--newgvn","--indvars","--early-cse-memssa","--elim-avail-extern","--die","--sccp","--dse","--lower-expect","--simplifycfg","--bdce","--lower-constant-intrinsics"]}}
# </tool_call>
# <|im_end|>
# <|im_start|>user
# <tool_response>
# {
#   "status": "success",
#   "feature_analysis": "Total InstCount unchanged",
#   "current_total_insts": 42
# }
# </tool_response>
# <|im_end|>
# <|im_start|>assistant
# <answer>
# ['--ipsccp', '--gvn-hoist', '--elim-avail-extern', '--inline', '--jump-threading', '--loop-reroll', '--instsimplify', '--loop-instsimplify', '--mem2reg', '--prune-eh', '--lower-expect', '--reassociate', '--mem2reg', '--adce', '--jump-threading', '--adce', '--load-store-vectorizer', '--die', '--mergefunc', '--newgvn', '--indvars', '--early-cse-memssa', '--elim-avail-extern', '--die', '--sccp', '--dse', '--lower-expect', '--simplifycfg', '--bdce', '--lower-constant-intrinsics']
# </answer>
# <|im_end|>
# """
# print("\n--- Testing V5 with Perfect Input ---")
# final_score_v5_perfect = compute_score_format(perfect_input_string)
# print(f"Final Score (V5, perfect): {final_score_v5_perfect}") # Expect high score, close 