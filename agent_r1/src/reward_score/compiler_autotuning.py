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
        passes = re.findall(r'--[a-zA-Z0-9-]+', sequence_str)
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


# --- Main Scoring Function (V6.5 - Baseline + 5 Rounds + Decision) ---
def compute_score_format(solution_str: str) -> float:
    """
    计算基于严格格式要求的奖励分数 (V6.5 - Baseline + 5 Rounds + Decision)。
    Evaluates a specific structure:
    1. Initial Baseline Assistant (<think>, <tool_call> with "-Oz")
    2. Initial Baseline User (<tool_response>)
    3. 5 Optimization Rounds (Assistant(<think>, <tool_call>) -> User(<tool_response>))
    4. Final Decision Assistant (<think>, <answer>)
    Rules:
    - Logs points awarded AND points *not* awarded for failed checks.
    - Specific block structure (13 blocks total) is required.
    - No <answer>...</answer> in the *final* block -> 0 score.
    - Checks are additive. Score >= 0.
    - Pass accumulation, recap vs plan, and final answer logic are checked.
    Args:
        solution_str: 模型生成的完整响应文本。

    Returns:
        格式奖励分数。
    """
    # Initialize debug log at the very beginning
    debug_log = ["[-- compute_score_format (V6.5 Baseline+5Rounds+Decision) Log --]"]
    function_name_for_log = compute_score_format.__name__ # Store function name

    # --- Constants ---
    NUM_OPTIMIZATION_ROUNDS = 5
    # V6.5: Updated required ending for optimization round think blocks
    THINK_BLOCK_OPTIMIZATION_ENDING = "Tool call analyzes the effect of applying the *cumulative* sequence generated so far (compared to previous round's state)."
    # V6.5: Required pass for baseline tool call
    BASELINE_PASS = ["-Oz"]

    # --- Point Values (Adjusted/Added for V6.5 structure) ---
    # Structure Points
    POINTS_CORRECT_BLOCK_COUNT_AND_ROLES = 3.0 # Base points for 13 blocks in sequence
    POINTS_BASELINE_ASSISTANT_STRUCTURE = 1.0 # think + tool_call
    POINTS_BASELINE_USER_STRUCTURE = 0.5     # tool_response
    POINTS_OPTIMIZATION_ROUND_STRUCTURE = 0.5 # think + tool_call + tool_response triplet (per round) = 2.5 total
    POINTS_FINAL_ASSISTANT_STRUCTURE = 1.5    # think + answer ONLY

    # Baseline Round Content Points
    POINTS_BASELINE_THINK_MARKER = 0.5
    POINTS_BASELINE_THINK_PLAN_OZ = 0.5
    POINTS_BASELINE_TOOL_CALL_PASS_OZ = 1.0

    # Optimization Round Content Points (Per Round) - Max 5 rounds * sum ~ 10.5
    POINTS_OPT_THINK_ROUND_MARKER = 0.4
    POINTS_OPT_THINK_RECAP_MATCHES_PREV_PLAN = 0.5 # New check: Recap vs Prev Plan
    POINTS_OPT_THINK_RESULT_PRESENT = 0.1
    POINTS_OPT_THINK_CURRENT_INST_PRESENT = 0.1 # New check
    POINTS_OPT_THINK_BASELINE_INST_PRESENT = 0.2 # New check (for Rounds 2-5)
    POINTS_OPT_THINK_PLAN_PRESENT_AND_PARSABLE = 0.3
    POINTS_OPT_THINK_PASS_DESCRIPTIONS = 0.2
    POINTS_OPT_THINK_ENDS_CORRECTLY = 0.3

    POINTS_TOOL_CALL_JSON_PARSE = 0.1 # Reduced value
    POINTS_TOOL_CALL_NAME_CORRECT = 0.2 # Reduced value
    POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID = 0.4 # Reduced value
    POINTS_TOOL_CALL_PASSES_LIST_FORMAT = 0.2 # Reduced value
    POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT = 1.0 # Kept high importance

    # Final Decision Block Content Points
    POINTS_FINAL_THINK_MARKER = 0.5
    POINTS_FINAL_THINK_COUNTS_PRESENT = 0.5 # Checks both counts are mentioned
    POINTS_FINAL_THINK_COMPARISON_PRESENT = 0.3
    POINTS_FINAL_THINK_CONCLUSION_PRESENT = 0.2
    POINTS_ANSWER_CORRECT_LOCATION = 1.0 # Included in FINAL_ASSISTANT_STRUCTURE now? Let's keep separate.
    POINTS_ANSWER_PYTHON_LIST_FORMAT = 1.0 # Reduced value
    POINTS_ANSWER_LOGIC_CORRECT = 4.0 # CRITICAL: Answer matches comparison logic

    # --- End Point Values ---
    # Max possible score rough estimate: 3+1+0.5+2.5+1.5 (Struct) + 0.5+0.5+1 (Baseline) + 5*(2.1 from think + 0.9 from toolcall + 1 accum) + 0.5+0.5+0.3+0.2+1+1+4 (Final)
    # = 8.5 (Struct) + 2 (Baseline) + 5*(4) (Opt Rounds) + 7.5 (Final) = 8.5 + 2 + 20 + 7.5 = 38.0 (Max)

    # --- Initial Checks ---
    if not solution_str:
        debug_log.append("    [FAIL] Solution is empty or None. Score = 0.0")
        # _save_debug_log(debug_log, solution_str if solution_str else "<Empty String>", function_name_for_log)
        return 0.0

    # 1. Extract Blocks
    all_blocks = extract_conversation_blocks(solution_str)
    num_total_blocks = len(all_blocks)
    debug_log.append(f"    [INFO] Extracted {num_total_blocks} conversation blocks.")

    # --- GATEKEEPER V6.5: Check overall structure and final answer tag ---
    EXPECTED_BLOCK_COUNT = 1 + 1 + NUM_OPTIMIZATION_ROUNDS * 2 + 1 # 13
    if num_total_blocks != EXPECTED_BLOCK_COUNT:
         debug_log.append(f"    [FAIL] Incorrect number of blocks. Expected {EXPECTED_BLOCK_COUNT}, found {num_total_blocks}. Score = 0.0")
        #  _save_debug_log(debug_log, solution_str, function_name_for_log)
         return 0.0
    debug_log.append(f"    [PASS] Correct number of blocks ({EXPECTED_BLOCK_COUNT}) found.")

    final_block_candidate = all_blocks[-1]
    answer_match_final = re.search(r'<answer>(.*?)</answer>', final_block_candidate["content"], re.IGNORECASE | re.DOTALL)
    think_present_final = '<think>' in final_block_candidate["content"]
    tool_call_present_final = '<tool_call>' in final_block_candidate["content"]

    if not (final_block_candidate["role"] == "assistant" and \
            answer_match_final and \
            think_present_final and \
            not tool_call_present_final):
        debug_log.append(f"    [FAIL] Final block structure incorrect. Role='{final_block_candidate['role']}', HasAnswer={bool(answer_match_final)}, HasThink={think_present_final}, HasToolCall={tool_call_present_final}. Expected assistant, answer, think, NO tool_call. Score = 0.0")
        # _save_debug_log(debug_log, solution_str, function_name_for_log)
        return 0.0
    debug_log.append("    [PASS] Final block basic structure (<think>, <answer>, no <tool_call>) is correct. Proceeding.")

    # --- Initialization ---
    total_score = 0.0
    structure_score = 0.0
    baseline_score = 0.0
    opt_rounds_score = 0.0
    final_decision_score = 0.0
    s_struct_log = ["[-- Section 1: Overall Structure Verification --]"]
    s_base_log = ["[-- Section 2: Baseline Round Checks --]"]
    s_opt_log = ["[-- Section 3: Optimization Rounds Checks (Rounds 1-5) --]"]
    s_final_log = ["[-- Section 4: Final Decision Block Checks --]"]

    baseline_inst_count: Optional[int] = None
    final_sequence_inst_count: Optional[int] = None
    cumulative_planned_passes: List[str] = [] # Stores passes from Plan sections FOR accumulation check
    last_opt_tool_call_passes: Optional[List[str]] = None # Passes from 5th opt tool call
    parsed_answer_passes: Optional[List[str]] = None
    all_optimization_rounds_data = [] # Store data for each opt round

    # --- Path Check Setup ---
    llvmir_dataset_path: Optional[str] = None; path_check_possible: bool = False
    try:
        # Assuming the script is somewhere within the project structure
        # Go up levels until we find 'examples/data_preprocess/llvmir_datasets'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjust these "../.." based on where the script running this function lives relative to the root
        base_path = os.path.abspath(os.path.join(script_dir, "../../..")) # Adjust as needed
        potential_path = os.path.join(base_path, "examples/data_preprocess/llvmir_datasets")
        if os.path.isdir(potential_path):
            llvmir_dataset_path = potential_path; path_check_possible = True
            debug_log.append(f"    [INFO] LLVMIR dataset path check: Enabled (Path: {llvmir_dataset_path})")
        else:
             debug_log.append(f"    [INFO] LLVMIR dataset path check: Disabled (Path not found: {potential_path})")
    except NameError: # Handle cases like interactive environments where __file__ isn't defined
         debug_log.append("    [INFO] LLVMIR dataset path check: Disabled (__file__ not defined).")
    except Exception as e:
        debug_log.append(f"    [WARN] Error finding dataset path: {e}. File checks skipped.")

    # --- Structure Verification (Detailed) ---
    current_block_index = 0
    structure_ok = True
    expected_roles = ["assistant", "user"] + ["assistant", "user"] * NUM_OPTIMIZATION_ROUNDS + ["assistant"]
    actual_roles = [block["role"] for block in all_blocks]

    if actual_roles != expected_roles:
        s_struct_log.append(f"    [FAIL] Role sequence incorrect.")
        s_struct_log.append(f"        Expected: {expected_roles}")
        s_struct_log.append(f"        Actual:   {actual_roles}")
        structure_ok = False
    else:
        s_struct_log.append(f"    [PASS] Role sequence correct.")
        # Add base points for correct roles *if* count was also okay (which it was)
        structure_score += POINTS_CORRECT_BLOCK_COUNT_AND_ROLES
        s_struct_log.append(f"    [+ {POINTS_CORRECT_BLOCK_COUNT_AND_ROLES:.2f}] Overall block count and role sequence correct.")


    # Check content tags within each block type
    # 1. Baseline Assistant
    block = all_blocks[current_block_index]
    if structure_ok and '<think>' in block["content"] and '<tool_call>' in block["content"] and '<answer>' not in block["content"]:
        structure_score += POINTS_BASELINE_ASSISTANT_STRUCTURE
        s_struct_log.append(f"    [+ {POINTS_BASELINE_ASSISTANT_STRUCTURE:.2f}] Baseline Assistant structure (<think>, <tool_call>) OK.")
    elif structure_ok: # Only log failure if roles were ok
        s_struct_log.append(f"    [+ 0.00] Baseline Assistant structure incorrect tags (Idx {current_block_index}).")
        structure_ok = False
    current_block_index += 1

    # 2. Baseline User
    block = all_blocks[current_block_index]
    if structure_ok and '<tool_response>' in block["content"] and '<think>' not in block["content"] and '<tool_call>' not in block["content"] and '<answer>' not in block["content"]:
        structure_score += POINTS_BASELINE_USER_STRUCTURE
        s_struct_log.append(f"    [+ {POINTS_BASELINE_USER_STRUCTURE:.2f}] Baseline User structure (<tool_response>) OK.")
    elif structure_ok:
        s_struct_log.append(f"    [+ 0.00] Baseline User structure incorrect tags (Idx {current_block_index}).")
        structure_ok = False
    current_block_index += 1

    # 3. Optimization Rounds (5x)
    for i in range(NUM_OPTIMIZATION_ROUNDS):
        round_num = i + 1
        if not structure_ok: break

        # Opt Assistant
        block_a = all_blocks[current_block_index]
        is_opt_assistant_ok = ('<think>' in block_a["content"] and \
                               '<tool_call>' in block_a["content"] and \
                               '<answer>' not in block_a["content"] and \
                               '<tool_response>' not in block_a["content"])

        # Opt User
        block_u = all_blocks[current_block_index + 1]
        is_opt_user_ok = ('<tool_response>' in block_u["content"] and \
                          '<think>' not in block_u["content"] and \
                          '<tool_call>' not in block_u["content"] and \
                          '<answer>' not in block_u["content"])

        if is_opt_assistant_ok and is_opt_user_ok:
            structure_score += POINTS_OPTIMIZATION_ROUND_STRUCTURE
            s_struct_log.append(f"    [+ {POINTS_OPTIMIZATION_ROUND_STRUCTURE:.2f}] Opt Round {round_num} structure (A:<think/call>, U:<response>) OK.")
        else:
            s_struct_log.append(f"    [+ 0.00] Opt Round {round_num} structure incorrect tags (Idx {current_block_index}/{current_block_index+1}). AsstOK={is_opt_assistant_ok}, UserOK={is_opt_user_ok}")
            structure_ok = False

        current_block_index += 2

    # 4. Final Decision Assistant (Tags checked by gatekeeper, add points here if structure_ok)
    if structure_ok:
        # Gatekeeper already confirmed <think>, <answer>, no <tool_call>
        structure_score += POINTS_FINAL_ASSISTANT_STRUCTURE
        s_struct_log.append(f"    [+ {POINTS_FINAL_ASSISTANT_STRUCTURE:.2f}] Final Decision Assistant structure (<think>, <answer>, no <tool_call>) OK.")
    elif not structure_ok and actual_roles == expected_roles: # Only log failure if roles were ok but tags failed earlier
         s_struct_log.append(f"    [+ 0.00] Final Decision Assistant structure points not awarded due to earlier tag error.")
         # Note: The final block tags themselves were validated by the gatekeeper if we got this far.

    debug_log.extend(s_struct_log)
    total_score += structure_score
    debug_log.append(f"    [INFO] Section 1 Score (Structure): {structure_score:.2f}")

    # --- Content Checks ---
    # Allow content checks even if structure points were lost, as long as block count and final block were ok
    # Structure_ok might be False due to intermediate tag errors, but we can still try to score content

    # --- Section 2: Baseline Round Content ---
    try:
        baseline_assistant_block = all_blocks[0]
        baseline_user_block = all_blocks[1]

        # Baseline Think Content
        think_content_match = re.search(r'<think>(.*?)</think>', baseline_assistant_block['content'], re.DOTALL)
        if think_content_match:
            think_text = think_content_match.group(1).strip()
            # Check Marker
            if '[Initial Baseline Check]' in think_text:
                baseline_score += POINTS_BASELINE_THINK_MARKER
                s_base_log.append(f"    [+ {POINTS_BASELINE_THINK_MARKER:.2f}] Think: '[Initial Baseline Check]' marker found.")
            else: s_base_log.append(f"    [+ 0.00] Think: '[Initial Baseline Check]' marker missing.")
            # Check Plan mentions -Oz explicitly in a list format
            if re.search(r"Plan:.*?['\"]-Oz['\"]", think_text, re.IGNORECASE | re.DOTALL):
                 baseline_score += POINTS_BASELINE_THINK_PLAN_OZ
                 s_base_log.append(f"    [+ {POINTS_BASELINE_THINK_PLAN_OZ:.2f}] Think: Plan mentions '-Oz'.")
            else: s_base_log.append(f"    [+ 0.00] Think: Plan does not mention '-Oz' correctly.")
        else: s_base_log.append(f"    [+ 0.00] Think: <think> tag missing in baseline assistant.")

        # Baseline Tool Call Content
        tool_call_content_match = re.search(r'<tool_call>(.*?)</tool_call>', baseline_assistant_block['content'], re.DOTALL)
        if tool_call_content_match:
            tool_call_text = tool_call_content_match.group(1).strip()
            try:
                tool_call_data = json.loads(tool_call_text)
                baseline_score += POINTS_TOOL_CALL_JSON_PARSE # Give baseline json points here
                s_base_log.append(f"    [+ {POINTS_TOOL_CALL_JSON_PARSE:.2f}] Tool Call: JSON parsed.")

                # Check Name
                if tool_call_data.get("name") == "analyze_autophase":
                    baseline_score += POINTS_TOOL_CALL_NAME_CORRECT # Give baseline name points here
                    s_base_log.append(f"    [+ {POINTS_TOOL_CALL_NAME_CORRECT:.2f}] Tool Call: Name 'analyze_autophase' correct.")
                else: s_base_log.append(f"    [+ 0.00] Tool Call: Name incorrect ('{tool_call_data.get('name')}').")

                args = tool_call_data.get("arguments")
                if isinstance(args, dict):
                    # Check Filename (Existence check is optional based on path_check_possible)
                    filename_value = args.get("filename")
                    filename_exists_validated = False
                    if filename_value and isinstance(filename_value, str):
                        if path_check_possible and llvmir_dataset_path:
                            # Normalize both the dataset path and the filename for robust comparison
                            norm_dataset_path = os.path.normpath(llvmir_dataset_path)
                            norm_filename = os.path.normpath(filename_value)

                            # Prevent directory traversal attacks
                            if ".." in norm_filename.split(os.path.sep):
                                 s_base_log.append(f"    [+ 0.00] Tool Call: Invalid filename path (contains '..').")
                            else:
                                ll_file_path = os.path.join(norm_dataset_path, norm_filename)
                                if os.path.exists(ll_file_path) and os.path.isfile(ll_file_path):
                                    baseline_score += POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID # Baseline filename points
                                    s_base_log.append(f"    [+ {POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID:.2f}] Tool Call: Filename exists and valid ('{ll_file_path}').")
                                    filename_exists_validated = True
                                else: s_base_log.append(f"    [+ 0.00] Tool Call: Filename path check failed: '{ll_file_path}' (Exists: {os.path.exists(ll_file_path)}, IsFile: {os.path.isfile(ll_file_path) if os.path.exists(ll_file_path) else 'N/A'})")
                        else: s_base_log.append(f"    [INFO] Tool Call: Filename check skipped (path not available).")
                    else: s_base_log.append(f"    [+ 0.00] Tool Call: Filename missing or not string.")

                    # Check Passes == ['-Oz']
                    passes_arg = args.get("optimization_passes")
                    if passes_arg == BASELINE_PASS:
                        baseline_score += POINTS_BASELINE_TOOL_CALL_PASS_OZ
                        s_base_log.append(f"    [+ {POINTS_BASELINE_TOOL_CALL_PASS_OZ:.2f}] Tool Call: Passes exactly {BASELINE_PASS}.")
                    else: s_base_log.append(f"    [+ 0.00] Tool Call: Passes incorrect (Expected {BASELINE_PASS}, Got {passes_arg}).")
                else: s_base_log.append(f"    [+ 0.00] Tool Call: 'arguments' key missing or not dict.")
            except json.JSONDecodeError as e: s_base_log.append(f"    [+ 0.00] Tool Call: JSON parse failed ({e}).")
            except Exception as e: s_base_log.append(f"    [+ 0.00] Tool Call: Error processing baseline tool call: {e}")
        else: s_base_log.append(f"    [+ 0.00] Tool Call: <tool_call> tag missing in baseline assistant.")

        # Baseline Tool Response: Extract InstCount
        tool_response_content_match = re.search(r'<tool_response>(.*?)</tool_response>', baseline_user_block['content'], re.DOTALL)
        if tool_response_content_match:
            tool_response_text = tool_response_content_match.group(1).strip()
            try:
                tool_response_data = json.loads(tool_response_text)
                count = tool_response_data.get("current_total_insts")
                if isinstance(count, int):
                    baseline_inst_count = count
                    s_base_log.append(f"    [INFO] Baseline InstCount extracted: {baseline_inst_count}")
                else: s_base_log.append(f"    [WARN] Baseline 'current_total_insts' missing or not an integer in response: {tool_response_data}")
            except json.JSONDecodeError as e: s_base_log.append(f"    [WARN] Baseline <tool_response> JSON parse failed ({e}). Response: {tool_response_text}")
            except Exception as e: s_base_log.append(f"    [WARN] Error processing baseline tool response: {e}")
        else: s_base_log.append(f"    [WARN] Baseline <tool_response> tag missing.")

    except IndexError: s_base_log.append("    [FAIL] Baseline blocks missing (Index out of range).")
    except Exception as e: s_base_log.append(f"    [FAIL] Unexpected error processing baseline section: {e}")

    debug_log.extend(s_base_log)
    total_score += baseline_score
    debug_log.append(f"    [INFO] Section 2 Score (Baseline): {baseline_score:.2f}")

    # --- Section 3: Optimization Rounds Content ---
    previous_round_planned_passes: List[str] = [] # Store passes added in the PREVIOUS round's plan
    previous_round_inst_count: Optional[int] = baseline_inst_count # Start count after baseline
    # Reset cumulative planned passes for checking tool call accumulation
    cumulative_planned_passes_for_tool_call: List[str] = []

    for i in range(NUM_OPTIMIZATION_ROUNDS):
        round_num = i + 1
        round_score = 0.0
        s_opt_log.append(f"  --- Round {round_num} ---")
        assistant_block_index = 2 + i * 2
        user_block_index = 3 + i * 2

        try:
            assistant_block = all_blocks[assistant_block_index]
            user_block = all_blocks[user_block_index]
            round_data = {"think": None, "tool_call": None, "tool_response": None,
                          "planned_passes_this_round": [], "tool_call_passes": None,
                          "response_inst_count": None} # Store info for this round

            # Parse Assistant Block Content
            think_match = re.search(r'<think>(.*?)</think>', assistant_block['content'], re.DOTALL)
            tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', assistant_block['content'], re.DOTALL)

            # Parse User Block Content
            response_match = re.search(r'<tool_response>(.*?)</tool_response>', user_block['content'], re.DOTALL)

            planned_passes_this_round: List[str] = [] # Define here to ensure scope

            # --- Think Block Checks (Round {round_num}) ---
            if think_match:
                think_text = think_match.group(1).strip()
                round_data["think"] = think_text
                # Check Round Marker
                marker_pattern = r'\[Optimization Round\s+' + str(round_num) + r'/' + str(NUM_OPTIMIZATION_ROUNDS) + r'\]'
                if re.search(marker_pattern, think_text):
                    round_score += POINTS_OPT_THINK_ROUND_MARKER; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_ROUND_MARKER:.2f}] Think: Round marker correct.")
                else: s_opt_log.append(f"    [+ 0.00] Think: Round marker incorrect or missing (Pattern: '{marker_pattern}').")

                # Check Recap vs Previous Plan (Rounds 2-5 only)
                if round_num > 1:
                    recap_match = re.search(r"-\s+Recap:.*?passes:\s*(\[.*?\])", think_text, re.DOTALL | re.IGNORECASE)
                    if recap_match:
                        try:
                            recap_passes_str = recap_match.group(1).strip()
                            parsed_recap_passes = ast.literal_eval(recap_passes_str)
                            # Basic validation of parsed recap passes
                            if isinstance(parsed_recap_passes, list) and all(isinstance(p, str) and p.startswith('-') for p in parsed_recap_passes):
                                if parsed_recap_passes == previous_round_planned_passes:
                                    round_score += POINTS_OPT_THINK_RECAP_MATCHES_PREV_PLAN; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_RECAP_MATCHES_PREV_PLAN:.2f}] Think: Recap passes match previous plan.")
                                else: s_opt_log.append(f"    [+ 0.00] Think: Recap passes mismatch previous plan (Expected {previous_round_planned_passes}, Got {parsed_recap_passes}).")
                            else: s_opt_log.append(f"    [+ 0.00] Think: Recap passes section found but not a valid list of '-pass' strings.")
                        except (ValueError, SyntaxError, TypeError) as e: s_opt_log.append(f"    [+ 0.00] Think: Recap passes failed to parse as Python literal ({e}).")
                    else: s_opt_log.append(f"    [+ 0.00] Think: Recap section/passes missing.")
                # else: s_opt_log.append(f"    [INFO] Think: Recap check skipped for Round 1.")

                # Check Result / Initial State Present
                # Allow "Result (after Round {round_num-1})" or "Initial State:" for R1, or "Result (Analysis after Round {round_num-1})"
                result_pattern = r"Result \((after|Analysis after) Round " + str(round_num - 1) + r"\):"
                if (round_num == 1 and "Initial State:" in think_text) or \
                   (round_num > 1 and re.search(result_pattern, think_text, re.IGNORECASE)):
                     round_score += POINTS_OPT_THINK_RESULT_PRESENT; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_RESULT_PRESENT:.2f}] Think: Result/Initial State line present.")
                # Check also for the "Total InstCount decreased by" or "unchanged" from the previous round analysis
                elif round_num > 1 and ("Total InstCount decreased by" in think_text or "Total InstCount unchanged" in think_text):
                     round_score += POINTS_OPT_THINK_RESULT_PRESENT * 0.5 # Half points if specific marker missing but decrease info present
                     s_opt_log.append(f"    [+ {POINTS_OPT_THINK_RESULT_PRESENT*0.5:.2f}] Think: Result line marker missing, but InstCount change info found.")
                else: s_opt_log.append(f"    [+ 0.00] Think: Result/Initial State line missing (Round {round_num}).")


                # Check Current InstCount Present and optionally check value
                current_inst_match = re.search(r"Current InstCount \(Sequence\):\s*(\d+)", think_text)
                if current_inst_match:
                    round_score += POINTS_OPT_THINK_CURRENT_INST_PRESENT; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_CURRENT_INST_PRESENT:.2f}] Think: Current InstCount line present.")
                    # Optional: Check if the number matches the previous round's response
                    try:
                        think_current_inst = int(current_inst_match.group(1))
                        if previous_round_inst_count is not None and think_current_inst != previous_round_inst_count:
                            s_opt_log.append(f"        [WARN] Think Current InstCount ({think_current_inst}) mismatch prev response ({previous_round_inst_count}).")
                        # elif previous_round_inst_count is None:
                        #     s_opt_log.append(f"        [INFO] Cannot verify Think Current InstCount value (previous response count unknown).")
                    except: pass # Ignore errors parsing number here if format was bad
                else: s_opt_log.append(f"    [+ 0.00] Think: Current InstCount line missing.")


                # Check Baseline InstCount Present (Rounds 2-5 only) & check value
                if round_num > 1:
                    baseline_inst_match = re.search(r"Baseline InstCount \(-Oz\):\s*(\d+)", think_text)
                    if baseline_inst_match:
                        try:
                            think_baseline_inst = int(baseline_inst_match.group(1))
                            if baseline_inst_count is not None and think_baseline_inst == baseline_inst_count:
                                round_score += POINTS_OPT_THINK_BASELINE_INST_PRESENT; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_BASELINE_INST_PRESENT:.2f}] Think: Baseline InstCount line present and correct value.")
                            elif baseline_inst_count is not None: # Value mismatch
                                round_score += POINTS_OPT_THINK_BASELINE_INST_PRESENT * 0.5 # Half points if present but wrong value
                                s_opt_log.append(f"    [+ {POINTS_OPT_THINK_BASELINE_INST_PRESENT*0.5:.2f}] Think: Baseline InstCount line present (value mismatch: Think={think_baseline_inst}, Actual={baseline_inst_count}).")
                            else: # Baseline count unknown from response
                                round_score += POINTS_OPT_THINK_BASELINE_INST_PRESENT * 0.5 # Half points if present but cannot verify value
                                s_opt_log.append(f"    [+ {POINTS_OPT_THINK_BASELINE_INST_PRESENT*0.5:.2f}] Think: Baseline InstCount line present (cannot verify value).")
                        except: # Error parsing number
                            s_opt_log.append(f"    [+ 0.00] Think: Baseline InstCount line found but number unparsable.")
                    else: s_opt_log.append(f"    [+ 0.00] Think: Baseline InstCount line missing.")
                # else: s_opt_log.append(f"    [INFO] Think: Baseline InstCount check skipped for Round 1.")


                # Check Plan Present and Parsable passes for THIS round
                plan_match = re.search(r"-\s+Plan \(Round " + str(round_num) + r"\):.*?Add passes:\s*(\[.*?\])", think_text, re.DOTALL)
                if not plan_match: # Try alternate phrasing if first fails
                     plan_match = re.search(r"-\s+Plan \(Round " + str(round_num) + r"\):.*?following passes:\s*(\[.*?\])", think_text, re.DOTALL)

                if plan_match:
                    try:
                        passes_str = plan_match.group(1).strip()
                        parsed_plan_passes = ast.literal_eval(passes_str)
                        if isinstance(parsed_plan_passes, list) and all(isinstance(p, str) and p.startswith('-') for p in parsed_plan_passes):
                            planned_passes_this_round = parsed_plan_passes # Store passes *added* this round
                            round_data["planned_passes_this_round"] = planned_passes_this_round
                            # Add to cumulative list for accumulation check LATER
                            cumulative_planned_passes_for_tool_call.extend(planned_passes_this_round)
                            round_score += POINTS_OPT_THINK_PLAN_PRESENT_AND_PARSABLE; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_PLAN_PRESENT_AND_PARSABLE:.2f}] Think: Plan line present and passes parsed.")
                        else: s_opt_log.append(f"    [+ 0.00] Think: Plan passes found but not a valid list of '-pass' strings ({parsed_plan_passes}).")
                    except (ValueError, SyntaxError, TypeError) as e: s_opt_log.append(f"    [+ 0.00] Think: Plan passes failed to parse as Python literal ({e}). Passes string: '{passes_str}'")
                else: s_opt_log.append(f"    [+ 0.00] Think: Plan line (with 'Add passes:' or 'following passes:') missing or incorrect format.")

                # Check Pass Descriptions
                # Look for lines starting with '- --passname:'
                if re.search(r'^\s*-\s+--\w+:\s+.*', think_text, re.MULTILINE):
                     round_score += POINTS_OPT_THINK_PASS_DESCRIPTIONS; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_PASS_DESCRIPTIONS:.2f}] Think: Descriptions found.")
                else: s_opt_log.append(f"    [+ 0.00] Think: Pass descriptions missing or incorrect format.")

                # Check Exact Ending Text
                # Normalize whitespace slightly for robustness (replace multiple spaces/newlines with single space) then check endswith
                normalized_think_ending = ' '.join(think_text.split())
                normalized_required_ending = ' '.join(THINK_BLOCK_OPTIMIZATION_ENDING.split())
                if normalized_think_ending.endswith(normalized_required_ending):
                    round_score += POINTS_OPT_THINK_ENDS_CORRECTLY; s_opt_log.append(f"    [+ {POINTS_OPT_THINK_ENDS_CORRECTLY:.2f}] Think: Ends correctly.")
                else:
                    # Find where the difference starts for debugging
                    diff_index = -1
                    min_len = min(len(normalized_think_ending), len(normalized_required_ending))
                    for k in range(1, min_len + 1):
                        if normalized_think_ending[-k] != normalized_required_ending[-k]:
                            diff_index = k-1
                            break
                    s_opt_log.append(f"    [+ 0.00] Think: Ending incorrect.")
                    s_opt_log.append(f"        Expected end: '...{normalized_required_ending[-50:]}'")
                    s_opt_log.append(f"        Actual end:   '...{normalized_think_ending[-50:]}'")
                    if diff_index != -1:
                         s_opt_log.append(f"        Difference starts ~{diff_index} chars from end.")

            else: s_opt_log.append(f"    [+ 0.00] Think: <think> tag missing.")


            # --- Tool Call Checks (Round {round_num}) ---
            current_tool_call_passes_list: Optional[List[str]] = None
            if tool_call_match:
                tool_call_text = tool_call_match.group(1).strip()
                round_data["tool_call"] = tool_call_text
                try:
                    tool_call_data = json.loads(tool_call_text)
                    round_score += POINTS_TOOL_CALL_JSON_PARSE; s_opt_log.append(f"    [+ {POINTS_TOOL_CALL_JSON_PARSE:.2f}] Tool Call: JSON parsed.")

                    if tool_call_data.get("name") == "analyze_autophase":
                        round_score += POINTS_TOOL_CALL_NAME_CORRECT; s_opt_log.append(f"    [+ {POINTS_TOOL_CALL_NAME_CORRECT:.2f}] Tool Call: Name correct.")
                    else: s_opt_log.append(f"    [+ 0.00] Tool Call: Name incorrect ('{tool_call_data.get('name')}').")

                    args = tool_call_data.get("arguments")
                    if isinstance(args, dict):
                        filename_value = args.get("filename")
                        if filename_value and isinstance(filename_value, str):
                            # Reuse validation logic from baseline, just add points differently if needed
                            if path_check_possible and llvmir_dataset_path:
                                norm_dataset_path = os.path.normpath(llvmir_dataset_path)
                                norm_filename = os.path.normpath(filename_value)
                                if ".." not in norm_filename.split(os.path.sep):
                                    ll_file_path = os.path.join(norm_dataset_path, norm_filename)
                                    if os.path.exists(ll_file_path) and os.path.isfile(ll_file_path):
                                        round_score += POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID; s_opt_log.append(f"    [+ {POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID:.2f}] Tool Call: Filename exists.")
                                    else: s_opt_log.append(f"    [+ 0.00] Tool Call: Filename does not exist: {ll_file_path}")
                                else: s_opt_log.append(f"    [+ 0.00] Tool Call: Invalid filename path (contains '..').")
                            else: s_opt_log.append(f"    [INFO] Tool Call: Filename existence check skipped.")
                        else: s_opt_log.append(f"    [+ 0.00] Tool Call: Filename missing or not string.")

                        passes_arg = args.get("optimization_passes")
                        if isinstance(passes_arg, list) and all(isinstance(p, str) and p.startswith('-') for p in passes_arg):
                            round_score += POINTS_TOOL_CALL_PASSES_LIST_FORMAT; s_opt_log.append(f"    [+ {POINTS_TOOL_CALL_PASSES_LIST_FORMAT:.2f}] Tool Call: Passes format correct.")
                            current_tool_call_passes_list = passes_arg # Store for accumulation check
                            round_data["tool_call_passes"] = current_tool_call_passes_list
                            if round_num == NUM_OPTIMIZATION_ROUNDS:
                                last_opt_tool_call_passes = current_tool_call_passes_list # Store for final answer check
                        else: s_opt_log.append(f"    [+ 0.00] Tool Call: Passes format incorrect or missing key.")
                    else: s_opt_log.append(f"    [+ 0.00] Tool Call: Arguments key missing or not dict.")
                except json.JSONDecodeError as e: s_opt_log.append(f"    [+ 0.00] Tool Call: JSON parse failed ({e}).")
                except Exception as e: s_opt_log.append(f"    [+ 0.00] Tool Call: Error processing tool call: {e}")
            else: s_opt_log.append(f"    [+ 0.00] Tool Call: <tool_call> tag missing.")


            # --- Pass Accumulation Check (Round {round_num}) ---
            # Compares this round's tool_call passes with *all* planned passes added up to THIS round
            if current_tool_call_passes_list is not None:
                 if current_tool_call_passes_list == cumulative_planned_passes_for_tool_call:
                     round_score += POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT
                     s_opt_log.append(f"    [+ {POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT:.2f}] Accumulation: Tool call passes match cumulative planned passes.")
                 else:
                     s_opt_log.append(f"    [+ 0.00] Accumulation: Tool call passes mismatch.")
                     # Log details only if lists are not excessively long
                     if len(cumulative_planned_passes_for_tool_call) < 50 and len(current_tool_call_passes_list) < 50:
                         s_opt_log.append(f"        Expected (Cumulative Plan): {cumulative_planned_passes_for_tool_call}")
                         s_opt_log.append(f"        Got (Tool Call): {current_tool_call_passes_list}")
                     else:
                          s_opt_log.append(f"        (Lists too long to display fully)")
                          s_opt_log.append(f"        Expected Len: {len(cumulative_planned_passes_for_tool_call)}, Got Len: {len(current_tool_call_passes_list)}")
            else: s_opt_log.append(f"    [+ 0.00] Accumulation: Cannot check (Tool call passes invalid or missing).")


            # --- Tool Response Check (Round {round_num}) ---
            # Extract InstCount for next round's checks and final decision
            if response_match:
                response_text = response_match.group(1).strip()
                round_data["tool_response"] = response_text
                try:
                    response_data = json.loads(response_text)
                    count = response_data.get("current_total_insts")
                    if isinstance(count, int):
                        round_data["response_inst_count"] = count
                        previous_round_inst_count = count # Update for next round's checks
                        if round_num == NUM_OPTIMIZATION_ROUNDS:
                            final_sequence_inst_count = count # Store for final decision check
                        s_opt_log.append(f"    [INFO] Response: InstCount extracted: {count}")
                    else: s_opt_log.append(f"    [WARN] Response: 'current_total_insts' missing or not int in round {round_num}.")
                except json.JSONDecodeError as e: s_opt_log.append(f"    [WARN] Response: JSON parse failed in round {round_num} ({e}).")
                except Exception as e: s_opt_log.append(f"    [WARN] Response: Error processing tool response in round {round_num}: {e}")
            else: s_opt_log.append(f"    [WARN] Response: <tool_response> tag missing in round {round_num}.")

            # --- End of Round Updates ---
            opt_rounds_score += round_score
            all_optimization_rounds_data.append(round_data)
            # Update previous round's *planned* passes for the *next* round's recap check
            previous_round_planned_passes = planned_passes_this_round

        except IndexError:
            s_opt_log.append(f"    [FAIL] Blocks missing for Round {round_num} (IndexError at Idx {assistant_block_index} or {user_block_index}).")
            break # Stop processing rounds if structure is broken
        except Exception as e:
            s_opt_log.append(f"    [FAIL] Unexpected error processing Round {round_num}: {e}")
            # Optionally add more error details like traceback
            import traceback
            s_opt_log.append(f"      Traceback: {traceback.format_exc(limit=2)}")
            break

    debug_log.extend(s_opt_log)
    total_score += opt_rounds_score
    debug_log.append(f"    [INFO] Section 3 Score (Optimization Rounds 1-{i+1}): {opt_rounds_score:.2f}")


    # --- Section 4: Final Decision Block Content ---
    try:
        final_assistant_block = all_blocks[-1] # Index -1 should be safe due to initial block count check

        # Final Think Content
        think_content_match = re.search(r'<think>(.*?)</think>', final_assistant_block['content'], re.DOTALL)
        final_think_parsed_baseline_count: Optional[int] = None
        final_think_parsed_sequence_count: Optional[int] = None
        comparison_logic_ok = False # Did the think block comparison make sense based on its own numbers?
        conclusion_states_oz = False
        conclusion_states_sequence = False

        if think_content_match:
            think_text = think_content_match.group(1).strip()
            # Check Marker
            if '[Final Decision]' in think_text:
                final_decision_score += POINTS_FINAL_THINK_MARKER
                s_final_log.append(f"    [+ {POINTS_FINAL_THINK_MARKER:.2f}] Think: '[Final Decision]' marker found.")
            else: s_final_log.append(f"    [+ 0.00] Think: '[Final Decision]' marker missing.")

            # Check Counts Present and Parse from think text itself
            baseline_count_match = re.search(r"Baseline InstCount \(.*?(-Oz)\):\s*(\d+)", think_text, re.IGNORECASE)
            sequence_count_match = re.search(r"Final InstCount \(.*?Sequence\):\s*(\d+)", think_text, re.IGNORECASE)
            counts_present_in_think = True
            if baseline_count_match:
                try: final_think_parsed_baseline_count = int(baseline_count_match.group(2))
                except ValueError: counts_present_in_think = False; s_final_log.append(f"    [WARN] Think: Baseline count unparsable: {baseline_count_match.group(2)}")
            else: counts_present_in_think = False; s_final_log.append(f"    [INFO] Think: Baseline count line not found.")

            if sequence_count_match:
                try: final_think_parsed_sequence_count = int(sequence_count_match.group(1))
                except ValueError: counts_present_in_think = False; s_final_log.append(f"    [WARN] Think: Sequence count unparsable: {sequence_count_match.group(2)}")
            else: counts_present_in_think = False; s_final_log.append(f"    [INFO] Think: Sequence count line not found.")

            if counts_present_in_think:
                 final_decision_score += POINTS_FINAL_THINK_COUNTS_PRESENT
                 s_final_log.append(f"    [+ {POINTS_FINAL_THINK_COUNTS_PRESENT:.2f}] Think: Baseline and Final sequence InstCounts found and parsed within think.")
                 # Optional: Check if parsed counts match stored counts from responses
                 if baseline_inst_count is not None and final_think_parsed_baseline_count != baseline_inst_count:
                      s_final_log.append(f"        [WARN] Think Baseline count ({final_think_parsed_baseline_count}) mismatch stored response count ({baseline_inst_count}).")
                 if final_sequence_inst_count is not None and final_think_parsed_sequence_count != final_sequence_inst_count:
                      s_final_log.append(f"        [WARN] Think Sequence count ({final_think_parsed_sequence_count}) mismatch stored response count ({final_sequence_inst_count}).")
            else: s_final_log.append(f"    [+ 0.00] Think: Baseline or Final sequence InstCount lines missing/unparsable in think.")

            # Check Comparison Present (simple keyword check)
            if "Comparison:" in think_text:
                 final_decision_score += POINTS_FINAL_THINK_COMPARISON_PRESENT
                 s_final_log.append(f"    [+ {POINTS_FINAL_THINK_COMPARISON_PRESENT:.2f}] Think: Comparison line found.")
            else: s_final_log.append(f"    [+ 0.00] Think: Comparison line missing.")

            # Check Conclusion Present and Parse Logic stated in conclusion
            conclusion_match = re.search(r"Conclusion:\s*(.*)", think_text, re.IGNORECASE | re.DOTALL) # DOTALL to catch multiline conclusions
            if conclusion_match:
                 final_decision_score += POINTS_FINAL_THINK_CONCLUSION_PRESENT
                 s_final_log.append(f"    [+ {POINTS_FINAL_THINK_CONCLUSION_PRESENT:.2f}] Think: Conclusion line found.")
                 conclusion_text = conclusion_match.group(1).lower().strip()
                 # Determine implied logic from conclusion text
                 if "selecting '-oz'" in conclusion_text or \
                    "using '-oz'" in conclusion_text or \
                    "baseline is better" in conclusion_text or \
                    "'-oz' resulted in fewer" in conclusion_text or \
                    "chose '-oz'" in conclusion_text:
                      conclusion_states_oz = True
                      s_final_log.append(f"        [INFO] Conclusion implies choosing '-Oz'.")

                 elif "sequence is better" in conclusion_text or \
                      "multi-round sequence" in conclusion_text or \
                      "selecting the sequence" in conclusion_text or \
                      "chose the sequence" in conclusion_text:
                      conclusion_states_sequence = True
                      s_final_log.append(f"        [INFO] Conclusion implies choosing the sequence.")
                 else:
                      s_final_log.append(f"        [WARN] Conclusion text unclear about choice: '{conclusion_text[:100]}...'")


                 # Check if conclusion matches the comparison result based on numbers *in the think block*
                 if final_think_parsed_baseline_count is not None and final_think_parsed_sequence_count is not None:
                     baseline_is_better_or_equal_in_think = final_think_parsed_baseline_count <= final_think_parsed_sequence_count
                     if baseline_is_better_or_equal_in_think and conclusion_states_oz:
                         comparison_logic_ok = True
                         s_final_log.append(f"        [INFO] Conclusion correctly selects '-Oz' based on think counts ({final_think_parsed_baseline_count} <= {final_think_parsed_sequence_count}).")
                     elif not baseline_is_better_or_equal_in_think and conclusion_states_sequence:
                         comparison_logic_ok = True
                         s_final_log.append(f"        [INFO] Conclusion correctly selects sequence based on think counts ({final_think_parsed_baseline_count} > {final_think_parsed_sequence_count}).")
                     elif conclusion_states_oz or conclusion_states_sequence: # Conclusion made a choice, but it was wrong based on think counts
                          s_final_log.append(f"        [WARN] Conclusion logic mismatch based on think counts. ThinkCounts: B={final_think_parsed_baseline_count}, S={final_think_parsed_sequence_count}. Conclusion implies: {'Oz' if conclusion_states_oz else 'Sequence'}")
                     # else: conclusion was unclear
                 else: s_final_log.append(f"        [INFO] Cannot verify conclusion logic against think counts (counts missing/unparsable from think).")
            else: s_final_log.append(f"    [+ 0.00] Think: Conclusion line missing.")
        else: s_final_log.append(f"    [+ 0.00] Think: <think> tag missing in final block.")


        # Final Answer Content (answer_match_final already checked for existence by gatekeeper)
        if answer_match_final:
            # Location point awarded only if structure was fully okay up to this point
            if structure_ok: # Check the flag tracking overall structure validity
                final_decision_score += POINTS_ANSWER_CORRECT_LOCATION
                s_final_log.append(f"    [+ {POINTS_ANSWER_CORRECT_LOCATION:.2f}] Answer: Tag found in correct final block location (overall structure ok).")
            else:
                 s_final_log.append(f"    [+ 0.00] Answer: Location point not awarded (overall structure had errors).")


            answer_content_raw = answer_match_final.group(1).strip()
            # Check Format
            try:
                # Use literal_eval for safety
                parsed_answer = ast.literal_eval(answer_content_raw)
                if isinstance(parsed_answer, list) and \
                   all(isinstance(item, str) and item.startswith('-') for item in parsed_answer):
                    parsed_answer_passes = parsed_answer # Store for logic check
                    final_decision_score += POINTS_ANSWER_PYTHON_LIST_FORMAT
                    s_final_log.append(f"    [+ {POINTS_ANSWER_PYTHON_LIST_FORMAT:.2f}] Answer: Content is valid Python literal list of '-pass' strings.")
                else: s_final_log.append(f"    [+ 0.00] Answer: Content format incorrect (Not list of '-pass' strings, parsed as: {type(parsed_answer)}).")
            except (ValueError, SyntaxError, TypeError) as e:
                 s_final_log.append(f"    [+ 0.00] Answer: Content format incorrect (Not valid Python literal: {e}). Raw: '{answer_content_raw}'")
            except Exception as e: # Catch any other unexpected errors during parsing
                 s_final_log.append(f"    [+ 0.00] Answer: Unexpected error checking answer format: {e}")

            # Check Answer Logic (CRITICAL) - based on *actual* counts from responses
            # Prerequisites: Answer parsed ok, baseline count known, final seq count known, final tool call passes known
            logic_prereqs_met = (parsed_answer_passes is not None and
                                 baseline_inst_count is not None and
                                 final_sequence_inst_count is not None and
                                 last_opt_tool_call_passes is not None)

            if logic_prereqs_met:
                # Determine the objectively correct answer based on response data
                baseline_better_or_equal_actual = baseline_inst_count <= final_sequence_inst_count
                correct_answer_expected = BASELINE_PASS if baseline_better_or_equal_actual else last_opt_tool_call_passes

                if parsed_answer_passes == correct_answer_expected:
                    final_decision_score += POINTS_ANSWER_LOGIC_CORRECT
                    s_final_log.append(f"    [+ {POINTS_ANSWER_LOGIC_CORRECT:.2f}] Answer: Content matches expected logic based on response counts (Base={baseline_inst_count}, Seq={final_sequence_inst_count}).")
                else:
                    s_final_log.append(f"    [+ 0.00] Answer: Content does NOT match expected logic based on response counts.")
                    if len(correct_answer_expected) < 50 and len(parsed_answer_passes) < 50:
                         s_final_log.append(f"        Expected: {correct_answer_expected}")
                         s_final_log.append(f"        Got: {parsed_answer_passes}")
                    else:
                         s_final_log.append(f"        (Lists too long to display fully)")
                         s_final_log.append(f"        Expected Len: {len(correct_answer_expected)}, Got Len: {len(parsed_answer_passes)}")
                    s_final_log.append(f"        (Based on Base={baseline_inst_count}, Seq={final_sequence_inst_count})")
            else:
                 s_final_log.append(f"    [+ 0.00] Answer: Cannot check logic (Prerequisites failed).")
                 s_final_log.append(f"        Prereqs: ParsedAnswerOK={bool(parsed_answer_passes)}, BaselineCountOK={baseline_inst_count is not None}, FinalSeqCountOK={final_sequence_inst_count is not None}, LastToolCallOK={last_opt_tool_call_passes is not None}")

        # No 'else' needed here as gatekeeper checked for answer tag presence

    except IndexError: s_final_log.append("    [FAIL] Final decision block missing (IndexError). Should have been caught by block count check.")
    except Exception as e:
        s_final_log.append(f"    [FAIL] Unexpected error processing final decision section: {e}")
        import traceback
        s_final_log.append(f"      Traceback: {traceback.format_exc(limit=2)}")

    debug_log.extend(s_final_log)
    total_score += final_decision_score
    debug_log.append(f"    [INFO] Section 4 Score (Final Decision): {final_decision_score:.2f}")

    # --- Final Score ---
    final_score = max(0.0, round(total_score, 2))
    debug_log.append(f"\n--- Final Calculated Score (V6.5): {final_score:.2f} ---")
    # Log the max possible score for reference
    # max_score = 8.5 + 2 + 5*(POINTS_OPT_THINK_ROUND_MARKER + POINTS_OPT_THINK_RECAP_MATCHES_PREV_PLAN + POINTS_OPT_THINK_RESULT_PRESENT + POINTS_OPT_THINK_CURRENT_INST_PRESENT + POINTS_OPT_THINK_BASELINE_INST_PRESENT + POINTS_OPT_THINK_PLAN_PRESENT_AND_PARSABLE + POINTS_OPT_THINK_PASS_DESCRIPTIONS + POINTS_OPT_THINK_ENDS_CORRECTLY + POINTS_TOOL_CALL_JSON_PARSE + POINTS_TOOL_CALL_NAME_CORRECT + POINTS_TOOL_CALL_FILENAME_EXISTS_AND_VALID + POINTS_TOOL_CALL_PASSES_LIST_FORMAT + POINTS_TOOL_CALL_PASS_ACCUMULATION_CORRECT) + 7.5 # Approx based on points
    # debug_log.append(f"--- (Approx. Max Possible Score: {max_score:.2f}) ---") # Recalculate if points change significantly


    # # --- Logging ---
    # # Call the helper function to save the detailed log and the input solution
    # _save_debug_log(debug_log, solution_str, function_name_for_log)

    # print("\n".join(debug_log)) # Optional: print log to console during development
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
        
        print(f"[DEBUG] Format reward: {format_reward}, Answer reward: {answer_reward}")
        
        total_reward = 0.05 * format_reward + 0.95 * answer_reward
        # Ensure reward is within acceptable bounds
        # total_reward = min(max(total_reward, -10.0), 15.0)
        
        return total_reward
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0
