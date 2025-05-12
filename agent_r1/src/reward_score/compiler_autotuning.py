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
        
    # try:
    #     # Try to parse as JSON directly
    #     return json.loads(sequence_str)
    # except json.JSONDecodeError:
    #     # If direct parsing fails, try to extract JSON array pattern
    #     json_array_pattern = r'\[(.*?)\]'
    #     match = re.search(json_array_pattern, sequence_str, re.DOTALL)
    #     if match:
    #         try:
    #             # Try to parse the extracted content
    #             return json.loads(f"[{match.group(1)}]")
    #         except json.JSONDecodeError:
    #             pass
                
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


# --- Assume helper functions exist (_extract_json, _extract_answer_list, _extract_think_content) ---
# --- Use the same robust helper implementations from the previous response ---
def _extract_json(tag: str, text: str) -> Optional[Dict[str, Any]]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if not match: return None
    content = match.group(1).strip()
    try:
        if content.startswith("```json"): content = content[len("```json"):].strip()
        if content.endswith("```"): content = content[:-len("```")].strip()
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError: return None

def _extract_answer_list(text: str) -> Optional[List[str]]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match: return None
    content = match.group(1).strip()
    try:
        if content.startswith("```"):
             lines = content.splitlines()
             if len(lines) > 1: content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
             else: content = content[3:-3].strip()
        parsed = ast.literal_eval(content)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
             return parsed if parsed == ['-Oz'] or (parsed and parsed != ['-Oz']) else None
        return None
    except (ValueError, SyntaxError, TypeError): return None

def _extract_think_content(text: str) -> Optional[str]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# --- Scoring Function (Additive Core + Perfection Flag - FIXED) ---
def compute_score_format(text: str) -> float:
    """
    Computes a format score. Assigns 1.0 ONLY if the sample perfectly follows
    either Path A or Path B logic and format. Otherwise, returns < 1.0.

    Args:
        text: The SFT sample string to evaluate.

    Returns:
        1.0 for a perfect path, otherwise a float score < 1.0.
    """
    if not text or not isinstance(text, str):
       # print("Format Check Failed: Input is null or not a string.")
        return 0.0

    score = 0.0
    is_perfect_so_far = True # Assume perfection until proven otherwise

    # --- Initial Checks ---
    if text.strip().startswith("<error>"):
        # print("Format Check Failed: Text starts with <error> tag.")
        is_perfect_so_far = False
        return 0.0 # Cannot be perfect, score 0
    if "<error>" in text:
         # print("Format Check Warning: Text contains an <error> tag mid-stream.")
         is_perfect_so_far = False
         # Additive score might increase, but perfection is lost
    else:
        score += 0.10 # Base points for no error tags

    has_correct_start = text.startswith("<|im_start|>assistant")
    has_correct_end = text.strip().endswith("<|im_end|>")

    if has_correct_start: score += 0.08
    else:
        # print("Format Check Failed: Text does not start with '<|im_start|>assistant'.")
        is_perfect_so_far = False
    if has_correct_end: score += 0.08
    else:
        # print("Format Check Failed: Text does not end with '<|im_end|>'.")
        is_perfect_so_far = False

    # --- Split into Turns ---
    turns_raw = re.split(r'(<\|im_start\|>)', text)
    turns = []
    if len(turns_raw) > 1:
        for i in range(1, len(turns_raw), 2):
            if i + 1 < len(turns_raw):
                turns.append(turns_raw[i] + turns_raw[i + 1])

    if not turns:
        # print("Format Check Failed: Could not split text into turns.")
        is_perfect_so_far = False
        return min(score, 0.99) if score > 0 else 0.0 # Return whatever score accumulated < 1.0

    score += 0.04 # Points for being able to split

    start_count = len(turns)
    end_count = text.count("<|im_end|>")
    if start_count == end_count:
        score += 0.10
    else:
         # print(f"Format Check Failed: Mismatched start/end markers ({start_count} starts, {end_count} ends).")
         is_perfect_so_far = False

    # --- State Variables ---
    initial_optimization_flags: Optional[List[str]] = None
    instrcount_improvement: Optional[float] = None
    find_best_improvement: Optional[float] = None
    find_best_sequence: Optional[List[str]] = None
    expected_path: Optional[str] = None
    last_tool_called: Optional[str] = None
    found_answer_tag_in_final_assistant_turn = False # Specifically tracks if <answer> was parsed in the *correct* final turn
    final_answer_content_validated = False # Specifically tracks if the *value* in <answer> was correct for the path

    # --- Turn-by-Turn Validation ---
    expected_turn_type = 'assistant'
    num_turns = len(turns)
    points_per_think_phrase = 0.02

    for i, turn_text in enumerate(turns):
        # If perfection is already lost, we still parse to calculate additive score,
        # but we don't need to perform the detailed *perfection* checks anymore.
        # However, setting the flag multiple times is harmless.

        is_last_turn = (i == num_turns - 1)
        turn_text_stripped = turn_text.strip()
        current_turn_score = 0.0 # Additive points for this turn

        # Basic turn structure
        has_turn_end_tag = turn_text_stripped.endswith("<|im_end|>")
        if has_turn_end_tag: current_turn_score += 0.01
        else:
             # print(f"Format Check Failed: Turn {i+1} missing '<|im_end|>'.")
             is_perfect_so_far = False # IMPERFECTION

        content_match = re.match(r"<\|im_start\|>(.*?)<\|im_end\|>?$", turn_text_stripped, re.DOTALL)
        if not content_match:
             # print(f"Format Check Failed: Could not extract content for Turn {i+1}.")
             is_perfect_so_far = False # IMPERFECTION
             score += current_turn_score # Add any points earned so far for this broken turn
             continue # Cannot process content
        turn_content = content_match.group(1).strip()

        # Turn Role and Payload
        actual_turn_type = None
        turn_payload = ""
        if turn_content.startswith("assistant"): actual_turn_type = 'assistant'
        elif turn_content.startswith("user"): actual_turn_type = 'user'

        if actual_turn_type:
            turn_payload = turn_content[len(actual_turn_type):].strip()
            current_turn_score += 0.01 # Points for valid role
            if actual_turn_type != expected_turn_type:
                 # print(f"Format Check Failed: Turn {i+1} expected '{expected_turn_type}' but got '{actual_turn_type}'.")
                 is_perfect_so_far = False # IMPERFECTION
                 expected_turn_type = actual_turn_type # Adjust to try parsing next turn
            else:
                 current_turn_score += 0.01 # Points for correct role sequence
        else:
            # print(f"Format Check Failed: Turn {i+1} content missing role 'assistant' or 'user'.")
            is_perfect_so_far = False # IMPERFECTION
            expected_turn_type = 'user' if expected_turn_type == 'assistant' else 'assistant' # Guess next
            score += current_turn_score
            continue # Cannot process payload

        # --- Content Validation ---
        think_content = _extract_think_content(turn_payload)
        tool_call_json = _extract_json("tool_call", turn_payload)
        tool_response_json = _extract_json("tool_response", turn_payload)
        answer_list = None # Parse later only if appropriate

        # Check for <think> tag in assistant turns (REQUIRED for perfection)
        has_think_tag = think_content is not None
        if actual_turn_type == 'assistant':
             if has_think_tag: current_turn_score += 0.01
             else:
                 # print(f"Format Check Failed: Turn {i+1} (Assistant) missing <think> tag.")
                 is_perfect_so_far = False # IMPERFECTION

        # Check for required phrases (REQUIRED for perfection)
        phrase_found = False
        required_phrase = None

        # == Turn 1: Assistant ==
        if i == 0 and actual_turn_type == 'assistant':
            required_phrase = "[Initial Pass Sequence Analysis]"
            if has_think_tag and required_phrase in think_content:
                 phrase_found = True
                 current_turn_score += points_per_think_phrase
            elif has_think_tag: 
                pass
                # print(f"Format Check Info: Turn 1 <think> missing required phrase '{required_phrase}'.")
            if not phrase_found and has_think_tag : # Only mark imperfect if think tag exists but phrase missing
                 is_perfect_so_far = False # IMPERFECTION

            tool_call_ok = False
            if tool_call_json:
                current_turn_score += 0.03
                last_tool_called = tool_call_json.get("name")
                if last_tool_called == "instrcount":
                    current_turn_score += 0.04
                    args = tool_call_json.get("arguments", {})
                    if isinstance(args, dict):
                         current_turn_score += 0.01
                         initial_optimization_flags = args.get("optimization_flags")
                         filename_ok = isinstance(args.get("filename"), str) and args["filename"]
                         flags_ok = isinstance(initial_optimization_flags, list) # Content validation later
                         if filename_ok: current_turn_score += 0.02
                         else:
                            # print(f"Format Check Failed: Turn 1 'filename' missing/invalid."); 
                            is_perfect_so_far = False # IMPERFECTION
                         if flags_ok: current_turn_score += 0.02
                         else: 
                            # print(f"Format Check Failed: Turn 1 'optimization_flags' missing/invalid."); 
                            is_perfect_so_far = False # IMPERFECTION
                         tool_call_ok = filename_ok and flags_ok
                    else: 
                        # print(f"Format Check Failed: Turn 1 <tool_call> 'arguments' not dictionary."); 
                        is_perfect_so_far = False # IMPERFECTION
                else: 
                    # print(f"Format Check Failed: Turn 1 <tool_call> name incorrect."); 
                    is_perfect_so_far = False # IMPERFECTION
            else: 
                # print(f"Format Check Failed: Turn 1 missing <tool_call>."); 
                is_perfect_so_far = False # IMPERFECTION
            expected_turn_type = 'user'

        # == Turn 2: User ==
        elif i == 1 and actual_turn_type == 'user':
            response_ok = False
            if tool_response_json:
                current_turn_score += 0.04
                status = tool_response_json.get("status")
                if status == "success":
                    current_turn_score += 0.02
                    imp = tool_response_json.get("improvement_over_oz")
                    if isinstance(imp, (int, float)):
                        current_turn_score += 0.04
                        instrcount_improvement = float(imp)
                        expected_path = 'A' if instrcount_improvement > 0 else 'B'
                        response_ok = True
                    else: 
                        # print(f"Format Check Failed: Turn 2 'improvement_over_oz' missing/invalid."); 
                        is_perfect_so_far = False # IMPERFECTION
                elif status == "error":
                    current_turn_score += 0.02
                    instrcount_improvement = -1.0
                    expected_path = 'B'
                    response_ok = True # Error is valid status, forces Path B
                    # print(f"Format Check Info: Turn 2 reported error, proceeding with Path B.")
                else: 
                    # print(f"Format Check Failed: Turn 2 invalid 'status'."); 
                    is_perfect_so_far = False # IMPERFECTION
            else: 
                # print(f"Format Check Failed: Turn 2 missing <tool_response>."); 
                is_perfect_so_far = False # IMPERFECTION
            expected_turn_type = 'assistant'

        # == Turn 3: Assistant ==
        elif i == 2 and actual_turn_type == 'assistant':
            if expected_path is None: # Error occurred in Turn 2
                 # print(f"Format Check Failed: Cannot process Turn 3 - path undetermined due to Turn 2 error.")
                 is_perfect_so_far = False # IMPERFECTION
            elif expected_path == 'A':
                required_phrase = "[Result Analysis]"
                if has_think_tag and required_phrase in think_content:
                     phrase_found = True
                     current_turn_score += points_per_think_phrase
                elif has_think_tag: 
                    pass
                    # print(f"Format Check Info: Turn 3 (Path A) <think> missing required phrase '{required_phrase}'.")
                if not phrase_found and has_think_tag: is_perfect_so_far = False # IMPERFECTION

                if tool_call_json:
                    # print(f"Format Check Failed: Turn 3 (Path A) has unexpected <tool_call>.")
                    is_perfect_so_far = False # IMPERFECTION
                else: current_turn_score += 0.02 # Correctly no tool call

                if not is_last_turn:
                     # print(f"Format Check Failed: Turn 3 (Path A) should be the last turn.")
                     is_perfect_so_far = False # IMPERFECTION
                else:
                     current_turn_score += 0.03 # Correctly the last turn
                     answer_list = _extract_answer_list(turn_payload) # Parse answer here
                     if answer_list is not None:
                         current_turn_score += 0.05 # Valid format
                         found_answer_tag_in_final_assistant_turn = True
                         if initial_optimization_flags is not None and answer_list == initial_optimization_flags:
                             current_turn_score += 0.08 # Correct value
                             final_answer_content_validated = True
                         else:
                             # print(f"Format Check Failed: Turn 3 (Path A) <answer> content mismatch (Got: {answer_list}, Expected: {initial_optimization_flags}).")
                             is_perfect_so_far = False # IMPERFECTION
                     else:
                         # print(f"Format Check Failed: Turn 3 (Path A) missing correctly formatted <answer>.")
                         is_perfect_so_far = False # IMPERFECTION

            elif expected_path == 'B':
                required_phrase = "[Finding Better Pass Sequence]"
                if has_think_tag and required_phrase in think_content:
                     phrase_found = True
                     current_turn_score += points_per_think_phrase
                elif has_think_tag: 
                    pass
                    # print(f"Format Check Info: Turn 3 (Path B) <think> missing required phrase '{required_phrase}'.")
                if not phrase_found and has_think_tag: is_perfect_so_far = False # IMPERFECTION

                # Check specifically if <answer> tag exists, should not be here
                if _extract_answer_list(turn_payload) is not None:
                    # print(f"Format Check Failed: Turn 3 (Path B) has unexpected <answer>.")
                    is_perfect_so_far = False # IMPERFECTION
                else: current_turn_score += 0.02 # Correctly no answer

                tool_call_ok = False
                if tool_call_json:
                    current_turn_score += 0.03
                    last_tool_called = tool_call_json.get("name")
                    if last_tool_called == "find_best_pass_sequence":
                        current_turn_score += 0.04
                        args = tool_call_json.get("arguments", {})
                        if isinstance(args, dict):
                            current_turn_score += 0.01
                            if isinstance(args.get("filename"), str) and args["filename"]:
                                current_turn_score += 0.02
                                tool_call_ok = True
                            else: 
                                # print(f"Format Check Failed: Turn 3 (Path B) 'filename' missing/invalid."); 
                                is_perfect_so_far = False # IMPERFECTION
                        else: 
                            # print(f"Format Check Failed: Turn 3 (Path B) 'arguments' not dictionary."); 
                            is_perfect_so_far = False # IMPERFECTION
                    else: 
                        # print(f"Format Check Failed: Turn 3 (Path B) <tool_call> name incorrect."); 
                        is_perfect_so_far = False # IMPERFECTION
                else: 
                    # print(f"Format Check Failed: Turn 3 (Path B) missing <tool_call>."); 
                    is_perfect_so_far = False # IMPERFECTION
                expected_turn_type = 'user'

        # == Turn 4: User ==
        elif i == 3 and actual_turn_type == 'user':
             if expected_path == 'B':
                 response_ok = False
                 if tool_response_json:
                     current_turn_score += 0.04
                     status = tool_response_json.get("status")
                     if status == "success":
                         current_turn_score += 0.02
                         imp = tool_response_json.get("improvement_percentage")
                         seq = tool_response_json.get("best_pass_sequence")
                         imp_ok = isinstance(imp, (int, float))
                         seq_ok = isinstance(seq, list) and all(isinstance(p, str) for p in seq)
                         if imp_ok: current_turn_score += 0.03
                         else: 
                            # print(f"Format Check Failed: Turn 4 'improvement_percentage' missing/invalid."); 
                            is_perfect_so_far = False # IMPERFECTION
                         if seq_ok: current_turn_score += 0.03
                         else: 
                            # print(f"Format Check Failed: Turn 4 'best_pass_sequence' missing/invalid format."); 
                            is_perfect_so_far = False # IMPERFECTION

                         if imp_ok and seq_ok:
                             find_best_improvement = float(imp)
                             find_best_sequence = seq
                             response_ok = True
                     elif status == "error":
                        current_turn_score += 0.02
                        find_best_improvement = -1.0 # Error implies fallback needed
                        find_best_sequence = None
                        response_ok = True
                        # print(f"Format Check Info: Turn 4 reported error, assuming Oz fallback.")
                     else: 
                        # print(f"Format Check Failed: Turn 4 invalid 'status'."); 
                        is_perfect_so_far = False # IMPERFECTION
                 else: 
                    # print(f"Format Check Failed: Turn 4 (Path B) missing <tool_response>."); 
                    is_perfect_so_far = False # IMPERFECTION
                 expected_turn_type = 'assistant'
             else: # Should not be in Turn 4 if Path A or error
                  # print(f"Format Check Failed: Unexpected Turn 4 (User) - path was '{expected_path}'.")
                  is_perfect_so_far = False # IMPERFECTION

        # == Turn 5: Assistant ==
        elif i == 4 and actual_turn_type == 'assistant':
             if expected_path == 'B':
                 # Phrase check depends on Turn 4 result state
                 required_phrase = None
                 if find_best_improvement is not None: # Only check if Turn 4 was processed ok
                     required_phrase = "[Final Decision - Found Improved Sequence]" if find_best_improvement > 0 else "[Final Decision - Fallback to -Oz]"
                     if has_think_tag and required_phrase in think_content:
                          phrase_found = True
                          current_turn_score += points_per_think_phrase
                     elif has_think_tag: 
                         pass
                        # print(f"Format Check Info: Turn 5 (Path B) <think> missing required phrase '{required_phrase}'.")
                     if not phrase_found and has_think_tag: is_perfect_so_far = False # IMPERFECTION
                 elif has_think_tag: # Turn 4 failed, cannot determine required phrase
                     # print(f"Format Check Info: Cannot check Turn 5 phrase due to Turn 4 error.")
                     is_perfect_so_far = False # Cannot be perfect if T4 failed but T5 exists

                 if tool_call_json:
                    # print(f"Format Check Failed: Turn 5 (Path B) has unexpected <tool_call>.")
                    is_perfect_so_far = False # IMPERFECTION
                 else: current_turn_score += 0.02 # Correctly no tool call

                 if not is_last_turn:
                     # print(f"Format Check Failed: Turn 5 (Path B) should be the last turn.")
                     is_perfect_so_far = False # IMPERFECTION
                 else:
                     current_turn_score += 0.03 # Correctly the last turn
                     answer_list = _extract_answer_list(turn_payload) # Parse answer here
                     if answer_list is not None:
                         current_turn_score += 0.05 # Valid format
                         found_answer_tag_in_final_assistant_turn = True
                         # Value validation depends on Turn 4 result state
                         answer_value_ok = False
                         if find_best_improvement is not None and find_best_sequence is not None and find_best_improvement > 0 :
                             if answer_list == find_best_sequence: answer_value_ok = True
                             else: 
                                 pass
                                 # print(f"Format Check Failed: Turn 5 (Path B) <answer> content mismatch (expected found sequence). Got {answer_list}, expected {find_best_sequence}")
                         elif find_best_improvement is not None: # <= 0 or error in T4
                             if answer_list == ['-Oz']: answer_value_ok = True
                             else: 
                                pass
                                 # print(f"Format Check Failed: Turn 5 (Path B) <answer> content mismatch (expected ['-Oz']). Got {answer_list}")
                         else: 
                             pass
                             # print(f"Format Check Failed: Cannot validate Turn 5 <answer> value due to Turn 4 error.") # T4 must have failed

                         if answer_value_ok:
                             current_turn_score += 0.08 # Correct value points
                             final_answer_content_validated = True
                         else:
                             is_perfect_so_far = False # IMPERFECTION (Value incorrect)

                     else: # Missing/invalid answer tag/format
                         # print(f"Format Check Failed: Turn 5 (Path B) missing correctly formatted <answer>.")
                         is_perfect_so_far = False # IMPERFECTION

             else: # Should not be in Turn 5 if Path A or error
                  # print(f"Format Check Failed: Unexpected Turn 5 (Assistant) - path was '{expected_path}'.")
                  is_perfect_so_far = False # IMPERFECTION

        # == Unexpected Turns ==
        elif i > 4 : # Any turn beyond 5 is unexpected
             # print(f"Format Check Failed: Unexpected Turn {i+1} ({actual_turn_type}). Too many turns.")
             is_perfect_so_far = False # IMPERFECTION

        score += current_turn_score # Add additive points for this turn
        # Update expectation for next turn (only relevant if loop continues)
        expected_turn_type = 'user' if actual_turn_type == 'assistant' else 'assistant'


    # --- Final Global Checks for Perfection ---
    # These act as final confirmation that the overall structure matches the path taken.
    if expected_path == 'A':
        if num_turns != 3:
            # print(f"Format Check Failed: Path A expects 3 turns, found {num_turns}.")
            is_perfect_so_far = False # IMPERFECTION
        # Final answer tag and content validation flags already checked perfection in Turn 3
        elif not found_answer_tag_in_final_assistant_turn or not final_answer_content_validated:
             # This case should have already set is_perfect_so_far=False in Turn 3 check
             # print(f"DEBUG: Final state confirms Path A answer/content was imperfect.")
             pass # Flag already set if needed

    elif expected_path == 'B':
        if num_turns != 5:
            # print(f"Format Check Failed: Path B expects 5 turns, found {num_turns}.")
            is_perfect_so_far = False # IMPERFECTION
        # Final answer tag and content validation flags already checked perfection in Turn 5
        elif not found_answer_tag_in_final_assistant_turn or not final_answer_content_validated:
             # This case should have already set is_perfect_so_far=False in Turn 5 check
             # print(f"DEBUG: Final state confirms Path B answer/content was imperfect.")
             pass # Flag already set if needed

    elif expected_path is None and num_turns > 0:
        # Path was never determined due to error, cannot be perfect if turns exist
        # print(f"Format Check Failed: Path could not be determined.")
        # is_perfect_so_far should already be False from Turn 2 failure
        is_perfect_so_far = False # Explicitly ensure it's false

    # --- Determine Final Score ---
    final_additive_score = score

    if is_perfect_so_far:
        final_additive_score = 1.5 
        # Only return 1.0 if ALL checks passed and the flag remains True
        print(f"Additive Score Accumulated: {final_additive_score:.2f}")
        return final_additive_score
    else:
        print(f"Additive Score Accumulated: {final_additive_score:.2f}")
        # Otherwise, return the additive score, capped below 1.0
        return min(final_additive_score, 1.49)

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
        
        total_reward = 0.5 * format_reward + 0.5 * answer_reward
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