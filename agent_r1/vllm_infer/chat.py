#!/usr/bin/env python3
"""
Script to process a Parquet dataset, run inference, extract optimization passes,
and calculate the average OverOz score. If interaction/extraction fails,
OverOz is calculated using an empty flag list for that record.
"""

import argparse
import json
import importlib
import os
import sys
import re
import pandas as pd
import ast
import time
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from typing import List, Optional, Dict

# Assuming these imports are correct
from agent_r1.tool import ToolEnv
from agent_r1.tool.tools import _default_tools
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_instrcount
import agent_r1.vllm_infer.config as default_config

# Retry Configuration Constants
MAX_ROW_RETRIES = 10 # Number of retries for the entire row processing block
ROW_RETRY_DELAY = 60 # Seconds to wait between row processing retries
MAX_API_RETRIES_INTERNAL = 10 # Retries within get_model_response
API_RETRY_DELAY_INTERNAL = 60 # Delay within get_model_response
MAX_INTERACTION_ATTEMPTS = 10 # Max turns for model interaction within one attempt

# ANSI color codes
COLORS = {
    "info": "\033[1;34m", "success": "\033[1;32m", "warning": "\033[1;33m",
    "error": "\033[1;31m", "retry": "\033[1;36m", "reset": "\033[0m",
    "user": "\033[1;34m", "assistant": "\033[1;32m", "tool": "\033[1;33m",
    "tool_call": "\033[1;35m", "bg_user": "\033[44m", "bg_assistant": "\033[42m",
    "bg_tool": "\033[43m", "bg_tool_call": "\033[45m",
}

# --- Helper Functions (get_overOz, read_llvm_ir_file) - No changes needed ---
def get_overOz(ll_code: Optional[str], opt_flags: List[str], llvm_tools_path: Optional[str] = None) -> Optional[float]:
    """Calculates OverOz score."""
    # (Keep implementation from previous version - it handles empty list input correctly)
    if ll_code is None: return None
    if not isinstance(opt_flags, list): return None # Check if it's a list first
    # Allow empty list, but ensure all elements *if any* are strings
    if not all(isinstance(f, str) for f in opt_flags): return None

    # Treat empty list as valid input for comparison (e.g., vs -Oz)
    try:
        # Ensure flags don't contain empty strings if get_instrcount can't handle them
        # This is safe even if opt_flags is empty
        valid_opt_flags = [flag for flag in opt_flags if flag]
        # Note: get_instrcount needs to handle an empty list `valid_opt_flags` gracefully

        ic_value_result = get_instrcount(ll_code, valid_opt_flags, llvm_tools_path=llvm_tools_path)
        oz_value_result = get_instrcount(ll_code, ["-Oz"], llvm_tools_path=llvm_tools_path)

        if oz_value_result is None or ic_value_result is None: return None
        ic_value = ic_value_result.get('ic') if isinstance(ic_value_result, dict) else ic_value_result
        oz_value = oz_value_result.get('ic') if isinstance(oz_value_result, dict) else oz_value_result
        if oz_value is None or ic_value is None: return None

        try:
            oz_value = float(oz_value)
            ic_value = float(ic_value)
        except (ValueError, TypeError): return None
        if oz_value == 0: return None

        overoz = (oz_value - ic_value) / oz_value
        return overoz
    except Exception: return None

def read_llvm_ir_file(file_path: str) -> Optional[str]:
    """Reads LLVM IR code."""
    # (Keep implementation from previous version)
    try:
        with open(file_path, 'r', encoding='utf-8') as file: return file.read()
    except Exception: return None

# --- Argument Parsing and Config Loading - No changes needed ---
def parse_args():
    parser = argparse.ArgumentParser(description='Run batch inference on Parquet data and calculate OverOz.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input Parquet file')
    parser.add_argument('--llvm-ir-dir', type=str, default='/PATH_PLACEHOLDER/NIPS_Material/examples/data_preprocess/llvmir_datasets/', help='Base directory containing the LLVM IR files')
    parser.add_argument('--llvm-tools-path', type=str, default="/PATH_PLACEHOLDER/NIPS_Material/agent_r1/tool/tools/comiler_autotuning/raw_tool/", help='Path to LLVM tools directory')
    parser.add_argument('--env', type=str, default=default_config.ENV, help='Environment for tool selection')
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY, help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE, help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME, help='Model name for inference')
    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE, help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P, help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS, help='Maximum number of tokens to generate')
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY, help='Repetition penalty')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config file')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    return parser.parse_args()

def load_custom_config(config_path):
    # (Keep implementation from previous version)
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config); return custom_config

# --- get_model_response with Internal Retry Logic - No changes needed ---
def get_model_response(client, model_name, messages, env, temperature, top_p, max_tokens, repetition_penalty):
    """Send messages to the model and get the response, with internal retries."""
    # (Keep implementation from previous version)
    last_exception = None
    for attempt in range(MAX_API_RETRIES_INTERNAL + 1):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, tools=env.tool_desc,
                tool_choice="auto", temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, extra_body={"repetition_penalty": repetition_penalty,},
                stop=["</tool_call>"] # Keep or remove based on model need
            )
            return response
        except (APIError, APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_exception = e
            # print(f"{COLORS['warning']}Warning: API call failed (Internal Attempt {attempt + 1}/{MAX_API_RETRIES_INTERNAL + 1}): {e}{COLORS['reset']}")
            if attempt < MAX_API_RETRIES_INTERNAL: time.sleep(API_RETRY_DELAY_INTERNAL)
            # else: print(f"{COLORS['error']}Error: API call failed after {MAX_API_RETRIES_INTERNAL + 1} internal attempts.{COLORS['reset']}")
        except Exception as e:
            last_exception = e
            # print(f"{COLORS['error']}Error: An unexpected error occurred during the API call (Internal Attempt {attempt + 1}): {e}{COLORS['reset']}")
            break
    return None

# --- extract_answer_passes - No changes needed ---
# --- extract_answer_passes - MODIFIED ---
def extract_answer_passes(response_content: Optional[str]) -> Optional[List[str]]:
    """
    Extracts the list of passes from the <answer> tag in the response.
    It first finds the content within <answer>...</answer>, then looks for a
    Python list literal (e.g., ['--pass1', '--pass2']) within that content.
    """
    if response_content is None:
        return None

    # 1. Extract content within <answer>...</answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", response_content, re.DOTALL | re.IGNORECASE)
    if not answer_match:
        # print("Debug: <answer> tags not found in response_content.")
        return None # <answer> tags are essential

    content_within_answer_tags = answer_match.group(1).strip()
    # print(f"Debug: Content within <answer> tags: '{content_within_answer_tags}'")

    if not content_within_answer_tags:
        # print("Debug: Content within <answer> tags is empty.")
        return None

    # 2. Find a Python list literal string (e.g., "['--pass1', ...]") within the extracted content.
    # This regex looks for an opening square bracket, followed by any characters (non-greedy),
    # and then a closing square bracket. It tries to find the *last* such occurrence
    # in case there are multiple list-like strings (e.g., in thought process text).
    # A more robust approach might involve looking for the most "list-like" string
    # if multiple are found, but for now, we'll try the last one or the first one.

    # Option A: Find the first list-like string
    # list_match = re.search(r"(\[.*?\])", content_within_answer_tags, re.DOTALL)

    # Option B: Find the last list-like string (often more likely to be the final answer)
    # We can iterate through all matches and take the last one.
    list_matches = list(re.finditer(r"(\[.*?\])", content_within_answer_tags, re.DOTALL))

    if not list_matches:
        # print(f"Debug: No list literal '[]' found within: '{content_within_answer_tags}'")
        # Fallback: If no brackets, try to parse the whole content_within_answer_tags as if it's just passes
        # This handles cases where the answer is just space-separated passes without brackets.
        # However, the problem description implies a list is expected.
        # For now, let's stick to requiring the list format.
        return None

    # Try to parse the last found list-like string
    # Iterate backwards through matches to find the first one that successfully parses.
    for list_match in reversed(list_matches):
        list_str_candidate = list_match.group(1)
        # print(f"Debug: Trying to parse list candidate: '{list_str_candidate}'")
        try:
            # Basic sanity check: does it look like a list of strings?
            # This is a heuristic. `ast.literal_eval` is the main safety.
            if not (list_str_candidate.count("'") >= 2 or list_str_candidate.count('"') >= 2):
                # If no quotes, it's unlikely to be a list of strings.
                # This helps avoid parsing things like "[1, 2, 3]" if passes are expected.
                # However, an empty list "[]" is valid.
                if list_str_candidate.strip() != "[]":
                    # print(f"Debug: Candidate '{list_str_candidate}' doesn't look like a list of strings (missing quotes).")
                    continue

            pass_list = ast.literal_eval(list_str_candidate)
            if isinstance(pass_list, list):
                # Further validation: ensure all items are strings (or can be converted)
                # and filter out empty strings after stripping.
                processed_list = []
                valid_list = True
                for item in pass_list:
                    if isinstance(item, str):
                        stripped_item = item.strip()
                        if stripped_item: # Only add non-empty strings
                            processed_list.append(stripped_item)
                    else:
                        # print(f"Debug: Item '{item}' in parsed list is not a string.")
                        valid_list = False
                        break # Not a list of strings
                
                if valid_list:
                    # print(f"Debug: Successfully parsed list: {processed_list}")
                    return processed_list # Return the first successfully parsed valid list
            else:
                # print(f"Debug: ast.literal_eval result is not a list: {type(pass_list)}")
                pass # Try next match if this one wasn't a list

        except (ValueError, SyntaxError) as e:
            # print(f"Debug: ast.literal_eval failed for '{list_str_candidate}': {e}")
            pass # `ast.literal_eval` failed, try next match

    # print("Debug: No valid pass list found after trying all list-like candidates within <answer> tags.")
    return None # No valid list found in any of the candidates


# --- process_tool_calls - No changes needed ---
def process_tool_calls(response_message, messages, env, use_colors=True):
    """Process any tool calls in the response"""
    # Format the assistant's message properly
    assistant_message = {
        "role": "assistant",
        "content": response_message.content
    }
    
    # Add tool calls if any
    if response_message.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in response_message.tool_calls
        ]
    
    # Add the formatted message to the conversation
    messages.append(assistant_message)
    
    # Display assistant's response with color
    if use_colors:
        print(f"\n{COLORS['bg_assistant']} Assistant {COLORS['reset']} {COLORS['assistant']}{response_message.content}{COLORS['reset']}")
    else:
        print(f"\nAssistant: {response_message.content}")
    
    # Check if there are any tool calls
    if response_message.tool_calls:
        # Process each tool call
        for tool_call in response_message.tool_calls:
            # Pretty format the arguments for better readability
            try:
                args_dict = json.loads(tool_call.function.arguments)
                formatted_args = json.dumps(args_dict, indent=2)
            except json.JSONDecodeError:
                formatted_args = tool_call.function.arguments
            
            # Log function call details with color
            if use_colors:
                print(f"\n{COLORS['bg_tool_call']} Tool Call {COLORS['reset']} {COLORS['tool_call']}Function: {tool_call.function.name}{COLORS['reset']}")
                print(f"{COLORS['tool_call']}Arguments:{COLORS['reset']}\n{formatted_args}")
            else:
                print(f"\n[Tool Call] Function: {tool_call.function.name}")
                print(f"Arguments:\n{formatted_args}")
            
            # Execute the tool
            result = env.tool_map[tool_call.function.name].execute(json.loads(tool_call.function.arguments))
            
            # Display tool result with color
            if use_colors:
                print(f"\n{COLORS['bg_tool']} Tool {COLORS['reset']} {COLORS['tool']}{result}{COLORS['reset']}")
            else:
                print(f"\nTool: {result}")
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })
        
        return True  # There were tool calls
    else:
        return False  # No tool calls

# --- Main logic with Modified Row-Level Retry ---
def main():
    args = parse_args()
    use_colors = not args.no_color
    if args.no_color:
        for key in COLORS: COLORS[key] = ""

    config = default_config
    if args.config:
        try: config = load_custom_config(args.config); print(f"{COLORS['info']}Info: Loaded custom config.{COLORS['reset']}")
        except Exception as e: print(f"{COLORS['error']}Error loading config: {e}. Using defaults.{COLORS['reset']}")

    ENV = args.env; OPENAI_API_KEY = args.api_key; OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model; TEMPERATURE = args.temperature; TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens; REPETITION_PENALTY = args.repetition_penalty
    INSTRUCTION_FOLLOWING = config.INSTRUCTION_FOLLOWING; LLVM_IR_DIR = args.llvm_ir_dir
    LLVM_TOOLS_PATH = args.llvm_tools_path

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    try: # Initialize ToolEnv
        tools = _default_tools(ENV); env = ToolEnv(tools=tools)
        if not isinstance(env.tool_desc, list) or not all(isinstance(t, dict) and t.get("type") == "function" and "function" in t for t in env.tool_desc):
             print(f"{COLORS['error']}Error: ToolEnv format check failed.{COLORS['reset']}"); sys.exit(1)
        # print(f"{COLORS['info']}Info: ToolEnv initialized.{COLORS['reset']}")
    except Exception as e: print(f"{COLORS['error']}Error initializing ToolEnv: {e}.{COLORS['reset']}"); sys.exit(1)

    print(f"{COLORS['info']}Info: Processing file: {args.input_file}{COLORS['reset']}")
    try: # Load Parquet
        df = pd.read_parquet(args.input_file)
        print(f"{COLORS['info']}Info: Loaded {len(df)} records.{COLORS['reset']}")
    except Exception as e: print(f"{COLORS['error']}Error loading Parquet: {e}{COLORS['reset']}"); sys.exit(1)

    overoz_scores = [] # Store all calculated scores (including those from failures)
    processed_count = 0
    error_count = 0 # Count rows that failed *all* retries

    for index, row in df.iterrows():
        processed_count += 1
        print(f"\n--- Record {index+1}/{len(df)} ---")

        # --- Initial Extraction (outside retry loop) ---
        final_prompt, filename, ll_code = None, None, None
        initial_extraction_failed = False
        try:
            user_prompt = next((msg['content'] for msg in row['prompt'] if msg['role'] == 'user'), None)
            filename = row['reward_model']['ground_truth']
            if not user_prompt or not filename: raise ValueError("Missing prompt or filename")
            final_prompt = INSTRUCTION_FOLLOWING + "Question: " + user_prompt
            file_path = os.path.join(LLVM_IR_DIR, filename)
            ll_code = read_llvm_ir_file(file_path)
            if ll_code is None: raise ValueError(f"Failed to read LLVM IR: {filename}")
        except Exception as e:
            print(f"{COLORS['warning']}Warning: Initial data extraction/read failed for row {index}: {e}. Skipping permanently.{COLORS['reset']}")
            error_count += 1 # Count as failed if initial step fails
            continue

        # --- Row Processing Retry Loop ---
        current_row_retries = 0
        processing_successful = False
        final_calculated_overoz = None # Store the score for this row

        while current_row_retries <= MAX_ROW_RETRIES and not processing_successful:
            if current_row_retries > 0:
                print(f"{COLORS['retry']}Retrying record {index+1} (Attempt {current_row_retries + 1}/{MAX_ROW_RETRIES + 1})...{COLORS['reset']}")
                time.sleep(ROW_RETRY_DELAY)

            messages = [{"role": "user", "content": final_prompt}]
            final_response_content = None
            interaction_failed = False
            flags_to_use = None # Will hold extracted flags or []

            # 3 & 4: Model Interaction Loop
            interaction_attempts = 0
            while interaction_attempts < MAX_INTERACTION_ATTEMPTS:
                interaction_attempts += 1
                response = get_model_response(client, MODEL_NAME, messages, env, TEMPERATURE, TOP_P, MAX_TOKENS, REPETITION_PENALTY)
                
                if response is None or not response.choices:
                    print(f"{COLORS['warning']}Warning: Model interaction failed (API error after internal retries) on attempt {current_row_retries + 1}.{COLORS['reset']}")
                    interaction_failed = True
                    break # Failed interaction
                
                response_message = response.choices[0].message
                had_tool_calls = process_tool_calls(response_message, messages, env, use_colors)

                if not had_tool_calls:
                    final_response_content = response_message.content
                    break # Interaction successful, got final answer
            
            # Check outcome of interaction loop
            if interaction_failed:
                print(f"{COLORS['warning']}Warning: Using “-Oz” flags due to interaction failure on attempt {current_row_retries + 1}.{COLORS['reset']}")
                flags_to_use = ["-Oz"]
                error_count += 1
            elif interaction_attempts == MAX_INTERACTION_ATTEMPTS and final_response_content is None:
                print(f"{COLORS['warning']}Warning: Using “-Oz” flags due to max interaction attempts reached on attempt {current_row_retries + 1}.{COLORS['reset']}")
                flags_to_use = ["-Oz"]
                error_count += 1
            elif final_response_content is not None:
                # 5. Extract Answer Passes (only if interaction yielded a final response)
                extracted_flags = extract_answer_passes(final_response_content)
                if extracted_flags is None:
                    print(f"{COLORS['warning']}Warning: Using “-Oz” flags due to extraction failure on attempt {current_row_retries + 1}. Response:\n{final_response_content[:200]}...{COLORS['reset']}")
                    flags_to_use = ["-Oz"]
                    error_count += 1
                else:
                    flags_to_use = extracted_flags
                    print(f"{COLORS['info']}Info: Extracted flags: {flags_to_use}{COLORS['reset']}") # Log successful extraction
            else:
                # Should not happen if logic above is correct, but safeguard
                 print(f"{COLORS['warning']}Warning: Interaction ended unexpectedly. Using “-Oz” flags on attempt {current_row_retries + 1}.{COLORS['reset']}")
                 flags_to_use = ["-Oz"]
                 error_count += 1


            # 6. Calculate OverOz (Always attempt using determined flags)
            # print(f"{COLORS['info']}Info: Calculating OverOz using flags: {flags_to_use} (Attempt {current_row_retries + 1}).{COLORS['reset']}")
            overoz_value = get_overOz(ll_code, flags_to_use, llvm_tools_path=LLVM_TOOLS_PATH)

            if overoz_value is not None:
                # --- SUCCESS CASE (even if flags were []) ---
                print(f"{COLORS['success']}Success: Record {index+1} processed (Flags: {'Extracted' if flags_to_use else 'DefaultEmpty'}). OverOz = {overoz_value:.6f}{COLORS['reset']}")
                final_calculated_overoz = overoz_value # Store the successful value
                processing_successful = True # Mark success to exit retry loop
                break # Exit the while retry loop for this row
            else:
                # OverOz calculation itself failed (e.g., div by zero, internal tool error)
                print(f"{COLORS['warning']}Warning: OverOz calculation failed for flags {flags_to_use} (Attempt {current_row_retries + 1}). Retrying row process.{COLORS['reset']}")
                # Don't set final_calculated_overoz
                current_row_retries += 1
                # Continue to the next iteration of the outer retry loop

        # After the while loop finishes for a row
        if processing_successful and final_calculated_overoz is not None:
            overoz_scores.append(final_calculated_overoz) # Append the successfully calculated score
        else:
            # All retries failed to yield a valid OverOz calculation
            print(f"{COLORS['error']}Error: Failed to process record {index+1} and calculate OverOz after {MAX_ROW_RETRIES + 1} attempts. Excluding from average.{COLORS['reset']}")
            error_count += 1 # Increment final error count only if calculation failed after all retries

    # Final Summary
    print("\n" + "="*50)
    print("Batch Processing Summary")
    print("="*50)
    print(f"Total records attempted: {processed_count}")
    print(f"Records included in Average OverOz: {len(overoz_scores)}") # Count of actual scores added
    print(f"Records finally failed (excluded from avg): {error_count}")

    if overoz_scores: # Calculate average only on successfully calculated scores
        average_overoz = sum(overoz_scores) / len(overoz_scores)
        print(f"{COLORS['success']}Average OverOz Score (for included records): {average_overoz:.6f}{COLORS['reset']}")
    else:
         print(f"{COLORS['warning']}Warning: No valid OverOz scores were calculated for any record.{COLORS['reset']}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProcessing interrupted."); sys.exit(0)
    except Exception as e:
        print(f"\n{COLORS['error']}An unexpected critical error occurred: {e}{COLORS['reset']}")
        import traceback; traceback.print_exc(); sys.exit(1)





# #!/usr/bin/env python3
# """
# Script to run interactive chat inference with configurable parameters
# """

# import argparse
# import json
# import importlib
# import os
# import sys
# from openai import OpenAI

# from agent_r1.tool import ToolEnv
# from agent_r1.tool.tools import _default_tools
# import agent_r1.vllm_infer.config as default_config

# # ANSI color codes for colored output
# COLORS = {
#     "user": "\033[1;34m",      # Bold Blue
#     "assistant": "\033[1;32m",  # Bold Green
#     "tool": "\033[1;33m",       # Bold Yellow
#     "tool_call": "\033[1;35m",  # Bold Purple
#     "reset": "\033[0m",         # Reset to default
#     "bg_user": "\033[44m",      # Blue background
#     "bg_assistant": "\033[42m", # Green background
#     "bg_tool": "\033[43m",      # Yellow background
#     "bg_tool_call": "\033[45m", # Purple background
# }

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run interactive VLLM chat with configurable parameters')
    
#     # Environment and API settings
#     parser.add_argument('--env', type=str, default=default_config.ENV,
#                         help='Environment for tool selection')
#     parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY,
#                         help='OpenAI API key')
#     parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE,
#                         help='OpenAI API base URL')
#     parser.add_argument('--model', type=str, default=default_config.MODEL_NAME,
#                         help='Model name for inference')
    
#     # Model inference parameters
#     parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE,
#                         help='Temperature for sampling')
#     parser.add_argument('--top-p', type=float, default=default_config.TOP_P,
#                         help='Top-p for nucleus sampling')
#     parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS,
#                         help='Maximum number of tokens to generate')
#     parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY,
#                         help='Repetition penalty for generation')
    
#     # Config file
#     parser.add_argument('--config', type=str, default=None,
#                         help='Path to custom config file to override defaults')
    
#     # Add option to disable colors
#     parser.add_argument('--no-color', action='store_true',
#                         help='Disable colored output')
    
#     return parser.parse_args()

# def load_custom_config(config_path):
#     """Load custom configuration from a Python file"""
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
    
#     spec = importlib.util.spec_from_file_location("custom_config", config_path)
#     custom_config = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(custom_config)
#     return custom_config

# def get_model_response(client, model_name, messages, env, temperature, top_p, max_tokens, repetition_penalty):
#     """Send messages to the model and get the response"""
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         tools=env.tool_desc,
#         tool_choice="auto",
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#         extra_body={
#             "repetition_penalty": repetition_penalty,
#         },
#         stop=["</tool_call>"]
#     )
    
#     return response

# def process_tool_calls(response_message, messages, env, use_colors=True):
#     """Process any tool calls in the response"""
#     # Format the assistant's message properly
#     assistant_message = {
#         "role": "assistant",
#         "content": response_message.content
#     }
    
#     # Add tool calls if any
#     if response_message.tool_calls:
#         assistant_message["tool_calls"] = [
#             {
#                 "id": tool_call.id,
#                 "type": tool_call.type,
#                 "function": {
#                     "name": tool_call.function.name,
#                     "arguments": tool_call.function.arguments
#                 }
#             }
#             for tool_call in response_message.tool_calls
#         ]
    
#     # Add the formatted message to the conversation
#     messages.append(assistant_message)
    
#     # Display assistant's response with color
#     if use_colors:
#         print(f"\n{COLORS['bg_assistant']} Assistant {COLORS['reset']} {COLORS['assistant']}{response_message.content}{COLORS['reset']}")
#     else:
#         print(f"\nAssistant: {response_message.content}")
    
#     # Check if there are any tool calls
#     if response_message.tool_calls:
#         # Process each tool call
#         for tool_call in response_message.tool_calls:
#             # Pretty format the arguments for better readability
#             try:
#                 args_dict = json.loads(tool_call.function.arguments)
#                 formatted_args = json.dumps(args_dict, indent=2)
#             except json.JSONDecodeError:
#                 formatted_args = tool_call.function.arguments
            
#             # Log function call details with color
#             if use_colors:
#                 print(f"\n{COLORS['bg_tool_call']} Tool Call {COLORS['reset']} {COLORS['tool_call']}Function: {tool_call.function.name}{COLORS['reset']}")
#                 print(f"{COLORS['tool_call']}Arguments:{COLORS['reset']}\n{formatted_args}")
#             else:
#                 print(f"\n[Tool Call] Function: {tool_call.function.name}")
#                 print(f"Arguments:\n{formatted_args}")
            
#             # Execute the tool
#             result = env.tool_map[tool_call.function.name].execute(json.loads(tool_call.function.arguments))
            
#             # Display tool result with color
#             if use_colors:
#                 print(f"\n{COLORS['bg_tool']} Tool {COLORS['reset']} {COLORS['tool']}{result}{COLORS['reset']}")
#             else:
#                 print(f"\nTool: {result}")
            
#             # Add tool result to messages
#             messages.append({
#                 "role": "tool",
#                 "content": result,
#                 "tool_call_id": tool_call.id
#             })
        
#         return True  # There were tool calls
#     else:
#         return False  # No tool calls

# def main():
#     args = parse_args()
    
#     # Check if colors should be disabled
#     use_colors = not args.no_color
    
#     # Load custom config if provided
#     config = default_config
#     if args.config:
#         try:
#             config = load_custom_config(args.config)
#             print(f"Loaded custom config from {args.config}")
#         except Exception as e:
#             print(f"Error loading custom config: {e}")
#             print("Falling back to default config")
    
#     # Override config with command-line arguments
#     ENV = args.env
#     OPENAI_API_KEY = args.api_key
#     OPENAI_API_BASE = args.api_base
#     MODEL_NAME = args.model
#     TEMPERATURE = args.temperature
#     TOP_P = args.top_p
#     MAX_TOKENS = args.max_tokens
#     REPETITION_PENALTY = args.repetition_penalty
#     INSTRUCTION_FOLLOWING = config.INSTRUCTION_FOLLOWING
    
#     # Initialize OpenAI client
#     client = OpenAI(
#         api_key=OPENAI_API_KEY,
#         base_url=OPENAI_API_BASE,
#     )
    
#     # Set up tools
#     tools = _default_tools(ENV)
#     env = ToolEnv(tools=tools)
    
#     print(f"Starting interactive chat with model: {MODEL_NAME}")
#     print("Type 'exit', 'quit', or 'q' to end the conversation")
#     print("="*50)
    
#     # Keep conversation history
#     messages = []
#     aaa = 1
#     # Interactive chat loop
#     while aaa:
#         aaa -= 1
#         # Get user input with color
#         if use_colors:
#             print(f"\n{COLORS['bg_user']} User {COLORS['reset']} ", end="")
#         else:
#             print("\nUser: ", end="")
        
#         user_input = '''
# Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count.
# The LLVM IR code is represented by autophase features, the initial autophase features are:

# Initial instruction count: 32

# Note: When calling the 'instrcount' and 'find_best_pass_sequence' tools, use the exact filename provided above: test/cbench-v1/cbench-v1_adpcm.ll"

# Your task is to:

# Evaluate the provided Initial Candidate Pass Sequence using the instrcount tool to determine its instruction count improvement compared to the default -Oz optimization.
# If the initial sequence provides a positive improvement (improvement_over_oz > 0), recommend it as the final answer.
# If the initial sequence does not provide a positive improvement (improvement_over_oz <= 0), use the find_best_pass_sequence tool to search for a better sequence.
# If the search finds a sequence with positive improvement (improvement_percentage > 0), recommend that sequence.
# If the search tool fails to find a sequence with positive improvement, recommend the default ['-Oz'] sequence as the safest option.
# Present your reasoning step-by-step using <think> tags and tool interactions using <tool_call> and <tool_response> structure, concluding with the final recommended sequence in an <answer> tag.
# '''
        
#         # # Check if user wants to exit
#         # if user_input.lower() in ['exit', 'quit', 'q']:
#         #     print("Ending conversation. Goodbye!")
#         #     break
        
#         # # Skip empty inputs
#         # if not user_input:
#         #     continue
        
#         # Add user message to history
#         messages.append({
#             "role": "user",
#             "content": INSTRUCTION_FOLLOWING + "Question: " + user_input
#         })
        
#         # Process the conversation with possible multiple tool calls
#         has_tool_calls = True
#         while has_tool_calls:
#             response = get_model_response(
#                 client, MODEL_NAME, messages, env, 
#                 TEMPERATURE, TOP_P, MAX_TOKENS, REPETITION_PENALTY
#             )
            
#             # Get and process the response
#             response_message = response.choices[0].message
#             has_tool_calls = process_tool_calls(response_message, messages, env, use_colors)

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nConversation interrupted. Goodbye!")
#         sys.exit(0) 