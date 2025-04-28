#!/usr/bin/env python3
"""
Tool for finding the best LLVM optimization pass sequence for a given LLVM IR file.
This tool generates multiple pass sequences, applies them to the input file, and returns
the sequence that produces the lowest instruction count. (Single-Threaded Version)
"""

import os
import sys
import json
import time
# Removed concurrent.futures import
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

# Import the required functions from existing modules
# Ensure these imports point to the correct location of your modules
try:
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_goodpasssequence import build_graph, generate_population, synerpairs
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode, get_inst_count_obs
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure the script is run from a location where 'agent_r1' package is accessible,")
    print("or adjust the import paths accordingly.")
    sys.exit(1)

def read_llvm_ir_file(file_path: str) -> str:
    """Read the LLVM IR content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def evaluate_sequence(ll_code: str, pass_sequence: List[str], llvm_tools_path: str) -> Optional[int]:
    """
    Apply the pass sequence to the LLVM IR code and return the instruction count.

    Args:
        ll_code: The LLVM IR code as a string
        pass_sequence: List of optimization passes to apply
        llvm_tools_path: Path to LLVM tools

    Returns:
        Instruction count of the optimized code, or None on error.
    """
    try:
        # Generate optimized code using the pass sequence
        optimized_code = GenerateOptimizedLLCode(ll_code, pass_sequence, llvm_tools_path)

        # Handle potential failure in optimization
        if not optimized_code:
            return None

        # Get instruction count from the optimized code
        instr_count = get_inst_count_obs(optimized_code)

        return instr_count
    except Exception as e:
        # Print minimal error for debugging if needed, but return None
        # print(f"  Error evaluating sequence {pass_sequence[:5]}...: {e}", file=sys.stderr)
        return None


# Removed worker_task function

def find_best_pass_sequence(file_path: str, llvm_tools_path: str,
                            population_size: int = 100,
                            max_length: int = 30,
                            min_length: int = 10) -> Dict: # Removed max_workers
    """
    Find the best pass sequence for the given LLVM IR file using single-threading.

    Args:
        file_path: Path to the LLVM IR file
        llvm_tools_path: Path to LLVM tools
        population_size: Number of pass sequences to generate
        max_length: Maximum length of a pass sequence
        min_length: Minimum length of a pass sequence

    Returns:
        Dictionary containing the best pass sequence and its improvement percentage.
    """
    # Removed max_workers calculation

    # Read the LLVM IR code from the file
    ll_code = read_llvm_ir_file(file_path)

    # Generate a population of pass sequences
    pass_sequences = generate_population(synerpairs, size=population_size,
                                         max_length=max_length, min_length=min_length)

    # Removed task list creation

    # Track best sequence and its instruction count
    best_sequence = None
    best_instr_count = float('inf')

    # Removed progress tracking setup

    # --- Start Sequential Evaluation ---
    for sequence in pass_sequences:
        # Directly evaluate the sequence in the main thread
        instr_count = evaluate_sequence(ll_code, sequence, llvm_tools_path)

        # Update best sequence if this one is better
        if instr_count is not None and instr_count < best_instr_count:
            best_instr_count = instr_count
            best_sequence = sequence
    # --- End Sequential Evaluation ---


    # Evaluate the baseline '-Oz' sequence
    baseline_instr_count = evaluate_sequence(ll_code, ['-Oz'], llvm_tools_path)

    # Handle cases where evaluation might fail
    if baseline_instr_count is None:
        # If baseline fails, improvement cannot be calculated meaningfully
        # Return best found sequence, or default to Oz if none worked.
        improvement = 0 # Or signal error? Keep simple for now.
        if best_sequence is None:
            best_sequence = ['-Oz'] # Default if nothing worked
    elif best_instr_count == float('inf'):
        # No generated sequence was successfully evaluated or improved
        improvement = 0
        best_sequence = ['-Oz'] # Default to Oz
    elif baseline_instr_count == 0:
         # Avoid division by zero
         improvement = 0
         if best_sequence is None: # Ensure best_sequence is not None
             best_sequence = ['-Oz']
    else:
        # Calculate improvement percentage
        improvement = ((baseline_instr_count - best_instr_count) / baseline_instr_count) * 100
        # If best_sequence is still None here (only possible if all evaluated sequences failed
        # but baseline succeeded), default to Oz.
        if best_sequence is None:
             best_sequence = ['-Oz']
             improvement = 0 # Revert improvement if we fallback


    result = {
        "best_pass_sequence": best_sequence,
        "improvement_percentage": round(improvement, 2),
    }

    return result
