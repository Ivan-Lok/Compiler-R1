"""
Tool for finding the best LLVM optimization pass sequence for a given LLVM IR file.
"""

import json
import os
from typing import Dict

from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.find_best_pass_sequence import find_best_pass_sequence

class FindBestPassSequenceTool(Tool):
    """
    Tool for finding the best LLVM optimization pass sequence for a given LLVM IR file.
    The tool returns the pass sequence that produces the minimum instruction count.
    """
    
    def __init__(self, llvm_tools_path=os.path.join(os.path.dirname(__file__), 'raw_tool'),
                       llvm_ir_dir="/PATH_PLACEHOLDER/NIPS_Material/examples/data_preprocess/llvmir_datasets/"
                       ):
        """
        Initialize the tool for finding the best pass sequence
        
        Args:
            llvm_tools_path: Path to LLVM tools.
            llvm_ir_dir: Path to directory containing LLVM IR files.
        """
        name = "find_best_pass_sequence"
        description = "Find the best LLVM optimization pass sequence for a given LLVM IR file"
        parameters = {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename of the LLVM IR code"
                }
            },
            "required": ["filename"]
        }
        
        self.llvm_tools_path = llvm_tools_path
        self.llvm_ir_dir = llvm_ir_dir
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute the tool to find the best pass sequence
        
        Args:
            args: Tool parameters, including:
                - "filename": Path to the LLVM IR file
        Returns:
            JSON string containing the best pass sequence and related metrics
        """
        filename = args.get("filename", "")
        population_size = args.get("population_size", 100)
        max_length = args.get("max_length", 30)
        min_length = args.get("min_length", 10)
        max_workers = args.get("max_workers", 0)
        llvm_tools_path = args.get("llvm_tools_path", self.llvm_tools_path)
        filename = os.path.join(self.llvm_ir_dir, filename.replace(" ", ""))
        
        if not filename:
            return json.dumps({"error": "Filename not provided"})
        
        if not os.path.exists(filename):
            return json.dumps({"error": f"File not found: {filename}"})
        
        # 检查路径是否是目录而不是文件
        if os.path.isdir(filename):
            return json.dumps({"status": "error", "error": f"Provided path is a directory, not a file: {filename}"})
        
        # 检查文件扩展名是否为.ll
        if not filename.endswith('.ll'):
            return json.dumps({"status": "error", "error": f"File must be a LLVM IR file with .ll extension: {filename}"})
        
        try:
            # Call the find_best_pass_sequence function
            result = find_best_pass_sequence(
                file_path=filename,
                llvm_tools_path=llvm_tools_path,
                population_size=population_size,
                max_length=max_length,
                min_length=min_length
            )
            
            return json.dumps({
                "status": "success",
                "best_pass_sequence": result["best_pass_sequence"],
                # "instruction_count": result["instruction_count"],
                # "baseline_instruction_count": result["baseline_instruction_count"], 
                "improvement_percentage": result["improvement_percentage"],
                # "execution_time_seconds": result["execution_time_seconds"],
                # "workers_used": max_workers or (min(32, os.cpu_count() + 4)),
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Error finding best pass sequence: {str(e)}",
                "status": "error"
            })
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate the reward value for tool execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_dict = json.loads(result)
            
            # If there is an error, give a small reward
            if "error" in result_dict:
                return 0.03
            
            # Base reward for successful execution
            reward = 0.3
            
            # Additional reward based on improvement percentage
            if "improvement_percentage" in result_dict:
                improvement = result_dict["improvement_percentage"]
                # Normalize improvement to give more reward for better results
                # Assuming 30% improvement is excellent
                normalized_improvement = min(1.0, improvement / 30.0)
                reward += 0.2 * normalized_improvement
            
            return reward
            
        except Exception:
            return 0.0  # Handle errors during reward calculation
