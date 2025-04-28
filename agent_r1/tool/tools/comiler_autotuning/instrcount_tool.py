"""
Tool for counting instructions in LLVM IR code with various optimizations
"""

import json
import os
from typing import Dict
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_instrcount, get_overOz

class InstrCountTool(Tool):
    """
    Tool for counting instructions in LLVM IR code with specified optimization flags
    """
    
    def __init__(self, llvm_tools_path=os.path.join(os.path.dirname(__file__), 'raw_tool'), 
                llvm_ir_dir=os.path.join(os.path.dirname(__file__), '/../../../../examples/data_preprocess/llvmir_datasets/')
                ):
        """
        Initialize the tool for counting instructions in LLVM IR code
        
        Args:
            llvm_tools_path: Path to LLVM tools (e.g., opt).
            llvm_ir_dir: Directory containing LLVM IR files.
        """
        name = "instrcount"
        description = "Count instructions in LLVM IR code after applying specified optimization flags"
        parameters = {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the LLVM IR file (will be loaded from the llvm_ir_dir)"
                },
                "optimization_flags": {
                    "type": "array",
                    "description": "LLVM optimization flags to apply before counting instructions (e.g., ['--sroa', '--early-cse'])"
                },
            },
            "required": ["filename", "optimization_flags"]
        }
        
        self.llvm_tools_path = llvm_tools_path
        self.llvm_ir_dir = llvm_ir_dir
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute instruction counting on LLVM IR code after applying optimizations
        
        Args:
            args: Tool parameters, including:
                - "filename": Name of the LLVM IR file
                - "optimization_flags": Optimization flags to apply
            
        Returns:
            JSON string containing the instruction count results
        """
        filename = args.get("filename", "")
        optimization_flags = args.get("optimization_flags", [])
        calculate_over_oz = args.get("calculate_over_oz", True)
        llvm_tools_path = args.get("llvm_tools_path", self.llvm_tools_path)
        llvm_ir_dir = args.get("llvm_ir_dir", self.llvm_ir_dir)
        
        if not filename:
            return json.dumps({"error": "Filename not provided", "status": "error"})
        
        if not optimization_flags:
            return json.dumps({"error": "Optimization flags not provided", "status": "error"})
        
        # Process the filename to get the full path
        filename = os.path.join(llvm_ir_dir, filename.replace(" ", ""))
        
        try:
            # Read the LLVM IR code from the file
            with open(filename, 'r') as f:
                input_code = f.read()
            
            # Get instruction count after applying optimization flags
            instr_count = get_instrcount(
                input_code,
                *optimization_flags,
                llvm_tools_path=llvm_tools_path
            )
            
            result = {
                # "filename": filename,
                # "instruction_count": instr_count,
                # "optimization_flags": optimization_flags,
                "status": "success"
            }
            
            # Calculate improvement over -Oz if requested
            if calculate_over_oz:
                over_oz = get_overOz(
                    input_code,
                    optimization_flags,
                    llvm_tools_path=llvm_tools_path
                )
                result["improvement_over_oz"] = over_oz
            
            return json.dumps(result)
            
        except FileNotFoundError:
            return json.dumps({
                "error": f"File not found: {filename}",
                "status": "error"
            })
        except Exception as e:
            return json.dumps({
                "error": f"Error during instruction counting: {str(e)}",
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
            
            # Give higher reward for more comprehensive analysis
            if "improvement_over_oz" in result_dict:
                return 0.5
            
            return 0.3
            
        except Exception:
            return 0.0  # Handle errors during reward calculation
