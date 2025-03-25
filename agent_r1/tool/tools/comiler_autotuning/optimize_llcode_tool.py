"""
Tool for generating optimized LLVM code using LLVM's opt tool
"""

import json
import os
from typing import Dict
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode

class OptimizeLLCodeTool(Tool):
    """
    Tool for optimizing LLVM IR code using specific optimization options
    """
    
    def __init__(self, llvm_tools_path=os.path.join(os.path.dirname(__file__), 'raw_tool')):
        """
        Initialize the tool for optimizing LLVM IR code
        
        Args:
            llvm_tools_path: Path to LLVM tools (e.g., opt).
        """
        name = "optimize_llcode"
        description = "Optimize LLVM IR code using LLVM's opt tool with specified optimization option"
        parameters = {
            "type": "object",
            "properties": {
                "input_code": {
                    "type": "string",
                    "description": "Input LLVM IR code text"
                },
                "optimization_option": {
                    "type": "array",
                    "description": "Optimization option, e.g., ['-O3']. Only one option is allowed."
                },
                "llvm_tools_path": {
                    "type": "string",
                    "description": "(Optional) Path to LLVM tools. If not provided, use the path from initialization."
                }
            },
            "required": ["input_code", "optimization_option"]
        }
        
        self.llvm_tools_path = llvm_tools_path
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute the optimization of LLVM IR code
        
        Args:
            args: Tool parameters, including:
                - "input_code": Input LLVM IR code
                - "optimization_option": Optimization option
                - "llvm_tools_path": (Optional) Path to LLVM tools
            
        Returns:
            Optimized LLVM IR code
        """
        input_code = args.get("input_code", "")
        optimization_option = args.get("optimization_option", [])
        llvm_tools_path = args.get("llvm_tools_path", self.llvm_tools_path)
        
        if not input_code:
            return json.dumps({"error": "Input code not provided"})
        
        if not optimization_option:
            return json.dumps({"error": "Optimization options not provided"})
        
        try:
            # Call the GenerateOptimizedLLCode function
            optimized_code = GenerateOptimizedLLCode(
                input_code,
                optimization_option,
                llvm_tools_path
            )
            
            return json.dumps({
                "optimized_code": optimized_code,
                "status": "success"
            })
            
        except Exception as e:
            return json.dumps({
                "error": f"Error during optimization: {str(e)}",
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
            
            return 0.3
            
        except Exception:
            return 0.0  # Handle errors during reward calculation


if __name__ == "__main__":
    # Simple example of LLVM IR code
    sample_ir_code = """
define i32 @add(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

define i32 @main() {
entry:
  %call = call i32 @add(i32 5, i32 10)
  ret i32 %call
}
"""
    
    # Create tool instance, assuming LLVM tools are in the default path
    # Note: Ensure llvm_tools_path points to the correct LLVM tools path in actual use
    tool = OptimizeLLCodeTool()
    
    # Test different optimization options
    test_cases = [
        {"name": "O1 optimization", "option": ["-O1"]},
        {"name": "O2 optimization", "option": ["-O2"]},
        {"name": "mem2reg optimization", "option": ["-mem2reg"]},
    ]
    
    # Run test cases
    print("== LLVM IR Code Optimization Tool Test ==")
    print(f"Original code:\n{sample_ir_code}\n")
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        try:
            result = tool.execute({
                "input_code": sample_ir_code,
                "optimization_option": test["option"]
            })
            
            result_dict = json.loads(result)
            
            if "error" in result_dict:
                print(f"Error: {result_dict['error']}")
            else:
                print("Optimization successful!")
                print(f"Optimized code (first 10 lines):")
                # Only display the first few lines of the optimized code to avoid long output
                optimized_lines = result_dict["optimized_code"].split("\n")
                for i in range(min(10, len(optimized_lines))):
                    print(optimized_lines[i])
                if len(optimized_lines) > 10:
                    print("... (more lines omitted) ...")
            
            reward = tool.calculate_reward({"input_code": sample_ir_code, "optimization_option": test["option"]}, result)
            print(f"Reward value: {reward}\n")
            
        except Exception as e:
            print(f"Error during test: {str(e)}\n")