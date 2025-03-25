"""
Tool for generating LLVM optimization passes from number
"""

import json
from typing import Dict, List
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.gen_pass_from_number import Actions_LLVM_10_0_0, generate_pass

class GeneratePassTool(Tool):
    """
    Tool for generating LLVM optimization pass from a number
    """
    
    def __init__(self):
        """
        Initialize the tool for generating LLVM optimization pass
        """
        name = "generate_pass"
        description = "Generate LLVM optimization pass name from a numeric index"
        parameters = {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "The numeric index of the optimization pass (0-123)"
                }
            },
            "required": ["number"]
        }
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute the pass generation
        
        Args:
            args: Tool parameters, including:
                - "number": Numeric index of the optimization pass
            
        Returns:
            JSON string containing the generated pass
        """
        number = args.get("number")
        
        if number is None:
            return json.dumps({
                "error": "Number parameter is required",
                "status": "error"
            })
        
        try:
            # Validate the number is in range
            if not isinstance(number, int) or number < 0 or number >= len(list(Actions_LLVM_10_0_0)):
                return json.dumps({
                    "error": f"Number {number} is out of range (0-{len(list(Actions_LLVM_10_0_0))-1})",
                    "status": "error"
                })
            
            # Generate the pass
            pass_list = generate_pass(number)
            
            # Get the enum name for reference
            action_name = list(Actions_LLVM_10_0_0)[number].name
            
            return json.dumps({
                "pass": pass_list,
                "pass_name": action_name,
                "status": "success"
            })
            
        except Exception as e:
            return json.dumps({
                "error": f"Error generating pass: {str(e)}",
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
    # Create tool instance
    tool = GeneratePassTool()
    
    # Test different indices
    test_cases = [
        {"name": "Valid index", "number": 5},  # Example: Simplifycfg
        {"name": "Another valid index", "number": 50},  # Example: LoopUnroll
        {"name": "Out of range index", "number": 200}
    ]
    
    # Run test cases
    print("== LLVM Pass Generation Tool Test ==")
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        try:
            result = tool.execute({
                "number": test["number"]
            })
            
            result_dict = json.loads(result)
            
            if "error" in result_dict:
                print(f"Error: {result_dict['error']}")
            else:
                print("Pass generation successful!")
                print(f"Generated pass: {result_dict['pass']}")
                print(f"Pass name: {result_dict['pass_name']}")
            
            reward = tool.calculate_reward({"number": test["number"]}, result)
            print(f"Reward value: {reward}")
            
        except Exception as e:
            print(f"Error during test: {str(e)}") 