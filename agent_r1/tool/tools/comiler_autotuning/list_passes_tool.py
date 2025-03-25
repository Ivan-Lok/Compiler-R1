"""
Tool for listing all available LLVM optimization passes
"""

import json
from typing import Dict
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.gen_pass_from_number import Actions_LLVM_10_0_0

class ListPassesTool(Tool):
    """
    Tool for listing all available LLVM optimization passes
    """
    
    def __init__(self):
        """
        Initialize the tool for listing LLVM optimization passes
        """
        name = "list_passes"
        description = "List all available LLVM optimization passes"
        parameters = {
            "type": "object",
            "properties": {}
        }
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute the listing of all available LLVM optimization passes
        
        Args:
            args: Tool parameters (none required)
            
        Returns:
            JSON string containing the list of all available passes with their descriptions
        """
        try:
            # 获取所有的pass
            all_passes = {}
            for action in Actions_LLVM_10_0_0:
                all_passes[action.name] = action.value
            
            # 创建按字母顺序排列的pass列表
            pass_list = []
            for name, value in sorted(all_passes.items()):
                pass_list.append({
                    "name": name,
                    "pass": value
                })
            
            return json.dumps({
                "passes": pass_list,
                "total": len(pass_list),
                "status": "success"
            })
            
        except Exception as e:
            return json.dumps({
                "error": f"Error listing passes: {str(e)}",
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
    tool = ListPassesTool()
    
    # Test the tool
    print("== LLVM Optimization Passes List Tool Test ==")
    
    try:
        result = tool.execute({})
        
        result_dict = json.loads(result)
        
        if "error" in result_dict:
            print(f"Error: {result_dict['error']}")
        else:
            print("Listing passes successful!")
            print(f"Total passes: {result_dict['total']}")
            print("First 5 passes:")
            for i, pass_info in enumerate(result_dict["passes"][:5]):
                print(f"  {pass_info['name']}: {pass_info['pass']}")
            print("...")
        
        reward = tool.calculate_reward({}, result)
        print(f"Reward value: {reward}")
        
    except Exception as e:
        print(f"Error during test: {str(e)}") 