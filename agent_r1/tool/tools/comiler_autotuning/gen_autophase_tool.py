"""
Tool for generating autophase features from LLVM code after applying optimization passes
"""

import json
import os
import tempfile
from typing import Dict, List, Any
import io
import subprocess
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode

class GenAutophaseTool(Tool):
    """
    Tool for generating autophase features for LLVM IR code after applying optimization passes
    """
    
    def __init__(self, llvm_tools_path=os.path.join(os.path.dirname(__file__), 'raw_tool'), 
                 llvm_ir_dir=os.path.join(os.path.dirname(__file__), '/../../../../examples/data_preprocess/llvmir_datasets/')
                 ):
        """
        Initialize the tool for generating autophase features
        
        Args:
            llvm_tools_path: Path to LLVM tools (e.g., opt).
            llvm_ir_dir: Path to directory containing LLVM IR files.
        """
        name = "gen_autophase"
        description = "Generate autophase features for LLVM IR code after applying specified optimization passes"
        parameters = {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename of the LLVM IR code"
                },
                "optimization_passes": {
                    "type": "array",
                    "description": "List of optimization passes to apply before extracting features, e.g. ['--adce', '--inline']"
                },
                "llvm_ir_dir": {
                    "type": "string",
                    "description": "(Optional) Path to directory containing LLVM IR files. If not provided, use the path from initialization."
                },
                "llvm_tools_path": {
                    "type": "string",
                    "description": "(Optional) Path to LLVM tools. If not provided, use the path from initialization."
                }
            },
            "required": ["filename", "optimization_passes"]
        }
        
        self.llvm_tools_path = llvm_tools_path
        self.llvm_ir_dir = llvm_ir_dir
        super().__init__(name, description, parameters)
    
    def read_llvm_ir_file(self, filename, llvm_ir_dir):
        """
        读取LLVM IR文件内容
        
        Args:
            filename: LLVM IR文件名
            llvm_ir_dir: LLVM IR文件所在目录
            
        Returns:
            LLVM IR代码，如果读取失败则返回None
        """
        try:
            file_path = os.path.join(llvm_ir_dir, filename.replace(" ", ""))
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading LLVM IR file: {e}")
            return None
    
    def execute(self, args: Dict) -> str:
        """
        Execute the generation of autophase features for LLVM IR code
        
        Args:
            args: Tool parameters, including:
                - "filename": Filename of the LLVM IR code
                - "optimization_passes": List of optimization passes to apply
                - "llvm_ir_dir": (Optional) Path to directory containing LLVM IR files
                - "llvm_tools_path": (Optional) Path to LLVM tools
            
        Returns:
            JSON string containing the autophase features dictionary
        """
        filename = args.get("filename", "")
        optimization_passes = args.get("optimization_passes", [])
        llvm_ir_dir = args.get("llvm_ir_dir", self.llvm_ir_dir)
        llvm_tools_path = args.get("llvm_tools_path", self.llvm_tools_path)
        
        if not filename:
            return json.dumps({"error": "Filename not provided", "status": "error"})
        
        # 读取LLVM IR文件
        input_code = self.read_llvm_ir_file(filename, llvm_ir_dir)
        if input_code is None:
            return json.dumps({"error": f"Failed to read LLVM IR file: {filename}", "status": "error"})
        
        try:
            # 如果有优化passes，首先应用它们
            if optimization_passes:
                # 调用GenerateOptimizedLLCode生成优化后的代码
                optimized_code = GenerateOptimizedLLCode(
                    input_code,
                    optimization_passes,
                    llvm_tools_path
                )
            else:
                optimized_code = input_code
            
            # 获取autophase特征
            autophase_features = get_autophase_obs(optimized_code)
            
            # 如果需要，添加额外信息
            result = {
                "autophase_features": autophase_features,
                "applied_passes": optimization_passes,
                "status": "success"
            }
            
            return json.dumps(result)

            
        except Exception as e:
            return json.dumps({
                "error": f"Error generating autophase features: {str(e)}",
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

