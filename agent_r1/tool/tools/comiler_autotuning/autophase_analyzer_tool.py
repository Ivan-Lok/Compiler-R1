"""
Tool for generating and analyzing autophase features changes from LLVM code
"""

import json
import os
from typing import Dict, List, Any
from agent_r1.tool.tool_base import Tool
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode

class AutophaseAnalyzerTool(Tool):
    """
    Combined tool for generating and analyzing autophase features for LLVM IR code
    after applying optimization passes
    """
    
    def __init__(self, llvm_tools_path=os.path.join(os.path.dirname(__file__), 'raw_tool'), 
                 llvm_ir_dir="/root/Agent-R1_phl/Agent-R1/examples/data_preprocess/llvmir_datasets/"):
        """
        Initialize the combined tool
        
        Args:
            llvm_tools_path: Path to LLVM tools (e.g., opt).
            llvm_ir_dir: Path to directory containing LLVM IR files.
        """
        name = "analyze_autophase"
        description = "Generate autophase features for LLVM IR code and analyze changes after applying optimization passes"
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
    
    def analyze_feature_changes(self, prev_features_dict, new_features_dict):
        """
        分析特征变化，为下一轮优化提供依据
        
        Args:
            prev_features_dict: 上一轮的特征（字典格式）
            new_features_dict: 新的特征（字典格式）
            
        Returns:
            特征变化分析
        """
        # 检查输入格式
        if not isinstance(prev_features_dict, dict) or not isinstance(new_features_dict, dict):
            return "无法比较特征变化：输入格式不正确"
        
        analysis = []
        try:
            # 找出所有变化的特征
            changed_features = []
            
            # 检查两个字典中共有的键
            common_keys = set(prev_features_dict.keys()) & set(new_features_dict.keys())
            for key in common_keys:
                if prev_features_dict[key] != new_features_dict[key]:
                    change = new_features_dict[key] - prev_features_dict[key]
                    direction = "increase" if change > 0 else "decrease"
                    changed_features.append((key, abs(change), direction, change))
            
            # 按变化幅度排序，取前5个变化最大的特征
            changed_features.sort(key=lambda x: x[1], reverse=True)
            top_changes = changed_features[:5]
            
            # 生成分析文本：只输出变化最大的前5个特征
            for feature, change_abs, direction, change in top_changes:
                analysis.append(f"{feature}: {prev_features_dict[feature]} -> {new_features_dict[feature]} ({direction} {change_abs})")
            
            # 添加TotalInsts的变化情况
            if "TotalInsts" in common_keys:
                total_insts_change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
                if total_insts_change > 0:
                    analysis.append(f"Total InstCount increased by {total_insts_change}")
                elif total_insts_change < 0:
                    analysis.append(f"Total InstCount decreased by {abs(total_insts_change)}")
                else:
                    analysis.append("Total InstCount unchanged")

            # 特殊情况：无显著变化
            if not analysis:
                # 检查总指令数
                if "TotalInsts" in common_keys:
                    change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
                    if change != 0:
                        direction = "increase" if change > 0 else "decrease"
                        analysis.append(f"TotalInst {direction} by {abs(change)}")
                    else:
                        analysis.append("Feature changes are not obvious")
                else:
                    analysis.append("Feature changes are not obvious")
                    
        except Exception as e:
            analysis.append(f"Feature analysis error: {str(e)}")
        
        return ", ".join(analysis) if analysis else "Feature changes are not obvious"
    
    def execute(self, args: Dict) -> str:
        """
        Execute the generation and analysis of autophase features for LLVM IR code
        
        Args:
            args: Tool parameters, including:
                - "filename": Filename of the LLVM IR code
                - "optimization_passes": List of optimization passes to apply
                - "llvm_ir_dir": (Optional) Path to directory containing LLVM IR files
                - "llvm_tools_path": (Optional) Path to LLVM tools
            
        Returns:
            JSON string containing both original and optimized autophase features and their analysis
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
            # 首先生成原始的autophase特征（不应用任何优化pass）
            original_features = get_autophase_obs(input_code)
            
            # 如果有优化passes，应用它们并获取优化后的特征
            if optimization_passes:
                # 调用GenerateOptimizedLLCode生成优化后的代码
                optimized_code = GenerateOptimizedLLCode(
                    input_code,
                    optimization_passes,
                    llvm_tools_path
                )
                
                # 获取优化后的autophase特征
                optimized_features = get_autophase_obs(optimized_code)
            else:
                # 如果没有提供优化passes，优化后的特征就是原始特征
                optimized_features = original_features
            
            # 分析特征变化
            feature_analysis = self.analyze_feature_changes(original_features, optimized_features)
            
            # 构建结果
            result = {
                # "original_features": original_features,
                # "optimized_features": optimized_features,
                "feature_analysis": feature_analysis,
                # "applied_passes": optimization_passes,
                "status": "success"
            }
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({
                "error": f"Error generating and analyzing autophase features: {str(e)}",
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