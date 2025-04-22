#!/usr/bin/env python
# Copyright 2024 RAG Tool for Compiler Autotuning
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
"""
基于compiler_autotuning_data.csv的RAG搜索工具，
使用CSV文件中的Autophase_embedding和PassSequence列建立索引，
提供编译器优化Pass序列的相似性搜索。
"""

import os
import json
import ast
import numpy as np
from typing import Dict, List, Any, Optional, Union
import faiss
import pandas as pd

from agent_r1.tool.tool_base import Tool


class CompilerAutotuningSimilaritySearch:
    """
    基于编译器自动调优数据集的相似性搜索
    
    使用CSV文件中的Autophase_embedding作为嵌入向量，
    使用FAISS索引进行相似性搜索，
    根据查询找到最相似的PassSequence。
    """
    
    def __init__(self, csv_path: str):
        """
        初始化搜索系统
        
        Args:
            csv_path: 编译器自动调优数据CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.index = None
        self.embeddings = None
        self.pass_sequences = []
        
    def load_dataset(self) -> None:
        """
        加载数据集并提取Autophase_embedding作为向量
        """
        print(f"[DEBUG] Loading dataset from {self.csv_path}...")
        
        # 读取CSV文件
        self.df = pd.read_csv(self.csv_path)
        
        # 解析Autophase_embedding为NumPy数组
        embeddings = []
        pass_sequences = []
        
        for _, row in self.df.iterrows():
            try:
                # 解析Autophase_embedding
                embedding_str = row['Autophase_embedding']
                
                # 解析字符串表示的嵌入向量
                try:
                    # 尝试解析为JSON（可能是字典格式）
                    embedding_data = json.loads(embedding_str)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试以Python字面量解析
                    embedding_data = ast.literal_eval(embedding_str)
                
                # 将嵌入向量转换为标准格式
                if isinstance(embedding_data, dict):
                    # 如果是字典格式，转换为向量
                    embedding = self.dict_to_vector(embedding_data)
                else:
                    # 如果是列表格式，直接转换为NumPy数组
                    embedding = np.array(embedding_data, dtype=np.float32)
                
                # 解析PassSequence
                pass_sequence_str = row['PassSequence']
                pass_sequence = ast.literal_eval(pass_sequence_str)
                
                embeddings.append(embedding)
                pass_sequences.append(pass_sequence)
                
            except Exception as e:
                print(f"[WARNING] Error processing row: {e}")
                continue
        
        # 转换为NumPy数组
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.pass_sequences = pass_sequences
        
        print(f"[DEBUG] Loaded {len(self.embeddings)} samples")
    
    def build_index(self) -> None:
        """
        构建FAISS索引
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available. Call load_dataset() first.")
        
        print("[DEBUG] Building FAISS index...")
        
        # 获取嵌入维度
        dim = self.embeddings.shape[1]
        
        # 创建使用内积(cosine similarity)的索引
        self.index = faiss.IndexFlatIP(dim)
        
        # 归一化向量以便使用内积进行余弦相似度计算
        faiss.normalize_L2(self.embeddings)
        
        # 将向量添加到索引
        self.index.add(self.embeddings)
        
        print("[DEBUG] FAISS index built successfully")
    
    def format_autophase_dict(self, autophase_embedding: np.ndarray) -> Dict:
        """
        将Autophase嵌入向量格式化为字典
        
        Args:
            autophase_embedding: Autophase嵌入向量
            
        Returns:
            格式化的字典
        """
        # 定义Autophase特征名称
        feature_names = [
            "TotalInsts", "IntInsts", "FloatInsts", "BranchInsts", "VectorInsts",
            "CallInsts", "PHIInsts", "AllocaInsts", "LoadInsts", "StoreInsts",
            "GetElemPtrInsts", "CmpInsts", "SelectInsts", "BBs", "CFGEdges",
            "CallsitesInvokes", "GlobalVars", "LocalVars", "DirectCalls", "IndirectCalls",
            "IntArgs", "FloatArgs", "Instructions", "BinaryOps", "UnaryOps",
            "SelectionsRets", "Insns", "Switches", "IntConstOperands", "FloatConstOperands",
            "ExternalSymOperands", "ConstantOperands", "MemIntrinsicInsts", "MemoryOperatedInsts",
            "MemoryAddressingInsts", "ExternalFunctions", "DynamicMemoryOps", "NonLocalLoads",
            "NonLocalStores", "UnknownInsts", "TextSize", "DataSize", "BSSSize",
            "FunctionCount", "DeadInsts", "NonTerminator", "DerivedVars", "TotalUses",
            "UniqueName", "UniqueConstants", "BasicBlocks", "FunctionMeanSize", "FunMaxSize",
            "FunMeanUses", "FunMaxUses", "IntConstZero", "FloatConstZero"
        ]
        
        # 创建特征字典
        feature_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(autophase_embedding):
                feature_dict[name] = int(autophase_embedding[i])
        
        return feature_dict
    
    # Helper method to convert dictionary to list in the standard order
    def dict_to_vector(self, autophase_dict: Dict) -> np.ndarray:
        """
        Convert autophase dictionary to vector in the standard order
        
        Args:
            autophase_dict: Dictionary containing autophase features
            
        Returns:
            NumPy array with features in the standard order
        """
        # Standard order of autophase features
        feature_names = ['BBNumArgsHi', 'BBNumArgsLo', 'onePred', 'onePredOneSuc', 'onePredTwoSuc', 
                         'oneSuccessor', 'twoPred', 'twoPredOneSuc', 'twoEach', 'twoSuccessor', 'morePreds', 
                         'BB03Phi', 'BBHiPhi', 'BBNoPhi', 'BeginPhi', 'BranchCount', 'returnInt', 'CriticalCount', 
                         'NumEdges', 'const32Bit', 'const64Bit', 'numConstZeroes', 'numConstOnes', 'UncondBranches', 
                         'binaryConstArg', 'NumAShrInst', 'NumAddInst', 'NumAllocaInst', 'NumAndInst', 'BlockMid', 
                         'BlockLow', 'NumBitCastInst', 'NumBrInst', 'NumCallInst', 'NumGetElementPtrInst', 
                         'NumICmpInst', 'NumLShrInst', 'NumLoadInst', 'NumMulInst', 'NumOrInst', 'NumPHIInst', 
                         'NumRetInst', 'NumSExtInst', 'NumSelectInst', 'NumShlInst', 'NumStoreInst', 'NumSubInst', 
                         'NumTruncInst', 'NumXorInst', 'NumZExtInst', 'TotalBlocks', 'TotalInsts', 'TotalMemInst', 
                         'TotalFuncs', 'ArgsPhi', 'testUnary'
                         ]
        
        # Convert dictionary to array
        vector = []
        for feature in feature_names:
            if feature in autophase_dict:
                vector.append(float(autophase_dict[feature]))
            else:
                vector.append(0.0)  # Default value for missing features
        
        return np.array(vector, dtype=np.float32)
        
    def search(self, query_autophase: Union[List, np.ndarray, Dict], top_k: int = 3) -> List[Dict]:
        """
        使用Autophase特征搜索最相似的编译器优化Pass序列
        
        Args:
            query_autophase: 查询的Autophase嵌入向量，可以是列表、NumPy数组或字典
            top_k: 返回的结果数量
            
        Returns:
            最相似的Pass序列及相似度
        """
        if self.index is None:
            raise ValueError("No index available. Call build_index() first.")
        
        # Convert query to NumPy array
        if isinstance(query_autophase, dict):
            # Convert dictionary to vector in the standard order
            query_vector = np.array([self.dict_to_vector(query_autophase)], dtype=np.float32)
            
        elif isinstance(query_autophase, list):
            query_vector = np.array([query_autophase], dtype=np.float32)
        else:
            query_vector = np.array([query_autophase], dtype=np.float32)
        
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Execute search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            
            if idx >= 0 and idx < len(self.pass_sequences):
                result = {
                    "similarity": float(distance),
                    "pass_sequence": self.pass_sequences[idx],
                    "index": int(idx)
                }
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """
        保存FAISS索引
        
        Args:
            index_path: 索引保存路径
        """
        if self.index is None:
            raise ValueError("No index available to save.")
        
        # 创建目录
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, index_path)
        
        print(f"[DEBUG] Index saved to {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """
        加载FAISS索引
        
        Args:
            index_path: 索引路径
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # 加载索引
        self.index = faiss.read_index(index_path)
        
        print(f"[DEBUG] Index loaded from {index_path}")


class RAGSearchTool(Tool):
    """
    基于RAG的编译器自动调优搜索工具
    
    基于预先构建的FAISS索引，
    为给定的Autophase嵌入向量查找最相似的Pass序列。
    """
    
    def __init__(self):
        """
        初始化编译器自动调优RAG搜索工具
        """
        name = "rag_search"
        description = "搜索最佳的编译器优化Pass序列，根据LLVM IR的Autophase特征进行相似度匹配。"
        parameters = {
            "type": "object",
            "properties": {
                "autophase_embedding": {
                    "type": ["array", "object"],
                    "description": "Autophase嵌入向量，可以是列表形式或字典形式"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量（默认为3）"
                }
            },
            "required": ["autophase_embedding"]
        }
        
        super().__init__(name, description, parameters)
        
        # 设置默认路径
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "../../../../examples/data_preprocess/compiler_autotuning_data.csv")
        index_path = os.path.join(current_dir, "./compiler_autotuning_index.bin")
        
        # 确保CSV文件存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"[DEBUG] Initializing RAGSearchTool with csv_path={csv_path}, index_path={index_path}")
        
        # 初始化搜索引擎
        self.search_engine = CompilerAutotuningSimilaritySearch(csv_path)
        
        # 加载数据集
        self.search_engine.load_dataset()
        
        try:
            # 尝试加载已存在的索引
            self.search_engine.load_index(index_path)
        except FileNotFoundError:
            # 如果索引不存在，则构建并保存
            print(f"[DEBUG] Index not found at {index_path}, building new index...")
            self.search_engine.build_index()
            self.search_engine.save_index(index_path)
        
        print("[DEBUG] RAGSearchTool initialized successfully")
    
    def execute(self, args: Dict) -> str:
        """
        执行搜索工具
        
        Args:
            args: 工具参数，包含:
                - "autophase_embedding": Autophase嵌入向量（列表或字典形式）
                - "top_k": 返回的结果数量
                
        Returns:
            格式化的搜索结果
        """
        # 获取参数
        autophase_embedding = args.get("autophase_embedding")
        top_k = args.get("top_k", 3)
        
        try:
            # 处理 autophase_embedding
            if isinstance(autophase_embedding, str):
                # 尝试解析字符串形式的嵌入向量
                try:
                    # 先尝试解析为字典
                    autophase_embedding = json.loads(autophase_embedding)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试以Python字面量解析
                    autophase_embedding = ast.literal_eval(autophase_embedding)
            
            # 如果 autophase_embedding 是None或无效类型，返回错误
            if autophase_embedding is None:
                raise ValueError("autophase_embedding is missing or invalid")
                
            # 现在 autophase_embedding 可能是字典或列表，search方法已经能处理两种情况
            # 执行搜索
            results = self.search_engine.search(autophase_embedding, top_k)
            
            # 格式化结果
            formatted_results = {
                "results": results
            }
            
            return json.dumps(formatted_results, indent=2)
        except Exception as e:
            error_msg = f"搜索过程中发生错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return json.dumps({"error": error_msg})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        批量执行搜索
        
        Args:
            args_list: 工具参数列表
            
        Returns:
            搜索结果列表
        """
        return [self.execute(args) for args in args_list]
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        计算工具调用的奖励值
        
        Args:
            args: 工具参数
            result: 工具执行结果
            
        Returns:
            奖励值
        """
        try:
            result_obj = json.loads(result)
            if "error" in result_obj:
                return -0.1  # 错误结果给予轻微惩罚
            
            # 检查结果中是否有有效的Pass序列
            if "results" in result_obj and len(result_obj["results"]) > 0:
                # 返回基于相似度的奖励
                similarity = result_obj["results"][0].get("similarity", 0)
                # 相似度在0到1之间，我们将其转换为0.0到0.1的奖励值
                return min(max(similarity, 0), 1) * 0.1
            
            return 0.0  # 默认无奖励
        except:
            return -0.1  # 解析结果出错 