# 编译器自动调优RAG搜索工具

这个工具实现了基于检索增强生成(RAG)的编译器自动调优功能，可以根据LLVM IR的Autophase特征，找到最相似的优化Pass序列。

## 功能概述

RAG搜索工具提供以下功能：

1. 从compiler_autotuning_data.csv数据集中加载编译器优化数据
2. 构建基于FAISS的高效向量索引
3. 根据输入的Autophase特征向量，找到最相似的优化Pass序列
4. 支持批量查询处理

## 安装依赖

工具依赖以下Python库：

```bash
pip install pandas numpy faiss-cpu
```

## 使用方法

### 在Agent-R1环境中使用

在Agent-R1环境中，RAGSearchTool会自动加载并注册。可以通过以下方式调用：

```python
from agent_r1.tool.tool_env import ToolEnv
from agent_r1.tool.tools.comiler_autotuning import RAGSearchTool

# 创建工具环境
tools = [RAGSearchTool()]  # 可以与其他工具一起使用
tool_env = ToolEnv(tools=tools)

# 准备工具调用
autophase_embedding = [1,0,3,2,1,2,0,0,0,2,1,1,0,4,1,4,0,1,6,5,0,3,2,2,1,0,0,0,0,1,4,6,4,3,0,0,0,2,0,1,2,1,0,3,0,6,0,0,0,0,5,48,11,3,6,8]
action_text = f"""<tool_call>{{"name": "rag_search", "arguments": {{"autophase_embedding": {autophase_embedding}, "top_k": 3}}}}</tool_call>"""

# 执行工具调用
observation, reward, done, info = tool_env.step(action_text)
print(observation)
```

### 在LLM指令中使用

当与LLM结合使用时，可以使用以下指令格式：

```
你可以使用RAG搜索工具来查找最佳的编译器优化Pass序列。

工具名: rag_search
参数:
  - autophase_embedding: LLVM IR的Autophase特征向量
  - top_k: 返回的结果数量(可选，默认为3)

例如，对于给定的Autophase向量 [1,0,3,2,...,8]，调用工具查找相似的Pass序列。
```

## 工具参数说明

RAGSearchTool接受以下参数：

- `autophase_embedding` (必填): LLVM IR的Autophase特征向量，应当是一个长度为56的数值列表
- `top_k` (可选): 返回的结果数量，默认为3

## 返回结果格式

工具返回JSON格式的结果，包含以下字段：

```json
{
  "results": [
    {
      "similarity": 0.9875,
      "pass_sequence": ["--pass1", "--pass2", "..."],
      "index": 42
    },
    ...
  ]
}
```

其中：
- `similarity`: 相似度得分，越高表示匹配度越好
- `pass_sequence`: 匹配到的优化Pass序列
- `index`: 在原始数据集中的索引位置

## 与其他编译器工具的结合使用

RAG搜索工具可以与其他编译器工具结合使用，例如：

1. 先使用`GenAutophaseTool`获取LLVM IR代码的Autophase特征
2. 然后使用`RAGSearchTool`根据Autophase特征查找相似的Pass序列
3. 最后使用`OptimizeLLCodeTool`应用找到的Pass序列来优化LLVM IR代码

示例流程：

```python
# 1. 获取Autophase特征
autophase_tool = GenAutophaseTool()
autophase_result = autophase_tool.execute({"ll_code": llvm_ir_code})
autophase_data = json.loads(autophase_result)
autophase_vector = autophase_data["features"]

# 2. 使用RAG搜索查找Pass序列
rag_tool = RAGSearchTool()
rag_result = rag_tool.execute({"autophase_embedding": autophase_vector})
rag_data = json.loads(rag_result)
best_pass_sequence = rag_data["results"][0]["pass_sequence"]

# 3. 应用Pass序列优化代码
optimize_tool = OptimizeLLCodeTool()
optimize_result = optimize_tool.execute({
    "ll_code": llvm_ir_code,
    "passes": best_pass_sequence
})
```

## 测试工具

可以使用提供的测试脚本来测试RAG搜索工具的功能：

```bash
python agent_r1/tool/tools/comiler_autotuning/test_rag_search_tool.py
```

指定测试类型：

```bash
# 只测试单个查询
python test_rag_search_tool.py --test_type single

# 只测试批量查询
python test_rag_search_tool.py --test_type batch

# 测试所有功能
python test_rag_search_tool.py --test_type all
``` 