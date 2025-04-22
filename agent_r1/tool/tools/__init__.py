"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.comiler_autotuning.autophase_analyzer_tool import AutophaseAnalyzerTool
from agent_r1.tool.tools.comiler_autotuning.rag_search_tool import RAGSearchTool

__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'AutophaseAnalyzerTool',
    'RAGSearchTool'
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'optimizer':
        return [AutophaseAnalyzerTool(),RAGSearchTool()]
    else:
        raise NotImplementedError
