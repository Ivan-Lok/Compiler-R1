"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.comiler_autotuning.autophase_analyzer_tool import AutophaseAnalyzerTool
from agent_r1.tool.tools.comiler_autotuning.find_best_pass_sequence_tool import FindBestPassSequenceTool
from agent_r1.tool.tools.comiler_autotuning.instrcount_tool import InstrCountTool

__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'AutophaseAnalyzerTool',
    'FindBestPassSequenceTool',
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'optimizer':
        return [FindBestPassSequenceTool(), InstrCountTool()]
    else:
        raise NotImplementedError
