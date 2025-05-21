"""
Specific tool implementations
"""

from agent_r1.tool.tools.comiler_autotuning.find_best_pass_sequence_tool import FindBestPassSequenceTool
from agent_r1.tool.tools.comiler_autotuning.instrcount_tool import InstrCountTool

__all__ = [
    'AutophaseAnalyzerTool',
    'FindBestPassSequenceTool',
] 

def _default_tools(env):
    if env == 'optimizer':
        return [FindBestPassSequenceTool(), InstrCountTool()]
    else:
        raise NotImplementedError
