from typing import Any, Dict, List, Optional, Tuple

from .base import ToolParser, ToolCallRequest, ToolCallResult
from agentflow.utils.tag_util import find_tags

class TagToolParser(ToolParser):
    """Tool parser that parse tool requests like 
    \<tool_name>content\</tool_name>

    """
    def __init__(self, tool_tags: List[str] = ["python"]):
        super().__init__()
        self.tool_tags = tool_tags
        

    def parse(self, text: str, meta: Dict=None) -> List[ToolCallRequest]:
        if not meta:
            meta = {}
        matches = find_tags(text,self.tool_tags)
        calls: List[ToolCallRequest] = []
        for idx, match in enumerate(matches):
            call = ToolCallRequest(
                index=idx,
                name = match.tag,
                content=match.body,
                meta=meta
            )
            calls.append(call)
        return calls

    def make_result_str(self, result: ToolCallResult) -> str:
        return f"<result>{result.output}</result>"
