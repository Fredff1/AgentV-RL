# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

@dataclass
class ToolCallRequest:
    """Wrapper class for a tool call request
    """
    index: int
    name: str
    content: str
    meta: Dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return ToolCallRequest(
            index = data["index"],
            name = data["name"],
            content=data["content"],
            meta=data.get("meta",{})
        )
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "name": self.name,
            "content": self.content,
            "meta": self.meta,
        }

@dataclass
class ToolCallResult:
    """Wrapper class for a tool call result
    """
    tool_name: str
    request_content: str
    output: Any
    meta: Dict 
    error: Optional[Any]
    index: int
    call: ToolCallRequest
    
    @classmethod
    def from_dict(cls, data: Dict) -> ToolCallResult:
        return ToolCallResult(
            tool_name=data["tool_name"],
            request_content=data["request_content"],
            output=data.get("output",""),
            meta=data.get("meta",{}),
            error=data.get("error",None),
            index=data.get("index",0),
            call=ToolCallRequest.from_dict(data.get("call")),
        )
        
    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "request_content": self.request_content,
            "output": self.output,
            "meta": self.meta,
            "error": self.error,
            "index": self.index,
            "call": self.call.to_dict(),
        }
    
    

class ToolParser:
    """A component to parse tool call requestsfom plain texts
    """
    def parse(self, text: str, meta: Dict = None) -> list[ToolCallRequest]: 
        """Parse requests from one sample
        """
        ...
    def parse_batch(self, texts: list[str], metas: List[Dict]=None) -> list[list[ToolCallRequest]]:
        """Parse requests from batch samples
        """
        if not metas:
            metas = [None]*len(texts)
        return [self.parse(t, meta) for t, meta in zip(texts,metas)]
    
    def make_result_str(self, result: ToolCallResult) -> str:...

class BaseTool(ABC):
    """Base class for a tool"""

    name: str = "tool"

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_rounds: int = 3) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.max_rounds: int = int(max_rounds)  

    @abstractmethod
    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        """Run the tool once.Automatically check tool quota by reading from request.meta["round_counter"]"""
        raise NotImplementedError

    def run_batch(self, calls: List[ToolCallRequest], **kwargs: Any) -> List[ToolCallResult]:
        """Run tool with batch inputs.Automatically check tool quota by reading from request.meta["round_counter"]

        """
        def _runner(allowed_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
            out: List[ToolCallResult] = []
            for c in allowed_calls:
                try:
                    r = self.run_one(c, **kwargs)
                except Exception as e:
                    r = self._make_error_result(c, f"{type(e).__name__}: {e}")
                out.append(r)
            return out
        return self._apply_round_quota(calls, _runner)


    def _apply_round_quota(
        self,
        calls: List[ToolCallRequest],
        runner: Callable[[List[ToolCallRequest]], List[ToolCallResult]],
    ) -> List[ToolCallResult]:
        if self.max_rounds and self.max_rounds > 0:
            pass
        allowed: List[ToolCallRequest] = []
        allowed_positions: List[int] = []
        results: List[Optional[ToolCallResult]] = [None] * len(calls)

        for i, call in enumerate(calls):
            if self._is_quota_exceeded(call):
                results[i] = self._make_exceeded_result(call)
            else:
                allowed.append(call)
                allowed_positions.append(i)

        if allowed:
            produced = runner(allowed)
            for pos, r in zip(allowed_positions, produced):
                results[pos] = r

        final: List[ToolCallResult] = []
        for i, r in enumerate(results):
            if r is None:
                final.append(self._make_error_result(calls[i], "InternalError: missing result"))
            else:
                final.append(r)
        return final

    def _is_quota_exceeded(self, call: ToolCallRequest) -> bool:
        if self.max_rounds <= 0:
            return False
        meta = getattr(call, "meta", None) or {}
        rc = meta.get("round_counter") or {}
        used = rc.get(self.name)
        try:
            if used is None:
                return False
            return int(used) >= self.max_rounds
        except Exception:
            return False

    def _make_exceeded_result(self, call: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            request_content=getattr(call, "content", None),
            output="Tool Quota Exceed Failure",
            meta={**(getattr(call, "meta", {}) or {}), "skipped": True, "reason": f"per-sample max_rounds={self.max_rounds}"},
            error=f"skipped: per-sample limit {self.max_rounds}",
            index=getattr(call, "index", -1),
            call=call,
        )

    def _make_error_result(self, call: ToolCallRequest, error_msg: str) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            request_content=getattr(call, "content", None),
            output=None,
            meta={**(getattr(call, "meta", {}) or {}), "exception": True},
            error=error_msg,
            index=getattr(call, "index", -1),
            call=call,
        )