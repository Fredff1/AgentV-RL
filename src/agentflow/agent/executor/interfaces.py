# agentflow/agent/executor/interfaces.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from agentflow.agent.planner.interfaces import Plan
from agentflow.tools.base import ToolCallResult


@dataclass
class SubtaskReport:
    subtask_id: str
    raw_trace: str                   # 聚合原始轨迹（含 tags）
    tool_traces: List[ToolCallResult] = field(default_factory=list)
    rounds_used: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        tool_traces = []
        raw = data.get("tool_traces",[])
        for trace in raw:
            tool_traces.append(ToolCallResult.from_dict(trace))
        return SubtaskReport(
            subtask_id=data.get("subtask_id"),
            raw_trace=data.get("raw_trace"),
            tool_traces=tool_traces,
            rounds_used=data.get("rounds_used",0),
            notes=data.get("notes",{}),
        )

@dataclass
class VerificationSubtaskReport(SubtaskReport):
    verdict: Optional[bool] = None          # True/False/None
    verify_text: str = ""                # <verify> 内容
    
    @classmethod
    def from_dict(cls, data: Dict):
        tool_traces = []
        raw = data.get("tool_traces",[])
        for trace in raw:
            tool_traces.append(ToolCallResult.from_dict(trace))
        return VerificationSubtaskReport(
            subtask_id=data.get("subtask_id"),
            raw_trace=data.get("raw_trace",""),
            tool_traces=tool_traces,
            rounds_used=data.get("rounds_used",0),
            notes=data.get("notes",{}),
            verdict=data.get("verdict"),
            verify_text=data.get("verify_text",""),
            
        )


@dataclass
class ExecutionReport:
    sequence_id: str
    subtask_reports: List[SubtaskReport]
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict):
        reports = data.get("subtask_reports",[])
        sub_reps = []
        for report in reports:
            sub_reps.append(VerificationSubtaskReport.from_dict(report))
        return ExecutionReport(
            sequence_id=data.get("sequence_id"),
            subtask_reports=sub_reps,
            meta=data.get("meta",{}),
        )
    
class SubtaskExecutor:
    def execute(self, *, sequences: List[str], plans: List[Plan]) -> List[ExecutionReport]:
        ...