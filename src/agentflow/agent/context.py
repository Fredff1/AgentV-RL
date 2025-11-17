# src/agent/context.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from agentflow.common.messages import Message
from agentflow.tools.base import ToolCallResult  

RoundRole = Literal["user", "system", "assistant", "tool"]


@dataclass
class RoundBuffer:
    messages: List[Message] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        parts = []
        for m in self.messages:
            name = f" ({m.name})" if getattr(m, "name", None) else ""
            parts.append(f"[{m.role}{name}] {m.content}")
        return "\n".join(parts)


@dataclass
class AgentContext:
    """Message based context for multi-turn generation
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = 0

    prompt_messages: List[Message] = field(default_factory=list)  
    rounds: List[RoundBuffer] = field(default_factory=list)
    global_round: int = 0

    round_counters: Dict[str, int] = field(default_factory=dict)
    tool_results: List[ToolCallResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def set_round(self, round_index: int) -> None:
        self.global_round = int(round_index)
        self._ensure_round_exists(self.global_round)

    def begin_new_round(self) -> int:
        self.global_round += 1
        self._ensure_round_exists(self.global_round)
        return self.global_round

    def _ensure_round_exists(self, idx: int) -> None:
        while len(self.rounds) <= idx:
            self.rounds.append(RoundBuffer())

    def append(
        self,
        content: str,
        *,
        role: RoundRole = "assistant",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a message for current round"""
        self._ensure_round_exists(self.global_round)
        msg = Message(role=role, content=content, name=name, metadata=metadata or {})
        self.rounds[self.global_round].messages.append(msg)
        return msg

    def append_user(self, text: str) -> Message:
        return self.append(text, role="user")

    def append_system(self, text: str) -> Message:
        return self.append(text, role="system")

    def append_assistant(self, text: str) -> Message:
        return self.append(text, role="assistant")

    def append_tool_observation(self, text: str, *, name: str = "observations", metadata: Optional[Dict[str, Any]] = None) -> Message:
        return self.append(text, role="tool", name=name, metadata=metadata)

    def append_messages(self, msgs: List[Message]) -> None:
        """append multiple messages to current round"""
        self._ensure_round_exists(self.global_round)
        self.rounds[self.global_round].messages.extend(msgs)

    def pop(self) -> Optional[Message]:
        """Pop the last non-empty message"""
        for r in range(len(self.rounds) - 1, -1, -1):
            if self.rounds[r].messages:
                return self.rounds[r].messages.pop()
        return None

    def current_round_buffer(self) -> RoundBuffer:
        self._ensure_round_exists(self.global_round)
        return self.rounds[self.global_round]

    def current_round_messages(self) -> List[Message]:
        return list(self.current_round_buffer().messages)

    def round_messages(self, round_index: int) -> List[Message]:
        self._ensure_round_exists(round_index)
        return list(self.rounds[round_index].messages)
    
    def all_round_messages(self) -> List[Message]:
        """Get all round messages"""
        msg_list = []
        for round in self.rounds:
            msg_list.extend(round.messages)
        return msg_list

    def all_messages(self) -> List[Message]:
        """Get all meesages (Include prompt and all round messages)"""
        out: List[Message] = list(self.prompt_messages)
        for rb in self.rounds:
            out.extend(rb.messages)
        return out

    def last_message(self) -> Optional[Message]:
        for r in range(len(self.rounds) - 1, -1, -1):
            msgs = self.rounds[r].messages
            if msgs:
                return msgs[-1]
        return None

    def update_round_counter(self, tool_result: ToolCallResult) -> None:
        tool_name = getattr(tool_result, "tool_name", None) or tool_result.get("tool_name", "tool")
        new_counter = self.round_counters.get(tool_name, 0) + 1
        self.round_counters[tool_name] = new_counter
        self.meta["round_counter"][tool_name] = new_counter
        self.tool_results.append(tool_result)

    @classmethod
    def from_messages(cls, messages: List[Message], meta: Optional[Dict[str, Any]] = None) -> AgentContext:
        ctx = cls(prompt_messages=list(messages), meta=meta or {})
        ctx._ensure_round_exists(0)
        if isinstance(meta, dict):
            round_counter_meta = meta.get("round_counter", {})
            if isinstance(round_counter_meta, dict):
                ctx.round_counters = dict(round_counter_meta)
            
        return ctx
