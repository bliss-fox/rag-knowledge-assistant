"""Agent state data structures for the ReAct loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


@dataclass
class ToolCall:
    """A single tool invocation requested by the agent."""
    tool_name: str
    tool_input: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_name": self.tool_name, "tool_input": self.tool_input}


@dataclass
class Turn:
    """One Thought→Action→Observation cycle in the ReAct loop."""
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action.to_dict() if self.action else None,
            "observation": self.observation,
        }


@dataclass
class AgentState:
    """Mutable state carried through the full ReAct loop for one question."""
    question: str
    history: List[Turn] = field(default_factory=list)
    final_answer: Optional[str] = None
    retrieval_context: List[RetrievalResult] = field(default_factory=list)

    def add_turn(self, turn: Turn) -> None:
        self.history.append(turn)

    def format_history(self) -> str:
        """Render previous turns as plain text for the prompt."""
        if not self.history:
            return ""
        lines: List[str] = []
        for t in self.history:
            lines.append(f"Thought: {t.thought}")
            if t.action:
                lines.append(f"Action: {t.action.tool_name}")
                lines.append(f"Action Input: {t.action.tool_input}")
            if t.observation is not None:
                lines.append(f"Observation: {t.observation}")
        return "\n".join(lines)
