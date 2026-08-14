"""ReAct Agent — Reasoning + Acting loop over a tool registry."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from src.agent.agent_state import AgentState, Turn, ToolCall
from src.agent.tool_registry import ToolRegistry, ToolDispatchError
from src.libs.llm.base_llm import Message
from src.core.settings import resolve_path
from src.production.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=\nAction|\nFinal Answer|$)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*:\s*(\S+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input\s*:\s*(\{.*?\})", re.IGNORECASE | re.DOTALL)


@dataclass
class AgentStreamEvent:
    """Incremental event emitted by ReActAgent.run_stream()."""
    type: str  # "thought" | "action" | "observation" | "done"
    turn_idx: int = 0
    thought: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    # Populated only in "done" event:
    answer: str = ""
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    citations: List[str] = field(default_factory=list)
    used_tools: List[str] = field(default_factory=list)
    turns_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Final output from the ReAct agent."""
    answer: str
    citations: List[str]
    turns: List[Turn]
    confidence: float
    used_tools: List[str]
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "turns": [t.to_dict() for t in self.turns],
            "confidence": self.confidence,
            "used_tools": self.used_tools,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class ReActAgent:
    """ReAct-style agent that alternates between Thought, Action, and Observation.

    The agent calls an LLM to produce Thought/Action pairs, dispatches the
    requested tool via ToolRegistry, feeds the observation back, and repeats
    until the LLM emits a "Final Answer" line or the turn limit is reached.

    Args:
        settings: Application settings (reads agent.max_turns, agent.confidence_threshold).
        llm: LLM instance used for reasoning.
        tool_registry: ToolRegistry with all available tools pre-registered.
        max_turns: Override for max ReAct iterations (falls back to settings).
        confidence_threshold: Override for minimum confidence (falls back to settings).
    """

    def __init__(
        self,
        settings: Optional[Settings],
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        max_turns: int = 5,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry

        # Prefer explicit args, then settings, then hard defaults
        agent_cfg = getattr(settings, "agent", None) if settings else None
        self.max_turns = max_turns if max_turns != 5 else (
            getattr(agent_cfg, "max_turns", 5) if agent_cfg else max_turns
        )
        self.confidence_threshold = confidence_threshold if confidence_threshold != 0.7 else (
            getattr(agent_cfg, "confidence_threshold", 0.7) if agent_cfg else confidence_threshold
        )

        self._prompt_template = self._load_prompt()

        from src.agent.reflection.self_checker import SelfChecker
        self._self_checker = SelfChecker(
            llm=self.llm,
            confidence_threshold=self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> AgentResponse:
        """Execute the ReAct loop for a question.

        Args:
            question: The user question.

        Returns:
            AgentResponse with the final answer, citations, and trace data.
        """
        t0 = time.monotonic()
        state = AgentState(question=question)
        used_tools: List[str] = []

        for turn_idx in range(self.max_turns):
            logger.debug("ReAct turn %d / %d", turn_idx + 1, self.max_turns)

            llm_output = self._call_llm(state)
            thought, action, is_final = self._parse_llm_output(llm_output)

            if is_final or action is None:
                # LLM decided it has enough info
                final_text = self._extract_final_answer(llm_output) or thought
                state.final_answer = final_text
                state.add_turn(Turn(thought=thought, action=None, observation=None))
                break

            # Dispatch tool
            observation = self._dispatch_tool(action)
            state.add_turn(Turn(thought=thought, action=action, observation=observation))
            used_tools.append(action.tool_name)

        else:
            # Hit max_turns without Final Answer — use last thought as answer
            state.final_answer = (
                state.history[-1].thought if state.history else "No answer found."
            )

        citations = self._collect_citations(state)
        try:
            check = self._self_checker.check(
                question=question,
                answer=state.final_answer or "",
                context=state.retrieval_context,
            )
            confidence = check.confidence
        except Exception:
            confidence = self._estimate_confidence(state)
        elapsed = (time.monotonic() - t0) * 1000.0

        logger.info(
            "ReAct finished: turns=%d, tools=%s, confidence=%.2f, elapsed_ms=%.0f",
            len(state.history), used_tools, confidence, elapsed,
        )

        return AgentResponse(
            answer=state.final_answer or "",
            citations=citations,
            turns=state.history,
            confidence=confidence,
            used_tools=list(dict.fromkeys(used_tools)),  # deduplicated, order preserved
            elapsed_ms=elapsed,
        )

    def run_stream(self, question: str) -> Generator[AgentStreamEvent, None, None]:
        """Like run() but yields AgentStreamEvent for each ReAct step.

        Yields thought → (action → observation)* → done events so callers can
        update a streaming UI incrementally without waiting for the full response.
        """
        t0 = time.monotonic()
        state = AgentState(question=question)
        used_tools: List[str] = []

        for turn_idx in range(self.max_turns):
            logger.debug("ReAct stream turn %d / %d", turn_idx + 1, self.max_turns)

            llm_output = self._call_llm(state)
            thought, action, is_final = self._parse_llm_output(llm_output)

            yield AgentStreamEvent(type="thought", turn_idx=turn_idx + 1, thought=thought)

            if is_final or action is None:
                final_text = self._extract_final_answer(llm_output) or thought
                state.final_answer = final_text
                state.add_turn(Turn(thought=thought, action=None, observation=None))
                break

            yield AgentStreamEvent(
                type="action",
                turn_idx=turn_idx + 1,
                tool_name=action.tool_name,
                tool_input=action.tool_input,
            )

            observation = self._dispatch_tool(action)
            state.add_turn(Turn(thought=thought, action=action, observation=observation))
            used_tools.append(action.tool_name)

            yield AgentStreamEvent(
                type="observation",
                turn_idx=turn_idx + 1,
                observation=observation,
            )
        else:
            state.final_answer = (
                state.history[-1].thought if state.history else "No answer found."
            )

        citations = self._collect_citations(state)
        try:
            check = self._self_checker.check(
                question=question,
                answer=state.final_answer or "",
                context=state.retrieval_context,
            )
            confidence = check.confidence
        except Exception:
            confidence = self._estimate_confidence(state)

        elapsed = (time.monotonic() - t0) * 1000.0
        logger.info(
            "ReAct stream finished: turns=%d, tools=%s, confidence=%.2f, elapsed_ms=%.0f",
            len(state.history), used_tools, confidence, elapsed,
        )

        yield AgentStreamEvent(
            type="done",
            answer=state.final_answer or "",
            confidence=confidence,
            elapsed_ms=elapsed,
            citations=citations,
            used_tools=list(dict.fromkeys(used_tools)),
            turns_data=[t.to_dict() for t in state.history],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_prompt(self) -> str:
        return PromptRegistry(resolve_path("config/prompts")).get("react_agent").template

    def _build_prompt(self, state: AgentState) -> str:
        return self._prompt_template.format(
            tools=self.tool_registry.get_tools_prompt(),
            question=state.question,
            memory=state.format_history(),
        )

    def _call_llm(self, state: AgentState) -> str:
        prompt = self._build_prompt(state)
        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages)
        return response.content

    def _parse_llm_output(self, text: str) -> tuple[str, Optional[ToolCall], bool]:
        """Extract (thought, tool_call, is_final_answer) from LLM output."""
        is_final = bool(_FINAL_ANSWER_RE.search(text))

        thought_match = _THOUGHT_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else text.strip()

        if is_final:
            return thought, None, True

        action_match = _ACTION_RE.search(text)
        input_match = _ACTION_INPUT_RE.search(text)

        if not action_match:
            # No action found — treat as final
            return thought, None, True

        tool_name = action_match.group(1).strip()
        tool_input: Dict[str, Any] = {}
        if input_match:
            try:
                tool_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Could not parse Action Input JSON: %s", input_match.group(1))

        return thought, ToolCall(tool_name=tool_name, tool_input=tool_input), False

    def _extract_final_answer(self, text: str) -> Optional[str]:
        m = _FINAL_ANSWER_RE.search(text)
        return m.group(1).strip() if m else None

    def _dispatch_tool(self, action: ToolCall) -> str:
        try:
            return self.tool_registry.dispatch(action.tool_name, action.tool_input)
        except ToolDispatchError as exc:
            logger.warning("Tool dispatch error: %s", exc)
            return f"[Tool error: {exc}]"

    def _collect_citations(self, state: AgentState) -> List[str]:
        """Extract unique source paths from retrieved context."""
        seen: set[str] = set()
        citations: List[str] = []
        for result in state.retrieval_context:
            src = result.metadata.get("source_path", "")
            if src and src not in seen:
                seen.add(src)
                citations.append(src)
        return citations

    def _estimate_confidence(self, state: AgentState) -> float:
        """Heuristic confidence: ratio of successful tool calls to total turns."""
        if not state.history:
            return 0.0
        successful = sum(
            1 for t in state.history
            if t.observation and not t.observation.startswith("[Tool error")
        )
        base = successful / len(state.history) if state.history else 0.0
        # Boost if we ended with Final Answer (not just hitting max_turns)
        if state.final_answer and state.history and state.history[-1].action is None:
            base = min(1.0, base + 0.2)
        return round(base, 3)
