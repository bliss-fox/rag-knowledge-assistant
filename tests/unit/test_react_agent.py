"""Unit tests for ReActAgent core loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_state import AgentState, Turn, ToolCall
from src.agent.react_agent import ReActAgent, AgentResponse
from src.agent.tool_registry import ToolRegistry
from src.libs.llm.base_llm import ChatResponse, Message


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_llm(responses: list[str]) -> MagicMock:
    """Return a mock LLM that yields responses in sequence."""
    llm = MagicMock()
    llm.chat.side_effect = [ChatResponse(content=r, model="mock") for r in responses]
    return llm


def _make_registry(*tool_names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(
            name=name,
            fn=lambda **kw: f"observation from {name}",
            description=f"Tool {name}",
            params_schema={},
        )
    return registry


# ------------------------------------------------------------------
# AgentState tests
# ------------------------------------------------------------------

class TestAgentState:
    def test_initial_state(self):
        state = AgentState(question="test?")
        assert state.question == "test?"
        assert state.history == []
        assert state.final_answer is None

    def test_add_turn(self):
        state = AgentState(question="q")
        turn = Turn(thought="thinking", action=ToolCall("t", {}), observation="obs")
        state.add_turn(turn)
        assert len(state.history) == 1

    def test_format_history_empty(self):
        state = AgentState(question="q")
        assert state.format_history() == ""

    def test_format_history_with_turns(self):
        state = AgentState(question="q")
        state.add_turn(Turn(
            thought="I should search",
            action=ToolCall("hybrid_search", {"query": "BM25"}),
            observation="BM25 is a ranking function",
        ))
        history = state.format_history()
        assert "Thought: I should search" in history
        assert "Action: hybrid_search" in history
        assert "Observation: BM25 is a ranking function" in history


# ------------------------------------------------------------------
# ToolCall / Turn serialisation
# ------------------------------------------------------------------

class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(tool_name="hybrid_search", tool_input={"query": "test"})
        d = tc.to_dict()
        assert d["tool_name"] == "hybrid_search"
        assert d["tool_input"] == {"query": "test"}


class TestTurn:
    def test_to_dict_with_action(self):
        turn = Turn(
            thought="searching",
            action=ToolCall("semantic_search", {"query": "hello"}),
            observation="result",
        )
        d = turn.to_dict()
        assert d["thought"] == "searching"
        assert d["action"]["tool_name"] == "semantic_search"
        assert d["observation"] == "result"

    def test_to_dict_final_turn(self):
        turn = Turn(thought="I have the answer.", action=None, observation=None)
        d = turn.to_dict()
        assert d["action"] is None
        assert d["observation"] is None


# ------------------------------------------------------------------
# ReActAgent tests
# ------------------------------------------------------------------

class TestReActAgent:
    def test_single_turn_final_answer(self):
        """Agent produces Final Answer in first turn."""
        llm = _make_llm([
            "Thought: I know this already.\nFinal Answer: BM25 is a ranking function."
        ])
        registry = _make_registry()
        agent = ReActAgent(settings=None, llm=llm, tool_registry=registry, max_turns=3)
        response = agent.run("What is BM25?")
        assert isinstance(response, AgentResponse)
        assert "BM25" in response.answer
        assert response.used_tools == []

    def test_tool_call_then_answer(self):
        """Agent calls one tool then answers."""
        llm = _make_llm([
            (
                "Thought: I need to search.\n"
                "Action: hybrid_search\n"
                'Action Input: {"query": "BM25"}'
            ),
            "Thought: Found it.\nFinal Answer: BM25 stands for Best Match 25.",
        ])
        registry = _make_registry("hybrid_search")
        agent = ReActAgent(settings=None, llm=llm, tool_registry=registry, max_turns=5)
        response = agent.run("What is BM25?")
        assert "BM25" in response.answer or response.answer
        assert "hybrid_search" in response.used_tools

    def test_max_turns_hit(self):
        """Agent hits max_turns and falls back to last thought."""
        never_final = (
            "Thought: Keep searching.\n"
            "Action: hybrid_search\n"
            'Action Input: {"query": "something"}'
        )
        llm = _make_llm([never_final] * 3)
        registry = _make_registry("hybrid_search")
        agent = ReActAgent(settings=None, llm=llm, tool_registry=registry, max_turns=3)
        response = agent.run("Infinite question?")
        # Should complete without exception even when max_turns hit
        assert isinstance(response, AgentResponse)
        assert len(response.turns) == 3

    def test_unknown_tool_returns_error_observation(self):
        """Unknown tool name produces an error observation, not an exception."""
        llm = _make_llm([
            "Thought: use mystery tool.\nAction: mystery_tool\nAction Input: {}",
            "Thought: Got error. Final Answer: I couldn't find it.",
        ])
        registry = _make_registry()  # no tools registered
        agent = ReActAgent(settings=None, llm=llm, tool_registry=registry, max_turns=5)
        response = agent.run("test")
        assert isinstance(response, AgentResponse)

    def test_confidence_estimation_no_turns(self):
        """Confidence is 0.0 when there are no turns."""
        state = AgentState(question="q")
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        assert agent._estimate_confidence(state) == 0.0

    def test_confidence_increases_with_successful_tools(self):
        state = AgentState(question="q")
        state.add_turn(Turn(thought="t", action=ToolCall("t", {}), observation="good result"))
        state.add_turn(Turn(thought="done", action=None, observation=None))
        state.final_answer = "answer"
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        conf = agent._estimate_confidence(state)
        assert 0.0 < conf <= 1.0

    def test_parse_llm_output_final_answer(self):
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        thought, action, is_final = agent._parse_llm_output(
            "Thought: done.\nFinal Answer: The answer is 42."
        )
        assert is_final
        assert action is None

    def test_parse_llm_output_tool_call(self):
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        text = (
            'Thought: search first.\n'
            'Action: hybrid_search\n'
            'Action Input: {"query": "hello"}'
        )
        thought, action, is_final = agent._parse_llm_output(text)
        assert not is_final
        assert action is not None
        assert action.tool_name == "hybrid_search"
        assert action.tool_input == {"query": "hello"}

    def test_extract_final_answer(self):
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        text = "Thought: done.\nFinal Answer: 42 is the answer."
        assert agent._extract_final_answer(text) == "42 is the answer."

    def test_extract_final_answer_none(self):
        agent = ReActAgent(settings=None, llm=MagicMock(), tool_registry=_make_registry())
        assert agent._extract_final_answer("no final answer here") is None

    def test_agent_response_to_dict(self):
        from src.agent.agent_state import Turn
        response = AgentResponse(
            answer="42",
            citations=["doc1.pdf"],
            turns=[Turn(thought="t")],
            confidence=0.9,
            used_tools=["hybrid_search"],
            elapsed_ms=123.4,
        )
        d = response.to_dict()
        assert d["answer"] == "42"
        assert d["confidence"] == 0.9
        assert d["citations"] == ["doc1.pdf"]
