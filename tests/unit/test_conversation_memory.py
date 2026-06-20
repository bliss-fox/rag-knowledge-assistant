"""Unit tests for ConversationMemory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.memory.conversation_memory import ConversationMemory
from src.libs.llm.base_llm import ChatResponse


def _make_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = ChatResponse(content=response_text, model="mock")
    return llm


class TestConversationMemory:
    def test_initial_state(self):
        mem = ConversationMemory()
        assert len(mem) == 0
        assert mem.get_context_messages() == []
        assert mem.format_as_text() == ""

    def test_add_turn_user(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Hello")
        assert len(mem) == 1
        msgs = mem.get_context_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_add_turn_assistant(self):
        mem = ConversationMemory()
        mem.add_turn("assistant", "Hi there")
        msgs = mem.get_context_messages()
        assert msgs[0]["role"] == "assistant"

    def test_invalid_role_raises(self):
        mem = ConversationMemory()
        with pytest.raises(ValueError, match="Invalid role"):
            mem.add_turn("system", "oops")

    def test_get_recent_queries(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Q1")
        mem.add_turn("assistant", "A1")
        mem.add_turn("user", "Q2")
        mem.add_turn("assistant", "A2")
        recent = mem.get_recent_queries(n=2)
        assert recent == ["Q1", "Q2"]

    def test_get_recent_queries_empty(self):
        mem = ConversationMemory()
        assert mem.get_recent_queries() == []

    def test_format_as_text(self):
        mem = ConversationMemory()
        mem.add_turn("user", "What is BM25?")
        mem.add_turn("assistant", "BM25 is a ranking function.")
        text = mem.format_as_text()
        assert "User: What is BM25?" in text
        assert "Assistant: BM25 is a ranking function." in text

    def test_format_as_text_with_summary(self):
        mem = ConversationMemory()
        mem._summary = "Previous discussion about BM25."
        mem.add_turn("user", "follow-up")
        text = mem.format_as_text()
        assert "Previous discussion" in text
        assert "follow-up" in text

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Q")
        mem.add_turn("assistant", "A")
        mem._summary = "some summary"
        mem.clear()
        assert len(mem) == 0
        assert mem._summary is None

    def test_rewrite_query_no_history(self):
        mem = ConversationMemory()
        llm = _make_llm("rewritten")
        result = mem.rewrite_query("What is it?", llm)
        # No history → return original unchanged
        assert result == "What is it?"
        llm.chat.assert_not_called()

    def test_rewrite_query_with_history(self):
        mem = ConversationMemory()
        mem.add_turn("user", "What is RRF?")
        mem.add_turn("assistant", "RRF is Reciprocal Rank Fusion.")
        llm = _make_llm("What is Reciprocal Rank Fusion used for?")
        result = mem.rewrite_query("What is it used for?", llm)
        assert result == "What is Reciprocal Rank Fusion used for?"

    def test_rewrite_query_llm_failure_returns_original(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Q1")
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API down")
        result = mem.rewrite_query("follow up", llm)
        assert result == "follow up"

    def test_summarize_if_long_not_triggered(self):
        mem = ConversationMemory(max_turns=10)
        for i in range(3):
            mem.add_turn("user", f"Q{i}")
            mem.add_turn("assistant", f"A{i}")
        llm = _make_llm("summary")
        mem.summarize_if_long(llm)
        # Not triggered — 6 turns < max_turns=10
        llm.chat.assert_not_called()

    def test_summarize_if_long_triggered(self):
        mem = ConversationMemory(max_turns=4)
        for i in range(6):
            mem.add_turn("user", f"Q{i}")
        llm = _make_llm("Summarised conversation.")
        mem.summarize_if_long(llm)
        llm.chat.assert_called_once()
        assert mem._summary == "Summarised conversation."
        assert len(mem) == 4  # only last 4 kept

    def test_summarize_llm_failure_leaves_history_unchanged(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(5):
            mem.add_turn("user", f"Q{i}")
        original_len = len(mem)
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("fail")
        mem.summarize_if_long(llm)
        assert len(mem) == original_len  # unchanged
