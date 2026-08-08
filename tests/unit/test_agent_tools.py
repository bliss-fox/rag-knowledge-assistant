"""Unit tests for agent atomic tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tool_registry import ToolRegistry, ToolDispatchError, ToolRegistrationError
from src.agent.tools.hybrid_search_tool import HybridSearchTool
from src.agent.tools.semantic_search_tool import SemanticSearchTool
from src.agent.tools.keyword_search_tool import KeywordSearchTool
from src.agent.tools.document_summary_tool import DocumentSummaryTool
from src.agent.tools.list_documents_tool import ListDocumentsTool
from src.core.types import RetrievalResult


def _mock_result(text: str = "chunk text", score: float = 0.9, source: str = "doc.pdf") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c1",
        score=score,
        text=text,
        metadata={"source_path": source, "title": "Doc Title"},
    )


# ------------------------------------------------------------------
# ToolRegistry
# ------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_dispatch(self):
        reg = ToolRegistry()
        reg.register("greet", lambda name="world": f"Hello {name}!", "greet tool")
        obs = reg.dispatch("greet", {"name": "Alice"})
        assert obs == "Hello Alice!"

    def test_dispatch_unknown_tool(self):
        reg = ToolRegistry()
        with pytest.raises(ToolDispatchError, match="Unknown tool"):
            reg.dispatch("nonexistent", {})

    def test_register_empty_name_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ToolRegistrationError):
            reg.register("", lambda: "x", "desc")

    def test_overwrite_warning(self, caplog):
        import logging
        reg = ToolRegistry()
        reg.register("t", lambda: "v1", "desc")
        with caplog.at_level(logging.WARNING):
            reg.register("t", lambda: "v2", "desc")
        assert "already registered" in caplog.text

    def test_tool_error_wrapped(self):
        reg = ToolRegistry()
        reg.register("boom", lambda: (_ for _ in ()).throw(ValueError("oops")), "boom")
        with pytest.raises(ToolDispatchError, match="boom"):
            reg.dispatch("boom", {})

    def test_get_tools_prompt(self):
        reg = ToolRegistry()
        reg.register("search", lambda **kw: "", "Searches stuff", {"type": "object"})
        prompt = reg.get_tools_prompt()
        assert "search" in prompt
        assert "Searches stuff" in prompt

    def test_contains(self):
        reg = ToolRegistry()
        reg.register("t", lambda: "", "d")
        assert "t" in reg
        assert "x" not in reg

    def test_list_tool_names(self):
        reg = ToolRegistry()
        reg.register("a", lambda: "", "d")
        reg.register("b", lambda: "", "d")
        names = reg.list_tool_names()
        assert set(names) == {"a", "b"}


# ------------------------------------------------------------------
# HybridSearchTool
# ------------------------------------------------------------------

class TestHybridSearchTool:
    def _make_tool(self, results):
        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = results
        context_sink = []
        tool = HybridSearchTool(
            mock_hybrid, default_top_k=5, context_sink=context_sink
        )
        return tool, context_sink, mock_hybrid

    def test_returns_results(self):
        tool, ctx, mock = self._make_tool([_mock_result("BM25 info")])
        obs = tool.run(query="BM25")
        assert "BM25 info" in obs or "BM25" in obs

    def test_populates_context_sink(self):
        r = _mock_result("content")
        tool, ctx, _ = self._make_tool([r])
        tool.run(query="test")
        assert r in ctx

    def test_empty_results(self):
        tool, ctx, _ = self._make_tool([])
        obs = tool.run(query="nothing")
        assert "No results" in obs

    def test_error_returns_error_string(self):
        mock_hybrid = MagicMock()
        mock_hybrid.search.side_effect = RuntimeError("DB down")
        tool = HybridSearchTool(mock_hybrid)
        obs = tool.run(query="q")
        assert "hybrid_search failed" in obs


# ------------------------------------------------------------------
# SemanticSearchTool
# ------------------------------------------------------------------

class TestSemanticSearchTool:
    def test_returns_results(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_mock_result("semantic content")]
        tool = SemanticSearchTool(mock_retriever, default_top_k=5)
        obs = tool.run(query="test")
        assert "semantic content" in obs

    def test_no_results(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        tool = SemanticSearchTool(mock_retriever)
        obs = tool.run(query="nothing")
        assert "No relevant results" in obs

    def test_error_handled(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("fail")
        tool = SemanticSearchTool(mock_retriever)
        obs = tool.run(query="q")
        assert "semantic_search failed" in obs


# ------------------------------------------------------------------
# KeywordSearchTool
# ------------------------------------------------------------------

class TestKeywordSearchTool:
    def test_returns_results(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_mock_result("keyword hit")]
        tool = KeywordSearchTool(mock_retriever, default_top_k=5)
        obs = tool.run(keywords=["BM25"])
        assert "keyword hit" in obs

    def test_empty_keywords(self):
        tool = KeywordSearchTool(MagicMock())
        obs = tool.run(keywords=[])
        assert "empty" in obs.lower()

    def test_no_results(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        tool = KeywordSearchTool(mock_retriever)
        obs = tool.run(keywords=["xyz"])
        assert "No results" in obs


# ------------------------------------------------------------------
# DocumentSummaryTool
# ------------------------------------------------------------------

class TestDocumentSummaryTool:
    def test_returns_summary(self):
        mock_store = MagicMock()
        mock_store.get_by_metadata.return_value = [
            {"text": "intro content", "metadata": {"source_path": "doc.pdf", "title": "My Doc", "tags": ["AI"]}}
        ]
        tool = DocumentSummaryTool(mock_store)
        obs = tool.run(source_path="doc.pdf")
        assert "doc.pdf" in obs
        assert "My Doc" in obs

    def test_no_document_found(self):
        mock_store = MagicMock()
        mock_store.get_by_metadata.return_value = []
        tool = DocumentSummaryTool(mock_store)
        obs = tool.run(source_path="missing.pdf")
        assert "No document found" in obs

    def test_error_handled(self):
        mock_store = MagicMock()
        mock_store.get_by_metadata.side_effect = Exception("fail")
        tool = DocumentSummaryTool(mock_store)
        obs = tool.run(source_path="any.pdf")
        assert "get_document_summary failed" in obs


# ------------------------------------------------------------------
# ListDocumentsTool
# ------------------------------------------------------------------

class TestListDocumentsTool:
    def test_lists_documents(self):
        mock_store = MagicMock()
        mock_store.get_all.return_value = [
            {"metadata": {"source_path": "a.pdf"}},
            {"metadata": {"source_path": "b.pdf"}},
        ]
        tool = ListDocumentsTool(mock_store)
        obs = tool.run()
        assert "a.pdf" in obs
        assert "b.pdf" in obs

    def test_empty_knowledge_base(self):
        mock_store = MagicMock()
        mock_store.get_all.return_value = []
        tool = ListDocumentsTool(mock_store)
        obs = tool.run()
        assert "empty" in obs.lower() or "No documents" in obs

    def test_error_handled(self):
        mock_store = MagicMock()
        mock_store.get_all.side_effect = Exception("fail")
        tool = ListDocumentsTool(mock_store)
        obs = tool.run()
        assert "list_documents failed" in obs
