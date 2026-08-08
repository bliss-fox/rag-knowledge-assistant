"""KeywordSearchTool — pure sparse (BM25) retrieval via SparseRetriever."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agent.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.core.query_engine.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class KeywordSearchTool(BaseTool):
    """Keyword (BM25) search over the knowledge base.

    Use this tool when the question contains specific technical terms,
    proper nouns, or exact phrases that benefit from keyword matching.
    """

    name = "keyword_search"
    description = (
        "Search the knowledge base using BM25 keyword matching. "
        "Best for exact term lookup, technical jargon, and proper nouns."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of keywords to search for",
            },
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            "collection": {"type": "string", "description": "Collection name (optional)"},
        },
        "required": ["keywords"],
    }

    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        default_top_k: int = 5,
        default_collection: Optional[str] = None,
    ) -> None:
        self._retriever = sparse_retriever
        self._default_top_k = default_top_k
        self._default_collection = default_collection

    def run(
        self,
        keywords: List[str],
        top_k: int = 5,
        collection: Optional[str] = None,
        **_: Any,
    ) -> str:
        if not keywords:
            return "[keyword_search error: keywords list is empty]"

        top_k = top_k or self._default_top_k
        coll = collection or self._default_collection

        try:
            results = self._retriever.retrieve(
                keywords=keywords,
                top_k=top_k,
                collection=coll,
            )
        except Exception as exc:
            logger.warning("KeywordSearchTool error: %s", exc)
            return f"[keyword_search failed: {exc}]"

        if not results:
            return f"No results found for keywords: {keywords}"

        lines = [f"Found {len(results)} results for keywords {keywords}:"]
        for i, r in enumerate(results, 1):
            src = r.metadata.get("source_path", "unknown")
            snippet = (r.text or "")[:200].replace("\n", " ")
            lines.append(f"[{i}] score={r.score:.4f} | source={src}\n    {snippet}")
        return "\n".join(lines)
