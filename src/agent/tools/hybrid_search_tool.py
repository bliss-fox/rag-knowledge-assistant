"""HybridSearchTool — combined dense+sparse retrieval via HybridSearch."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agent.tools.base_tool import BaseTool
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


class HybridSearchTool(BaseTool):
    """Hybrid (dense + BM25 + RRF fusion) search over the knowledge base.

    Use this as the default search tool for most questions, combining the
    strengths of semantic and keyword retrieval via Reciprocal Rank Fusion.
    """

    name = "hybrid_search"
    description = (
        "Search the knowledge base using Hybrid Search (Dense + BM25 + RRF fusion). "
        "Best all-round tool — use this first for most questions."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            "collection": {"type": "string", "description": "Collection name (optional)"},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        hybrid_search: HybridSearch,
        default_top_k: int = 5,
        default_collection: Optional[str] = None,
        # Shared list so the agent can read retrieved context after tool calls
        context_sink: Optional[List[RetrievalResult]] = None,
    ) -> None:
        self._hybrid = hybrid_search
        self._default_top_k = default_top_k
        self._default_collection = default_collection
        self._context_sink = context_sink  # agent writes results here for citation tracking

    def run(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        **_: Any,
    ) -> str:
        top_k = top_k or self._default_top_k
        filters = {"collection": collection or self._default_collection} if (
            collection or self._default_collection
        ) else None

        try:
            results = self._hybrid.search(query=query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning("HybridSearchTool error: %s", exc)
            return f"[hybrid_search failed: {exc}]"

        if not results:
            return f"No results found for '{query}'."

        # Feed results into context_sink for citation collection
        if self._context_sink is not None:
            self._context_sink.extend(results)

        lines = [f"Found {len(results)} results for hybrid search '{query}':"]
        for i, r in enumerate(results, 1):
            src = r.metadata.get("source_path", "unknown")
            title = r.metadata.get("title", "")
            snippet = (r.text or "")[:300].replace("\n", " ")
            label = f"{title} ({src})" if title else src
            lines.append(f"[{i}] score={r.score:.4f} | {label}\n    {snippet}")
        return "\n".join(lines)
