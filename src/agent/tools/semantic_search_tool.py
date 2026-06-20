"""SemanticSearchTool — pure dense (vector) retrieval via DenseRetriever."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.agent.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


class SemanticSearchTool(BaseTool):
    """Semantic (embedding-based) search over the knowledge base.

    Use this tool when the question requires conceptual or semantic matching
    rather than exact keyword matching.
    """

    name = "semantic_search"
    description = (
        "Search the knowledge base using dense vector embeddings (semantic similarity). "
        "Best for conceptual questions and paraphrase matching."
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
        dense_retriever: DenseRetriever,
        default_top_k: int = 5,
        default_collection: Optional[str] = None,
    ) -> None:
        self._retriever = dense_retriever
        self._default_top_k = default_top_k
        self._default_collection = default_collection

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
            results = self._retriever.retrieve(query=query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning("SemanticSearchTool error: %s", exc)
            return f"[semantic_search failed: {exc}]"

        if not results:
            return "No relevant results found for the semantic search query."

        lines = [f"Found {len(results)} results for semantic search '{query}':"]
        for i, r in enumerate(results, 1):
            src = r.metadata.get("source_path", "unknown")
            snippet = (r.text or "")[:200].replace("\n", " ")
            lines.append(f"[{i}] score={r.score:.4f} | source={src}\n    {snippet}")
        return "\n".join(lines)
