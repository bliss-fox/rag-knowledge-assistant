"""DocumentSummaryTool — retrieve title/summary/tags for a specific document."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.agent.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DocumentSummaryTool(BaseTool):
    """Get title, summary, and tags for a specific document in the knowledge base.

    Use this after hybrid_search when you need richer metadata about a particular
    document (e.g., to understand its scope before drilling into its chunks).
    """

    name = "get_document_summary"
    description = (
        "Get title, summary, and tags for a specific document. "
        "Provide the source_path (file path) of the document."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": "The source file path of the document (from search results)",
            },
            "collection": {
                "type": "string",
                "description": "Collection name (optional)",
            },
        },
        "required": ["source_path"],
    }

    def __init__(
        self,
        vector_store: BaseVectorStore,
        default_collection: Optional[str] = None,
    ) -> None:
        self._store = vector_store
        self._default_collection = default_collection

    def run(
        self,
        source_path: str,
        collection: Optional[str] = None,
        **_: Any,
    ) -> str:
        try:
            results = self._store.get_by_metadata(
                filters={"source_path": source_path},
                limit=10,
            )
        except Exception as exc:
            logger.warning("DocumentSummaryTool error: %s", exc)
            return f"[get_document_summary failed: {exc}]"

        if not results:
            return f"No document found with source_path='{source_path}'."

        # Aggregate metadata from the first few chunks
        first = results[0]
        meta = first.get("metadata", {}) if isinstance(first, dict) else getattr(first, "metadata", {})
        text = first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")

        title = meta.get("title", "") or meta.get("doc_title", "")
        summary = meta.get("summary", "") or (text[:300].replace("\n", " ") if text else "")
        tags = meta.get("tags", [])
        chunk_count = len(results)

        lines = [f"Document summary for: {source_path}"]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Chunks: {chunk_count}")
        if tags:
            lines.append(f"Tags: {', '.join(tags) if isinstance(tags, list) else tags}")
        if summary:
            lines.append(f"Summary: {summary[:400]}")
        return "\n".join(lines)
