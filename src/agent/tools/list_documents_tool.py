"""ListDocumentsTool — explore what documents exist in the knowledge base."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.agent.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class ListDocumentsTool(BaseTool):
    """List the documents available in the knowledge base.

    Use this tool at the start of a research session to understand what
    content is available before deciding which search strategy to use.
    """

    name = "list_documents"
    description = (
        "List documents available in the knowledge base with their source paths. "
        "Use this to explore what content is available before searching."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": "Collection name (optional; lists default collection if omitted)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of documents to list (default: 20)",
            },
        },
        "required": [],
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
        collection: Optional[str] = None,
        limit: int = 20,
        **_: Any,
    ) -> str:
        try:
            results = self._store.get_all(limit=limit)
        except Exception as exc:
            logger.warning("ListDocumentsTool error: %s", exc)
            return f"[list_documents failed: {exc}]"

        if not results:
            return "The knowledge base is empty. No documents have been ingested yet."

        seen_sources: dict[str, int] = {}
        for r in results:
            meta = r.get("metadata", {}) if isinstance(r, dict) else getattr(r, "metadata", {})
            src = meta.get("source_path", "unknown")
            seen_sources[src] = seen_sources.get(src, 0) + 1

        lines = [f"Knowledge base contains {len(seen_sources)} document(s):"]
        for src, chunk_count in list(seen_sources.items())[:limit]:
            lines.append(f"  - {src}  ({chunk_count} chunks)")
        return "\n".join(lines)
