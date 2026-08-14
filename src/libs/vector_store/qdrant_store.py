"""Qdrant vector-store adapter with isolated, versioned collections."""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class QdrantStore(BaseVectorStore):
    """Store project vectors in a collection isolated from the legacy KB."""

    def __init__(self, settings: Settings, client: Any = None, **kwargs: Any) -> None:
        config = settings.vector_store
        logical_name = str(kwargs.get("collection_name", config.collection_name))
        self.logical_collection_name = logical_name
        self.collection_name = self.physical_name(
            kwargs.get("collection_prefix", config.collection_prefix),
            logical_name,
            kwargs.get("index_version", config.index_version),
        )
        self.dimensions = int(kwargs.get("dimensions", settings.embedding.dimensions))
        self.url = str(kwargs.get("url", config.url))
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError("qdrant-client is required for QdrantStore") from exc
            api_key = kwargs.get("api_key", config.api_key) or None
            self.client = QdrantClient(url=self.url, api_key=api_key, timeout=30)
        else:
            self.client = client
        self._ensure_collection()

    @staticmethod
    def physical_name(prefix: str, logical_name: str, version: str) -> str:
        parts = [prefix, logical_name, version]
        clean_parts = [re.sub(r"[^a-zA-Z0-9_-]+", "_", p).strip("_") for p in parts]
        value = "_".join(part for part in clean_parts if part)
        if not value or len(value) > 255:
            raise ValueError("Invalid or overlong Qdrant collection name")
        if value == "local_knowledge":
            raise ValueError("The legacy local_knowledge collection is reserved")
        return value

    def _ensure_collection(self) -> None:
        try:
            if self.client.collection_exists(self.collection_name):
                return
            from qdrant_client.models import Distance, VectorParams
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Qdrant collection: {exc}") from exc

    def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
        self.validate_records(records)
        try:
            from qdrant_client.models import PointStruct
            points = []
            for record in records:
                metadata = dict(record.get("metadata", {}))
                text = str(record.get("text", metadata.pop("text", "")))
                original_id = str(record["id"])
                payload = {"original_id": original_id, "text": text, "metadata": self._json_safe(metadata)}
                points.append(PointStruct(id=self._point_id(original_id), vector=list(record["vector"]), payload=payload))
            self.client.upsert(self.collection_name, points=points, wait=True)
            if trace is not None:
                trace.record_stage("qdrant_upsert", {"collection": self.collection_name, "count": len(points)})
        except Exception as exc:
            raise RuntimeError(f"Qdrant upsert failed: {exc}") from exc

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)
        try:
            query_filter = self._build_filter(filters or {})
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=list(vector),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points
            else:  # qdrant-client compatibility
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=list(vector),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
            return [self._point_to_record(point) for point in points]
        except Exception as exc:
            raise RuntimeError(f"Qdrant query failed: {exc}") from exc

    def get_by_ids(self, ids: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        if not ids:
            raise ValueError("ids cannot be empty")
        try:
            points = self.client.retrieve(
                self.collection_name, ids=[self._point_id(item) for item in ids], with_payload=True
            )
            found = {
                str((getattr(point, "payload", None) or {}).get("original_id", point.id)):
                self._point_to_record(point) for point in points
            }
            return [found.get(str(item), {}) for item in ids]
        except Exception as exc:
            raise RuntimeError(f"Qdrant retrieve failed: {exc}") from exc

    def delete(self, ids: List[str], trace: Optional[Any] = None, **kwargs: Any) -> None:
        if not ids:
            raise ValueError("ids cannot be empty")
        try:
            from qdrant_client.models import PointIdsList
            self.client.delete(
                self.collection_name,
                points_selector=PointIdsList(points=[self._point_id(item) for item in ids]),
                wait=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Qdrant delete failed: {exc}") from exc

    def clear(self, collection_name: Optional[str] = None, trace: Optional[Any] = None, **kwargs: Any) -> None:
        target = collection_name or self.collection_name
        if target == "local_knowledge":
            raise ValueError("Refusing to clear reserved collection")
        self.client.delete_collection(target)
        if target == self.collection_name:
            self._ensure_collection()

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "count": int(getattr(info, "points_count", 0) or 0),
            "dimensions": self.dimensions,
        }

    def promote_alias(self, alias: str) -> None:
        """Atomically point an alias at this validated collection."""
        if alias == "local_knowledge":
            raise ValueError("The legacy alias is reserved")
        from qdrant_client.models import CreateAliasOperation, CreateAlias, DeleteAlias, DeleteAliasOperation
        operations: list[Any] = []
        try:
            aliases = self.client.get_aliases().aliases
            if any(item.alias_name == alias for item in aliases):
                operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
        except Exception:
            pass
        operations.append(CreateAliasOperation(create_alias=CreateAlias(
            collection_name=self.collection_name, alias_name=alias,
        )))
        self.client.update_collection_aliases(change_aliases_operations=operations)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _json_safe(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(metadata, ensure_ascii=False, default=str))

    @staticmethod
    def _point_to_record(point: Any) -> Dict[str, Any]:
        payload = dict(getattr(point, "payload", None) or {})
        metadata = dict(payload.get("metadata", {}) or {})
        text = str(payload.get("text", metadata.get("text", "")))
        return {
            "id": str(payload.get("original_id", point.id)),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "text": text,
            "metadata": metadata,
        }

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Any:
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
        must = []
        for key, value in filters.items():
            match = MatchAny(any=value) if isinstance(value, list) else MatchValue(value=value)
            must.append(FieldCondition(key=f"metadata.{key}", match=match))
        return Filter(must=must)

    @staticmethod
    def _point_id(original_id: str) -> str:
        try:
            return str(uuid.UUID(original_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"modular-rag:{original_id}"))
