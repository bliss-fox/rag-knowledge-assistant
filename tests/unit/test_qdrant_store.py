from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.libs.vector_store.qdrant_store import QdrantStore


def _settings():
    return SimpleNamespace(
        embedding=SimpleNamespace(dimensions=3),
        vector_store=SimpleNamespace(
            collection_name="default", collection_prefix="modular_rag", index_version="v1",
            url="http://127.0.0.1:6333", api_key=None,
        ),
    )


def test_collection_is_isolated_and_reserved_name_blocked():
    assert QdrantStore.physical_name("modular_rag", "default", "v1") == "modular_rag_default_v1"
    try:
        QdrantStore.physical_name("", "local", "knowledge")
    except ValueError:
        pass
    else:
        raise AssertionError("reserved collection was accepted")


def test_point_id_is_deterministic_uuid():
    first = QdrantStore._point_id("stable_chunk_1")
    assert first == QdrantStore._point_id("stable_chunk_1")
    assert first != QdrantStore._point_id("stable_chunk_2")


def test_get_by_ids_preserves_original_ids():
    client = Mock()
    client.collection_exists.return_value = True
    point = SimpleNamespace(
        id=QdrantStore._point_id("chunk-a"), score=0.0,
        payload={"original_id": "chunk-a", "text": "hello", "metadata": {"source_path": "x"}},
    )
    client.retrieve.return_value = [point]
    store = QdrantStore(_settings(), client=client)
    records = store.get_by_ids(["chunk-a", "missing"])
    assert records[0]["id"] == "chunk-a"
    assert records[1] == {}
