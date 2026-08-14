"""Tests for truthful vector-store metadata in ingestion traces."""

from types import SimpleNamespace

from src.ingestion.pipeline import (
    pipeline_vector_store_details,
    vector_store_observability_details,
)


def test_qdrant_trace_uses_physical_collection_and_url() -> None:
    settings = SimpleNamespace(
        vector_store=SimpleNamespace(
            provider="qdrant",
            url="http://127.0.0.1:6333",
            persist_directory="unused",
        )
    )
    store = SimpleNamespace(
        collection_name="modular_rag_benchmark_v1",
        url="http://qdrant.internal:6333",
    )

    details = vector_store_observability_details(settings, store, "benchmark")

    assert details == {
        "provider": "qdrant",
        "backend": "Qdrant",
        "logical_collection": "benchmark",
        "collection": "modular_rag_benchmark_v1",
        "location": "http://qdrant.internal:6333",
    }
    assert "Chroma" not in str(details)


def test_chroma_trace_uses_persistence_directory() -> None:
    settings = SimpleNamespace(
        vector_store=SimpleNamespace(
            provider="chroma",
            url="unused",
            persist_directory="data/db/chroma",
        )
    )
    store = SimpleNamespace(
        collection_name="docs",
        persist_directory="D:/runtime/chroma",
    )

    details = vector_store_observability_details(settings, store, "docs")

    assert details["backend"] == "ChromaDB"
    assert details["collection"] == "docs"
    assert details["location"] == "D:/runtime/chroma"


def test_dependency_injected_pipeline_gets_truthful_generic_store_details() -> None:
    pipeline = SimpleNamespace(
        collection="test",
        vector_upserter=SimpleNamespace(vector_store=SimpleNamespace()),
    )

    details = pipeline_vector_store_details(pipeline)

    assert details["backend"] == "SimpleNamespace"
    assert details["logical_collection"] == "test"
    assert details["collection"] == "test"
    assert details["location"] == "unknown"
