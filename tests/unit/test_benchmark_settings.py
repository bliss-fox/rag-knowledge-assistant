"""Deterministic benchmark configuration tests."""

import importlib.util
from pathlib import Path

from src.core.settings import load_settings


def _load_script_module():
    path = Path(__file__).parents[2] / "scripts" / "ingest_benchmark_docs.py"
    spec = importlib.util.spec_from_file_location("ingest_benchmark_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_overrides_are_deterministic_and_non_mutating() -> None:
    module = _load_script_module()
    production = load_settings()

    benchmark = module.deterministic_benchmark_settings(production)

    assert benchmark.ingestion.chunk_refiner["use_llm"] is False
    assert benchmark.ingestion.metadata_enricher["use_llm"] is False
    assert benchmark.vision_llm.enabled is False
    assert production.ingestion.chunk_refiner["use_llm"] is True
    assert production.ingestion.metadata_enricher["use_llm"] is True
    assert production.vision_llm.enabled is True
