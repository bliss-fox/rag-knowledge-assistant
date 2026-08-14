"""Regression tests for benchmark integrity and error reporting."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.types import RetrievalResult


def _load_script_module():
    path = Path(__file__).parents[2] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validation_rejects_replacement_characters() -> None:
    module = _load_script_module()
    case = module.TestCase(
        id="bad-1",
        query="broken \ufffd query",
        expected_sources=["doc.pdf"],
    )

    with pytest.raises(ValueError, match="replacement characters"):
        module.validate_test_cases([case])


def test_run_mode_records_search_failures_instead_of_silently_counting_miss() -> None:
    module = _load_script_module()

    class BrokenSearch:
        def search(self, **_kwargs):
            raise RuntimeError("qdrant unavailable")

    result = module.run_mode(
        "Dense Only",
        BrokenSearch(),
        [module.TestCase(id="q1", query="question", expected_sources=["doc.pdf"])],
        module.SourceMatchEvaluator(),
        top_k=5,
        repeats=2,
    )

    assert result.error_queries == 2
    assert all("qdrant unavailable" in item["error"] for item in result.per_query)


def test_run_mode_records_retrieval_degradation_reason() -> None:
    module = _load_script_module()
    chunk = RetrievalResult(
        chunk_id="c1",
        text="answer",
        score=1.0,
        metadata={"source_path": "doc.pdf"},
    )

    class DegradedSearch:
        def search(self, **_kwargs):
            return SimpleNamespace(
                results=[chunk],
                used_fallback=True,
                dense_error="embedding timeout",
                sparse_error=None,
            )

    result = module.run_mode(
        "Hybrid (RRF)",
        DegradedSearch(),
        [module.TestCase(id="q1", query="question", expected_sources=["doc.pdf"])],
        module.SourceMatchEvaluator(),
        top_k=5,
    )

    assert result.degraded_queries == 1
    assert result.error_queries == 0
    assert result.per_query[0]["fallback_reason"] == "embedding timeout"


def test_report_persists_unavailable_rerank_reason(tmp_path: Path) -> None:
    module = _load_script_module()
    output = tmp_path / "report.json"

    module.save_results(
        [],
        output,
        metadata={"status": "candidate"},
        unavailable_modes=[{"mode": "Hybrid+Rerank", "reason": "model missing"}],
    )

    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert report["unavailable_modes"] == [{"mode": "Hybrid+Rerank", "reason": "model missing"}]
