"""E2E smoke tests for the Streamlit Dashboard pages.

Uses Streamlit's ``AppTest`` framework to render each page's ``render()``
function in headless mode and verify that:

1. No Python exception is raised during render.
2. Each page produces at least one expected UI element (header / info / metric).

These tests do **not** require live data – they should pass on a fresh
checkout where the vector store is empty.

Usage::

    pytest tests/e2e/test_dashboard_smoke.py -v
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────


def _mock_settings() -> MagicMock:
    """Return a minimal mock Settings that satisfies all dashboard pages."""
    s = MagicMock()
    s.llm.provider = "azure"
    s.llm.model = "gpt-4o"
    s.llm.temperature = 0.0
    s.llm.max_tokens = 4096

    s.embedding.provider = "azure"
    s.embedding.model = "text-embedding-ada-002"
    s.embedding.dimensions = 1536

    s.vector_store.provider = "chroma"
    s.vector_store.collection_name = "default"
    s.vector_store.persist_directory = "./data/db/chroma"

    s.retrieval.dense_top_k = 20
    s.retrieval.sparse_top_k = 20
    s.retrieval.fusion_top_k = 10

    s.rerank.enabled = False
    s.rerank.provider = "none"
    s.rerank.model = ""
    s.rerank.top_k = 5

    s.vision_llm.enabled = False
    s.vision_llm.provider = "azure"
    s.vision_llm.model = "gpt-4o"
    s.vision_llm.max_image_size = 2048

    s.observability.log_level = "INFO"
    s.observability.trace_enabled = True
    s.observability.trace_file = "./logs/traces.jsonl"
    s.observability.structured_logging = True

    s.ingestion.chunk_size = 1000
    s.ingestion.chunk_overlap = 200
    s.ingestion.splitter = "recursive"
    s.ingestion.batch_size = 100
    return s


def _collect_text(at: Any) -> str:
    """Collect all rendered text from an AppTest run for assertion."""
    parts: list[str] = []
    for attr in ("markdown", "header", "subheader", "info", "error", "title", "text", "success", "warning"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


# ── Tests ─────────────────────────────────────────────────────────────


class TestDashboardSmoke:
    """Smoke tests: each page renders without uncaught exceptions."""

    # ------------------------------------------------------------------
    # 1. Overview page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_overview_page_renders(self) -> None:
        """Overview page loads and shows system overview header."""
        from streamlit.testing.v1 import AppTest

        def page_script():
            from src.observability.dashboard.pages.overview import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.services.config_service.load_settings",
            return_value=_mock_settings(),
        ):
            at.run()

        assert not at.exception, (
            f"Overview page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert any(label in text.lower() for label in ("overview", "system", "系统概览"))

    # ------------------------------------------------------------------
    # 2. Data Browser page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_data_browser_page_renders(self) -> None:
        """Data Browser page loads (may show 'no documents' info)."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []

        def page_script():
            from src.observability.dashboard.pages.data_browser import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.data_browser.DataService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Data Browser page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert any(
            label in text.lower()
            for label in ("data", "browser", "document", "知识库浏览", "文档")
        )

    # ------------------------------------------------------------------
    # 3. Ingestion Manager page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_ingestion_manager_page_renders(self) -> None:
        """Ingestion Manager page loads without errors."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_manager import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.ingestion_manager.DataService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Ingestion Manager page raised an exception: {at.exception}"
        )

    # ------------------------------------------------------------------
    # 4. Ingestion Traces page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_ingestion_traces_page_renders(self) -> None:
        """Ingestion Traces page loads (empty trace list is OK)."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_traces import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.ingestion_traces.TraceService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Ingestion Traces page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert any(label in text.lower() for label in ("trace", "ingestion", "导入追踪"))

    # ------------------------------------------------------------------
    # 5. Query Traces page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_query_traces_page_renders(self) -> None:
        """Query Traces page loads (empty trace list is OK)."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.query_traces import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.query_traces.TraceService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Query Traces page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert any(label in text.lower() for label in ("query", "trace", "查询追踪"))

    # ------------------------------------------------------------------
    # 6. Evaluation Panel page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_evaluation_panel_page_renders(self) -> None:
        """Production evaluation UI renders from API-backed history."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            [{
                "id": "eval-1", "created_at": "2026-08-13T00:00:00Z",
                "status": "completed", "dataset_name": "public_golden.json",
                "dataset_version": "1.0.1", "split": "dev", "metrics": {},
            }],
            {
                "id": "eval-1", "status": "completed", "dataset_version": "1.0.1",
                "split": "dev", "metrics": {
                    "bm25": {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0,
                               "ndcg_at_k": 1.0, "latency_p50_ms": 1.0,
                               "latency_p95_ms": 2.0},
                },
                "report": {"dataset": {"query_count": 3}, "variants": []},
            },
        ]

        def page_script():
            from src.observability.dashboard.app import _evaluations
            _evaluations()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()

        assert not at.exception, (
            f"Evaluation Panel page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "离线评测" in text
        assert "消融指标" in text

    @pytest.mark.e2e
    def test_production_trace_waterfall_and_rerank_comparison_render(self) -> None:
        """API-backed Trace page renders filters, timing chart and rank comparison."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            [{
                "id": "trace-1", "started_at": "2026-08-13T00:00:00Z",
                "status": "answered",
            }],
            {
                "id": "trace-1", "status": "answered", "total_elapsed_ms": 42.0,
                "input_tokens": 10, "output_tokens": 5, "prompt_version": "1.0.0",
                "model_version": "qwen3:8b", "dataset_version": "v1",
                "config_version": "cfg-1", "stages": [{
                    "stage": "rerank", "elapsed_ms": 12.0, "data": {
                        "input_order": [{"rank": 1, "chunk_id": "a", "score": .7}],
                        "output_order": [{"rank": 1, "chunk_id": "b", "score": .9}],
                        "used_fallback": False,
                    },
                }],
            },
        ]

        def page_script():
            from src.observability.dashboard.app import _traces
            _traces()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()

        assert not at.exception, f"Production Trace page raised an exception: {at.exception}"
        text = _collect_text(at)
        assert "Trace 瀑布" in text
        assert "阶段耗时瀑布" in text
        assert "Rerank 前后顺序" in text
        assert {item.label for item in at.text_input} >= {
            "Prompt 版本", "模型版本", "数据集 / 索引版本", "配置版本哈希",
        }
        request_path = mock_client.get.call_args_list[0].args[0]
        assert "start=" in request_path and "end=" in request_path

    @pytest.mark.e2e
    def test_production_user_management_renders_and_updates_account(self) -> None:
        """Admin user management exposes role, activation and password controls."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.return_value = [{
            "id": "user-1", "username": "alice", "role": "user", "active": True,
        }]

        def page_script():
            from src.observability.dashboard.app import _users
            _users()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()
            assert not at.exception, f"User management raised an exception: {at.exception}"
            assert "用户管理" in _collect_text(at)
            assert {item.label for item in at.text_input} >= {
                "新用户名", "初始密码", "重置密码（留空不改）",
            }
            at.button(key="save-user-user-1").click().run()

        assert not at.exception, f"Saving an account raised an exception: {at.exception}"
        mock_client.patch.assert_called_once_with(
            "/users/user-1", json={"role": "user", "active": True},
        )

    @pytest.mark.e2e
    def test_production_sources_render_management_and_cancel_job(self) -> None:
        """Data-source edit/delete controls and job cancellation render and call the API."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.return_value = [{
            "id": "source-1", "name": "知识库", "kind": "directory",
            "location": r"D:\AI-KnowledgeBase\documents", "collection_name": "default",
            "enabled": True, "config": {},
        }]
        mock_client.post.return_value = {"id": "job-1", "status": "cancel_requested"}

        def page_script():
            from src.observability.dashboard.app import _sources
            _sources()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()
            assert not at.exception, f"Data sources raised an exception: {at.exception}"
            labels = {item.label for item in at.button}
            assert {"保存", "开始增量同步", "删除数据源"} <= labels
            next(item for item in at.text_input if item.label == "任务 ID").set_value("job-1").run()
            labels = {item.label for item in at.button}
            assert {"刷新任务", "取消任务"} <= labels
            next(item for item in at.button if item.label == "取消任务").click().run()

        assert not at.exception, f"Cancelling a job raised an exception: {at.exception}"
        mock_client.post.assert_called_once_with("/jobs/job-1/cancel")

    @pytest.mark.e2e
    def test_production_observability_renders_trends_and_root_cause_candidates(self) -> None:
        """Metrics UI renders time series and clearly qualified root-cause candidates."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.side_effect = [
            {
                "request_count": 10, "latency_ms": {"p50": 100, "p95": 250},
                "failure_rate": 0.1, "citation_coverage": 0.96,
                "stage_latency_ms": {"generation": {"p50": 80, "p95": 200}},
                "status_counts": {"answered": 9, "error": 1},
                "error_counts": {"ollama_timeout": 1},
            },
            [{
                "bucket": "2026-08-13T08:00:00+00:00", "request_count": 10,
                "success_rate": 0.9, "failure_rate": 0.1, "refusal_rate": 0.0,
                "degraded_rate": 0.0, "latency_p50_ms": 100, "latency_p95_ms": 250,
                "input_tokens": 100, "output_tokens": 50,
                "citation_coverage": 0.96, "faithfulness": 0.9,
                "error_counts": {"ollama_timeout": 1},
            }],
            {
                "delta": {"failure_rate": 0.1, "p95_latency_ms": 50},
                "root_cause_candidates": [{
                    "kind": "error_increase", "signal": "ollama_timeout",
                    "delta": 0.1, "confidence": "correlated",
                }],
                "versions": {"current": {}, "baseline": {}},
                "deployment_events": [],
            },
        ]

        def page_script():
            from src.observability.dashboard.app import _observability
            _observability()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()

        assert not at.exception, f"Observability page raised an exception: {at.exception}"
        text = _collect_text(at)
        assert "质量与可靠性趋势" in text
        assert "根因候选" in text
        assert any("不代表已证明因果" in item.value for item in at.caption)
        paths = [call.args[0] for call in mock_client.get.call_args_list]
        assert any(path.startswith("/metrics/timeseries?") for path in paths)

    @pytest.mark.e2e
    def test_golden_candidate_review_renders_evidence_and_saves_approval(self) -> None:
        """Admin can inspect candidate evidence and persist a hash-bound approval."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        summary_payload = {
                "total": 100, "version": "2026.08.13.1", "dataset_sha256": "sha-1",
                "all_approved": False, "eligible_for_quality_gate": False,
                "counts": {"unreviewed": 100, "approved": 0, "rejected": 0, "pending": 0},
            }
        cases_payload = [{
                "query_id": "candidate-1", "query": "What is RRF?", "category": "fact",
                "language": "en", "split": "dev", "answerable": True,
                "review_status": "unreviewed", "review": None,
            }]
        detail_payload = {
                "dataset_sha256": "sha-1",
                "case": {
                    "query_id": "candidate-1", "query": "What is RRF?", "category": "fact",
                    "difficulty": "easy", "language": "en", "split": "dev",
                    "answerable": True, "answer_key_points": ["60"],
                    "expected_document_ids": ["doc-1"], "candidate_metadata": {},
                },
                "evidence": [{
                    "document_id": "doc-1", "title": "Evidence",
                    "content": "# Evidence\n\nRRF uses 60.",
                }],
                "review": None,
            }

        def fake_get(path: str):
            if path == "/golden-candidates/summary":
                return summary_payload
            if path.startswith("/golden-candidates/cases?"):
                return cases_payload
            if path == "/golden-candidates/cases/candidate-1":
                return detail_payload
            raise AssertionError(f"unexpected dashboard GET: {path}")

        mock_client.get.side_effect = fake_get

        def page_script():
            from src.observability.dashboard.app import _golden_review
            _golden_review()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()
            assert not at.exception, f"Golden review raised an exception: {at.exception}"
            text = _collect_text(at)
            assert "黄金集候选复核" in text
            assert "支持证据" in text
            at.button(key="approve-candidate-1").click().run()

        assert not at.exception, f"Golden approval raised an exception: {at.exception}"
        mock_client.put.assert_called_once_with(
            "/golden-candidates/cases/candidate-1/review",
            json={"dataset_sha256": "sha-1", "status": "approved", "notes": ""},
        )

    @pytest.mark.e2e
    def test_golden_candidate_review_exports_only_after_all_approved(self) -> None:
        """The admin UI exposes the reviewed artifact export at 100% approval."""
        from streamlit.testing.v1 import AppTest

        mock_client = MagicMock()
        mock_client.get.side_effect = lambda path: {
            "/golden-candidates/summary": {
                "total": 1, "version": "1", "dataset_sha256": "sha-1",
                "all_approved": True, "eligible_for_quality_gate": False,
                "counts": {"unreviewed": 0, "approved": 1, "rejected": 0, "pending": 0},
            },
            "/golden-candidates/cases?status=unreviewed": [],
        }[path]
        mock_client.post.return_value = {
            "path": "evaluation/reviewed/public_golden_reviewed-sha.json",
            "eligible_for_quality_gate": False,
        }

        def page_script():
            from src.observability.dashboard.app import _golden_review
            _golden_review()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch("src.observability.dashboard.app._client", return_value=mock_client):
            at.run()
            at.button(key="export-reviewed-golden").click().run()

        assert not at.exception, f"Golden export raised an exception: {at.exception}"
        mock_client.post.assert_called_once_with(
            "/golden-candidates/export-reviewed", json={"dataset_sha256": "sha-1"},
        )
        assert "仍不可用于质量门禁" in _collect_text(at)
