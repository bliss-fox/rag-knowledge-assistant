from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.core.trace import TraceContext
from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.jobs import JobQueue
from src.production.trace_store import SQLiteTraceStore, percentile
from src.production.audit import AuditService


def test_database_uses_wal_and_migrates(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    health = database.health()
    assert health == {"ok": True, "schema_version": 3, "journal_mode": "wal"}


def test_auth_bootstrap_login_refresh_revoke_and_roles(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("admin.user", "correct horse battery staple")
    assert admin.role == "admin"
    tokens = auth.authenticate("admin.user", "correct horse battery staple")
    assert auth.verify_access(tokens["access_token"]).user_id == admin.user_id
    normal = auth.create_user("normal.user", "another correct password", "user", admin)
    assert normal.role == "user"
    assert [item["username"] for item in auth.list_users(admin)] == ["admin.user", "normal.user"]
    replacement = auth.refresh(tokens["refresh_token"])
    try:
        auth.refresh(tokens["refresh_token"])
        raise AssertionError("rotated refresh token was accepted")
    except PermissionError:
        pass
    auth.logout(replacement["refresh_token"])
    try:
        auth.refresh(replacement["refresh_token"])
        raise AssertionError("revoked refresh token was accepted")
    except PermissionError:
        pass


def test_job_queue_resource_lock_retry_cancel_and_recovery(tmp_path):
    queue = JobQueue(ProductionDatabase(tmp_path / "production.db"))
    first = queue.enqueue("sync", {}, resource_key="source:1", max_attempts=2)
    try:
        queue.enqueue("sync", {}, resource_key="source:1")
        raise AssertionError("active resource lock was not enforced")
    except RuntimeError:
        pass
    claimed = queue.claim_next("worker")
    assert claimed and claimed["id"] == first and claimed["attempts"] == 1
    queue.fail(first, "temporary")
    assert queue.get(first)["status"] == "queued"
    claimed = queue.claim_next("worker")
    assert claimed and claimed["attempts"] == 2
    queue.cancel(first)
    queue.fail(first, "cancelled")
    assert queue.get(first)["status"] == "cancelled"


def test_trace_redaction_metrics_visibility_and_retention(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    user = auth.bootstrap_admin("trace.admin", "correct horse battery staple")
    store = SQLiteTraceStore(database, retention_days=30)
    trace = TraceContext(trace_type="query")
    trace.started_at = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    trace.metadata.update({
        "user_id": user.user_id, "query": "api_key=secret tell me", "response": "answer [1]",
        "status": "answered", "citation_coverage": 1.0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    trace.record_stage("generation", {
        "prompt": "password: secret",
        "response": "answer",
        "candidates": [{
            "chunk_id": "chunk-1",
            "score": 0.9,
            "metadata": {"text_preview": "private nested prose", "rank": 1},
        }],
    }, elapsed_ms=100)
    trace.finish()
    store.save(trace.to_dict())
    loaded = store.get(trace.trace_id, user.user_id, False)
    assert loaded and "secret" not in loaded["query_text"]
    assert store.get(trace.trace_id, "u2", False) is None
    metrics = store.metrics()
    assert metrics["request_count"] == 1
    assert metrics["latency_ms"]["p50"] > 0
    assert metrics["stage_latency_ms"]["generation"]["p95"] == 100
    recent = datetime.now(timezone.utc).isoformat()
    assert store.metrics(start=recent)["request_count"] == 0
    assert store.metrics(end=recent)["stage_latency_ms"]["generation"]["p50"] == 100
    assert store.purge_content() == 1
    purged = store.get(trace.trace_id, user.user_id, False)
    assert purged["query_text"] is None
    stage_data = purged["stages"][0]["data"]
    assert "prompt" not in stage_data and "response" not in stage_data
    assert stage_data["candidates"] == [{
        "chunk_id": "chunk-1", "score": 0.9, "metadata": {"rank": 1},
    }]
    assert "private nested prose" not in json.dumps(stage_data)
    assert percentile([1, 2, 100], .95) == 90.19999999999999


def test_window_comparison_includes_versions_and_deployments(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    store = SQLiteTraceStore(database)
    audit = AuditService(database)
    now = datetime.now(timezone.utc)
    audit.deployment("config", "cfg-2", None, {"reason": "test"})
    trace = TraceContext(trace_type="query")
    trace.metadata.update({
        "status": "answered", "config_version": "cfg-2",
        "model_version": "model-1", "prompt_version": "1.0.0",
    })
    trace.finish()
    store.save(trace.to_dict())
    result = store.compare_windows(
        (now - timedelta(minutes=1)).isoformat(),
        (now + timedelta(minutes=1)).isoformat(),
        (now - timedelta(minutes=2)).isoformat(),
        (now - timedelta(minutes=1)).isoformat(),
    )
    assert result["versions"]["current"]["config_version"] == {"cfg-2": 1}
    assert result["deployment_events"][0]["detail"] == {"reason": "test"}
    assert any(item["kind"] == "version_introduced" for item in result["root_cause_candidates"])
    assert "not proof of causation" in result["causality_notice"]


def test_trace_time_series_groups_metrics_without_content(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    store = SQLiteTraceStore(database)
    start = datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)
    for index, status in enumerate(("answered", "error")):
        store.save({
            "trace_id": f"trace-{index}", "trace_type": "query",
            "started_at": (start + timedelta(minutes=10 + index)).isoformat(),
            "finished_at": (start + timedelta(minutes=11 + index)).isoformat(),
            "total_elapsed_ms": 100 + index * 100,
            "metadata": {
                "query": f"private query {index}", "response": f"private answer {index}",
                "status": status, "citation_coverage": 1.0 - index,
                "faithfulness": 0.9 - index * 0.1,
                "error_type": "ollama_timeout" if status == "error" else None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            "stages": [],
        })
    rows = store.time_series(start.isoformat(), (start + timedelta(hours=1)).isoformat())
    assert rows == [{
        "bucket": start.isoformat(), "request_count": 2, "success_rate": 0.5,
        "failure_rate": 0.5, "refusal_rate": 0.0, "degraded_rate": 0.0,
        "latency_p50_ms": 150.0, "latency_p95_ms": 195.0,
        "input_tokens": 20, "output_tokens": 10, "citation_coverage": 0.5,
        "faithfulness": 0.8500000000000001, "error_counts": {"ollama_timeout": 1},
    }]
    assert "private" not in json.dumps(rows)
    try:
        store.time_series(start.isoformat(), start.isoformat())
        raise AssertionError("invalid time window was accepted")
    except ValueError:
        pass


def test_trace_list_supports_time_and_version_filters(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    store = SQLiteTraceStore(database)
    for version in ("prompt-1", "prompt-2"):
        trace = TraceContext(trace_type="query")
        trace.metadata.update({
            "status": "answered", "prompt_version": version,
            "model_version": "model-a", "dataset_version": "data-a",
            "config_version": "config-a",
        })
        trace.finish()
        store.save(trace.to_dict())
    rows = store.list(
        None, True, prompt_version="prompt-2", model_version="model-a",
        dataset_version="data-a", config_version="config-a",
    )
    assert [row["prompt_version"] for row in rows] == ["prompt-2"]
    assert store.list(None, True, start="2999-01-01T00:00:00+00:00") == []
