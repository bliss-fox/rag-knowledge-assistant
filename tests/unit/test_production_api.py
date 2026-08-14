from __future__ import annotations

import threading
from dataclasses import replace

from fastapi.testclient import TestClient

from src.core.settings import load_settings
from src.production.api import create_app
from src.production.audit import AuditService
from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.jobs import JobQueue
from src.production.runtime import Runtime
from src.production.sources import DataSourceService
from src.production.trace_store import SQLiteTraceStore


class FakeRAG:
    def search(self, query, collection, top_k, user_id, enable_rerank=True):
        return {
            "status": "answered", "query": query, "results": [], "stages": {},
            "degraded": False, "degradation_reason": None,
            "retrieval_method": "bm25+dense+rrf+fake", "trace_id": "search-trace",
        }

    def answer(self, query, collection, top_k, user_id):
        return {
            "status": "refused", "answer": "现有知识库证据不足，无法可靠回答。",
            "citations": [], "evidence": {"accepted": False, "reason": "empty_results"},
            "trace_id": "answer-trace", "prompt_version": "1.0.0", "model_version": "fake",
        }

    def close(self):
        return None


def _runtime(tmp_path):
    settings = load_settings()
    settings = replace(
        settings,
        observability=replace(settings.observability, sqlite_path=str(tmp_path / "production.db")),
        server=replace(settings.server, rate_limit_per_minute=1000),
        evidence=replace(settings.evidence, calibration_status="calibrated"),
    )
    database = ProductionDatabase(tmp_path / "production.db")
    return Runtime(
        settings=settings,
        database=database,
        auth=AuthService(database, secret="x" * 64),
        jobs=JobQueue(database),
        traces=SQLiteTraceStore(database),
        sources=DataSourceService(database, [tmp_path]),
        audit=AuditService(database),
        prompts=None,
        rag=FakeRAG(),
    )


def _bootstrap(client):
    response = client.post("/auth/bootstrap", json={
        "username": "admin.user", "password": "correct horse battery staple",
    })
    assert response.status_code == 201
    return response.json()


def test_auth_user_roles_and_refresh_contract(tmp_path):
    with TestClient(create_app(_runtime(tmp_path))) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        created = client.post("/users", headers=headers, json={
            "username": "normal.user", "password": "another correct password", "role": "user",
        })
        assert created.status_code == 201
        login = client.post("/auth/login", json={
            "username": "normal.user", "password": "another correct password",
        }).json()
        user_headers = {"Authorization": f"Bearer {login['access_token']}"}
        assert client.get("/users", headers=user_headers).status_code == 403
        refreshed = client.post("/auth/refresh", json={"refresh_token": login["refresh_token"]})
        assert refreshed.status_code == 200
        assert client.post("/auth/refresh", json={"refresh_token": login["refresh_token"]}).status_code == 401


def test_search_answer_source_job_and_health_contract(tmp_path):
    source_root = tmp_path / "documents"
    source_root.mkdir()
    with TestClient(create_app(_runtime(tmp_path))) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json()["journal_mode"] == "wal"
        source = client.post("/data-sources", headers=headers, json={
            "name": "local", "kind": "directory", "location": str(source_root),
            "collection_name": "default",
        })
        assert source.status_code == 201
        sync = client.post(f"/data-sources/{source.json()['id']}/sync", headers=headers)
        assert sync.status_code == 202
        assert client.get(f"/jobs/{sync.json()['job_id']}", headers=headers).json()["status"] == "queued"
        search = client.post("/search", headers=headers, json={"query": "RRF 是什么？"})
        assert search.status_code == 200
        assert search.json()["retrieval_method"] == "bm25+dense+rrf+fake"
        answer = client.post("/answer", headers=headers, json={"query": "没有证据的问题"})
        assert answer.status_code == 200
        assert answer.json()["status"] == "refused"
        assert client.get("/metrics/summary", headers=headers).status_code == 200
        series = client.get(
            "/metrics/timeseries?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z&bucket=hour",
            headers=headers,
        )
        assert series.status_code == 200 and series.json() == []


def test_source_delete_is_queued_and_conflicts_with_active_sync(tmp_path):
    source_root = tmp_path / "documents"
    source_root.mkdir()
    with TestClient(create_app(_runtime(tmp_path))) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        source = client.post("/data-sources", headers=headers, json={
            "name": "local", "kind": "directory", "location": str(source_root),
            "collection_name": "default",
        }).json()
        sync = client.post(f"/data-sources/{source['id']}/sync", headers=headers)
        assert sync.status_code == 202
        conflict = client.delete(f"/data-sources/{source['id']}", headers=headers)
        assert conflict.status_code == 409
        runtime = client.app.state.runtime
        runtime.jobs.cancel(sync.json()["job_id"])
        queued = client.delete(f"/data-sources/{source['id']}", headers=headers)
        assert queued.status_code == 202
        job = runtime.jobs.get(queued.json()["job_id"])
        assert job["kind"] == "delete_source"
        assert job["payload"] == {"source_id": source["id"]}
        assert runtime.sources.get(source["id"])["id"] == source["id"]


def test_trace_filters_are_forwarded_to_store(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    captured = {}

    def list_traces(user_id, is_admin, limit, **filters):
        captured.update({"user_id": user_id, "is_admin": is_admin, "limit": limit, **filters})
        return []

    monkeypatch.setattr(runtime.traces, "list", list_traces)
    with TestClient(create_app(runtime)) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.get(
            "/traces?limit=25&prompt_version=p1&model_version=m1&dataset_version=d1&config_version=c1",
            headers=headers,
        )
    assert response.status_code == 200
    assert captured == {
        "user_id": tokens["user"]["id"], "is_admin": True, "limit": 25,
        "start": None, "end": None, "prompt_version": "p1", "model_version": "m1",
        "dataset_version": "d1", "config_version": "c1",
    }


def test_validation_and_auth_fail_closed(tmp_path):
    with TestClient(create_app(_runtime(tmp_path))) as client:
        assert client.post("/search", json={"query": "x"}).status_code == 401
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        assert client.post("/search", headers=headers, json={"query": "x", "top_k": 0}).status_code == 422
        assert client.post("/documents/upload", headers=headers, files={
            "file": ("malware.exe", b"x", "application/octet-stream"),
        }).status_code == 415


def test_readiness_fails_when_warmup_dependency_failed(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.rag.warmup = lambda: (_ for _ in ()).throw(RuntimeError("qdrant unavailable"))
    with TestClient(create_app(runtime)) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"


def test_readiness_fails_when_configured_reranker_is_unavailable(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.rag.warmup = lambda: {
        "reranker_configured": True,
        "reranker_ready": False,
        "reranker_error": "model unavailable",
        "evidence_calibration": "calibrated",
    }
    with TestClient(create_app(runtime)) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["warmup"]["reranker_error"] == "model unavailable"


def test_readiness_fails_while_evidence_thresholds_are_pending_calibration(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.rag.warmup = lambda: {
        "reranker_configured": True,
        "reranker_ready": True,
        "reranker_error": None,
        "evidence_calibration": "pending",
    }
    with TestClient(create_app(runtime)) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["warmup"]["evidence_calibration"] == "pending"
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        assert client.post(
            "/search", headers=headers, json={"query": "pending calibration"},
        ).status_code == 200
        assert client.post(
            "/answer", headers=headers, json={"query": "pending calibration"},
        ).status_code == 503


def test_liveness_is_available_while_dependency_warmup_is_running(tmp_path):
    runtime = _runtime(tmp_path)
    release = threading.Event()
    started = threading.Event()

    def slow_warmup():
        started.set()
        release.wait(timeout=5)
        return {
            "reranker_configured": True, "reranker_ready": True,
            "reranker_error": None, "evidence_calibration": "calibrated",
        }

    runtime.rag.warmup = slow_warmup
    try:
        with TestClient(create_app(runtime)) as client:
            assert started.wait(timeout=1)
            assert client.get("/health/live").status_code == 200
            ready = client.get("/health/ready")
            assert ready.status_code == 503
            assert ready.json()["warmup"]["status"] == "starting"
            tokens = _bootstrap(client)
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            assert client.post(
                "/search", headers=headers, json={"query": "still warming"},
            ).status_code == 503
            release.set()
    finally:
        release.set()


def test_evaluation_path_is_confined_and_json_is_decoded(tmp_path):
    runtime = _runtime(tmp_path)
    evaluation_root = tmp_path / "evaluation"
    evaluation_root.mkdir()
    dataset = evaluation_root / "golden.json"
    dataset.write_text("{}", encoding="utf-8")
    with TestClient(create_app(runtime)) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        outside = client.post("/evaluations", headers=headers, json={
            "dataset_path": str(tmp_path / "outside.json"), "split": "final",
        })
        assert outside.status_code in {400, 404}


def test_evaluation_create_list_and_detail_contract(tmp_path):
    runtime = _runtime(tmp_path)
    with TestClient(create_app(runtime)) as client:
        tokens = _bootstrap(client)
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        created = client.post("/evaluations", headers=headers, json={
            "dataset_path": "evaluation/public_golden.json",
            "split": "dev",
            "collection": "evaluation",
            "top_k": 7,
            "repeats": 2,
        })
        assert created.status_code == 202
        evaluation_id = created.json()["evaluation_id"]
        history = client.get("/evaluations", headers=headers)
        assert history.status_code == 200
        assert history.json()[0]["id"] == evaluation_id
        detail = client.get(f"/evaluations/{evaluation_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["config"]["top_k"] == 7
        assert detail.json()["config"]["repeats"] == 2
        assert detail.json()["report"] is None


def test_evaluations_are_admin_only(tmp_path):
    with TestClient(create_app(_runtime(tmp_path))) as client:
        admin_tokens = _bootstrap(client)
        admin_headers = {"Authorization": f"Bearer {admin_tokens['access_token']}"}
        client.post("/users", headers=admin_headers, json={
            "username": "eval.user", "password": "another correct password", "role": "user",
        })
        login = client.post("/auth/login", json={
            "username": "eval.user", "password": "another correct password",
        }).json()
        headers = {"Authorization": f"Bearer {login['access_token']}"}
        assert client.get("/evaluations", headers=headers).status_code == 403


def test_golden_candidate_review_contract_and_role_isolation(tmp_path):
    runtime = _runtime(tmp_path)
    with TestClient(create_app(runtime)) as client:
        tokens = _bootstrap(client)
        admin_headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        summary = client.get("/golden-candidates/summary", headers=admin_headers)
        assert summary.status_code == 200
        assert summary.json()["total"] == 100
        cases = client.get(
            "/golden-candidates/cases?status=unreviewed", headers=admin_headers,
        )
        assert cases.status_code == 200 and len(cases.json()) == 100
        query_id = cases.json()[0]["query_id"]
        detail = client.get(f"/golden-candidates/cases/{query_id}", headers=admin_headers)
        assert detail.status_code == 200
        reviewed = client.put(
            f"/golden-candidates/cases/{query_id}/review", headers=admin_headers,
            json={
                "dataset_sha256": summary.json()["dataset_sha256"],
                "status": "approved", "notes": "evidence checked",
            },
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "approved"
        export = client.post(
            "/golden-candidates/export-reviewed", headers=admin_headers,
            json={"dataset_sha256": summary.json()["dataset_sha256"]},
        )
        assert export.status_code == 409
        assert "99 remain" in export.json()["detail"]

        client.post("/users", headers=admin_headers, json={
            "username": "review.user", "password": "another correct password", "role": "user",
        })
        login = client.post("/auth/login", json={
            "username": "review.user", "password": "another correct password",
        }).json()
        user_headers = {"Authorization": f"Bearer {login['access_token']}"}
        assert client.get("/golden-candidates/summary", headers=user_headers).status_code == 403
        assert client.post(
            "/golden-candidates/export-reviewed", headers=user_headers,
            json={"dataset_sha256": summary.json()["dataset_sha256"]},
        ).status_code == 403
