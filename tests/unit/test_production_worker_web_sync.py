from __future__ import annotations

import hashlib
from types import SimpleNamespace

from src.core.types import Document
from src.libs.loader.web_loader import WebCrawlResult
from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.sources import DataSourceService
from src.production.worker import Worker


class FakeJobs:
    def is_cancel_requested(self, _job_id):
        return False

    def heartbeat(self, _job_id):
        return None


class CancellingJobs(FakeJobs):
    def __init__(self):
        self.cancelled = False

    def is_cancel_requested(self, _job_id):
        return self.cancelled


class FakeVectorStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, ids):
        self.deleted.extend(ids)


def web_document(url: str, text: str) -> Document:
    digest = hashlib.sha256((url + "\n" + text).encode()).hexdigest()
    return Document(
        id=f"web_{digest[:16]}",
        text=text,
        metadata={
            "source_path": url,
            "source_uri": url,
            "doc_type": "web",
            "doc_hash": digest,
            "title": text,
        },
    )


def test_web_sync_is_incremental_and_deletes_only_after_complete_crawl(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("web.sync.admin", "correct horse battery staple")
    sources = DataSourceService(database)
    source = sources.create(
        "docs", "web", "https://example.com/", "default", admin.user_id,
        {"max_depth": 1, "max_pages": 10},
    )
    vectors = FakeVectorStore()
    runtime = SimpleNamespace(
        sources=sources,
        jobs=FakeJobs(),
        rag=SimpleNamespace(
            _pipeline=lambda _collection: SimpleNamespace(vector_store=vectors),
        ),
        settings=SimpleNamespace(
            web_crawler=SimpleNamespace(
                max_depth=1,
                max_pages=10,
                max_response_mb=1,
                timeout_seconds=1.0,
            ),
        ),
    )
    worker = Worker(runtime)
    snapshots = tmp_path / "web-snapshots"
    monkeypatch.setattr("src.core.settings.resolve_path", lambda _value: snapshots)
    removed_documents: list[tuple[str, str]] = []

    class FakeBM25:
        def __init__(self, index_dir):
            self.index_dir = index_dir

        def remove_document(self, document_id, collection):
            removed_documents.append((document_id, collection))

    monkeypatch.setattr("src.ingestion.storage.bm25_indexer.BM25Indexer", FakeBM25)
    crawl_reports = iter([
        WebCrawlResult((
            web_document("https://example.com/", "Root v1"),
            web_document("https://example.com/guide", "Guide v1"),
        ), fetched_pages=2, complete=True),
        WebCrawlResult((
            web_document("https://example.com/", "Root v1"),
        ), fetched_pages=1, complete=False, truncation_reason="page_limit_reached"),
        WebCrawlResult((
            web_document("https://example.com/", "Root v2"),
        ), fetched_pages=1, complete=True),
    ])
    monkeypatch.setattr(
        "src.libs.loader.web_loader.WebLoader.crawl_with_report",
        lambda _loader, _url: next(crawl_reports),
    )
    ingestion_calls: list[dict[str, object]] = []

    def fake_ingest(payload, _job_id):
        content = open(payload["path"], encoding="utf-8").read()
        digest = hashlib.sha256(content.encode()).hexdigest()[:12]
        ingestion_calls.append(dict(payload))
        return {
            "stages": {"loading": {"doc_id": f"doc-{digest}"}},
            "vector_ids": [f"chunk-{digest}"],
        }

    monkeypatch.setattr(worker, "_ingest_file", fake_ingest)

    first = worker._sync_web(source, "job-1")
    manifests_after_first = sources.list_manifests(source["id"])
    snapshot_paths = sorted((snapshots / source["id"]).glob("*.md"))
    assert first["processed"] == 2
    assert len(first["changes"]["added"]) == 2
    assert len(manifests_after_first) == 2
    assert len(snapshot_paths) == 2

    truncated = worker._sync_web(source, "job-2")
    assert truncated["processed"] == 0
    assert truncated["deleted"] == 0
    assert truncated["deletion_skipped_reason"] == "page_limit_reached"
    assert len(sources.list_manifests(source["id"])) == 2
    assert len(ingestion_calls) == 2

    final = worker._sync_web(source, "job-3")
    final_manifests = sources.list_manifests(source["id"])
    final_paths = sorted((snapshots / source["id"]).glob("*.md"))
    assert final["processed"] == 1
    assert len(final["changes"]["modified"]) == 1
    assert len(final["changes"]["deleted"]) == 1
    assert final["deleted"] == 1
    assert len(final_manifests) == 1
    assert len(final_paths) == 1
    assert all(call["force"] is True for call in ingestion_calls)
    assert len(vectors.deleted) == 2
    assert len(removed_documents) == 2


def test_web_sync_second_complete_crawl_is_idempotent(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("web.idempotent", "correct horse battery staple")
    sources = DataSourceService(database)
    source = sources.create("docs", "web", "https://example.com/", "default", admin.user_id)
    document = web_document("https://example.com/", "Same content")
    report = WebCrawlResult((document,), fetched_pages=1, complete=True)
    monkeypatch.setattr(
        "src.libs.loader.web_loader.WebLoader.crawl_with_report",
        lambda _loader, _url: report,
    )
    monkeypatch.setattr("src.core.settings.resolve_path", lambda _value: tmp_path / "snapshots")
    runtime = SimpleNamespace(
        sources=sources,
        jobs=FakeJobs(),
        settings=SimpleNamespace(
            web_crawler=SimpleNamespace(
                max_depth=1, max_pages=10, max_response_mb=1, timeout_seconds=1.0,
            ),
        ),
    )
    worker = Worker(runtime)
    calls = 0

    def fake_ingest(_payload, _job_id):
        nonlocal calls
        calls += 1
        return {"stages": {"loading": {"doc_id": "doc-1"}}, "vector_ids": ["chunk-1"]}

    monkeypatch.setattr(worker, "_ingest_file", fake_ingest)

    first = worker._sync_web(source, "job-1")
    second = worker._sync_web(source, "job-2")

    assert first["processed"] == 1
    assert second["processed"] == 0
    assert second["changes"]["unchanged"] == 1
    assert second["changes"]["added"] == []
    assert second["changes"]["modified"] == []
    assert calls == 1


def test_web_sync_cancellation_does_not_delete_previous_pages(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("web.cancel", "correct horse battery staple")
    sources = DataSourceService(database)
    source = sources.create("docs", "web", "https://example.com/", "default", admin.user_id)
    first_url = "https://example.com/first"
    second_url = "https://example.com/second"
    first_stable = hashlib.sha256(f"{source['id']}:{first_url}".encode()).hexdigest()
    second_stable = hashlib.sha256(f"{source['id']}:{second_url}".encode()).hexdigest()
    sources.save_manifest(source["id"], first_stable, "default", "old-1", "old", ["old-1"])
    sources.save_manifest(source["id"], second_stable, "default", "old-2", "old", ["old-2"])
    jobs = CancellingJobs()
    runtime = SimpleNamespace(
        sources=sources,
        jobs=jobs,
        settings=SimpleNamespace(
            web_crawler=SimpleNamespace(
                max_depth=1, max_pages=10, max_response_mb=1, timeout_seconds=1.0,
            ),
        ),
    )
    worker = Worker(runtime)
    monkeypatch.setattr("src.core.settings.resolve_path", lambda _value: tmp_path / "snapshots")
    monkeypatch.setattr(
        "src.libs.loader.web_loader.WebLoader.crawl_with_report",
        lambda _loader, _url: WebCrawlResult(
            (web_document(first_url, "new"),), fetched_pages=1, complete=True,
        ),
    )

    def cancel_during_ingest(_payload, _job_id):
        jobs.cancelled = True
        raise RuntimeError("Job cancellation requested")

    monkeypatch.setattr(worker, "_ingest_file", cancel_during_ingest)

    try:
        worker._sync_web(source, "job-cancel")
    except RuntimeError as exc:
        assert "cancellation" in str(exc)
    else:
        raise AssertionError("Expected cancellation to abort web synchronization")

    assert {item["stable_id"] for item in sources.list_manifests(source["id"])} == {
        first_stable, second_stable,
    }
