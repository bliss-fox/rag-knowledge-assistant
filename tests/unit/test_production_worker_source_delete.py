from __future__ import annotations

from types import SimpleNamespace

from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.sources import DataSourceService
from src.production.worker import Worker


class FakeJobs:
    def is_cancel_requested(self, _job_id):
        return False

    def heartbeat(self, _job_id):
        return None


class FakeVectorStore:
    def __init__(self):
        self.deleted = []

    def delete(self, ids):
        self.deleted.extend(ids)


def test_delete_source_cleans_exclusive_vectors_before_catalog_row(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("delete.admin", "correct horse battery staple")
    sources = DataSourceService(database, [root])
    target = sources.create("target", "directory", str(root), "default", admin.user_id)
    other = sources.create("other", "directory", str(root), "default", admin.user_id)
    sources.save_manifest(target["id"], "a", "default", "shared-doc", "h1", ["shared", "only-target"])
    sources.save_manifest(other["id"], "b", "default", "shared-doc", "h2", ["shared", "only-other"])
    vectors = FakeVectorStore()
    rag = SimpleNamespace(_pipeline=lambda _collection: SimpleNamespace(vector_store=vectors))
    removed_documents = []

    class FakeBM25:
        def __init__(self, index_dir):
            self.index_dir = index_dir

        def remove_document(self, document_id, collection):
            removed_documents.append((document_id, collection))

    monkeypatch.setattr("src.ingestion.storage.bm25_indexer.BM25Indexer", FakeBM25)
    runtime = SimpleNamespace(sources=sources, jobs=FakeJobs(), rag=rag)

    result = Worker(runtime)._delete_source({"source_id": target["id"]}, "job-1")

    assert vectors.deleted == ["only-target"]
    assert removed_documents == []
    assert result["deleted_chunks"] == 1
    assert result["deleted_documents"] == 0
    assert [item["id"] for item in sources.list()] == [other["id"]]


def test_delete_source_removes_unshared_bm25_document(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("delete.unique", "correct horse battery staple")
    sources = DataSourceService(database, [root])
    target = sources.create("target", "directory", str(root), "default", admin.user_id)
    sources.save_manifest(target["id"], "a", "default", "doc-a", "h1", ["a1", "a2"])
    vectors = FakeVectorStore()
    removed_documents = []

    class FakeBM25:
        def __init__(self, index_dir):
            self.index_dir = index_dir

        def remove_document(self, document_id, collection):
            removed_documents.append((document_id, collection))

    monkeypatch.setattr("src.ingestion.storage.bm25_indexer.BM25Indexer", FakeBM25)
    runtime = SimpleNamespace(
        sources=sources,
        jobs=FakeJobs(),
        rag=SimpleNamespace(_pipeline=lambda _collection: SimpleNamespace(vector_store=vectors)),
    )

    result = Worker(runtime)._delete_source({"source_id": target["id"]}, "job-2")

    assert vectors.deleted == ["a1", "a2"]
    assert removed_documents == [("doc-a", "default")]
    assert result["deleted_documents"] == 1
    assert result["deleted_snapshots"] == 0
    assert sources.list() == []


def test_delete_web_source_removes_only_its_owned_snapshots(tmp_path, monkeypatch):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("delete.web", "correct horse battery staple")
    sources = DataSourceService(database)
    target = sources.create(
        "target", "web", "https://example.com", "default", admin.user_id,
    )
    other = sources.create(
        "other", "web", "https://example.org", "default", admin.user_id,
    )
    snapshots_root = tmp_path / "web"
    target_root = snapshots_root / target["id"]
    other_root = snapshots_root / other["id"]
    target_root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    (target_root / "one.md").write_text("one", encoding="utf-8")
    (target_root / "two.md").write_text("two", encoding="utf-8")
    (other_root / "keep.md").write_text("keep", encoding="utf-8")
    monkeypatch.setattr("src.core.settings.resolve_path", lambda _value: snapshots_root)
    runtime = SimpleNamespace(
        sources=sources,
        jobs=FakeJobs(),
        rag=SimpleNamespace(_pipeline=lambda _collection: None),
    )

    result = Worker(runtime)._delete_source({"source_id": target["id"]}, "job-web")

    assert result["deleted_snapshots"] == 2
    assert not target_root.exists()
    assert (other_root / "keep.md").read_text(encoding="utf-8") == "keep"
    assert [item["id"] for item in sources.list()] == [other["id"]]
