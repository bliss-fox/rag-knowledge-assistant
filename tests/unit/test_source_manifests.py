from __future__ import annotations

from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.sources import DataSourceService


def test_source_manifest_round_trip_and_safe_delete(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("manifest.admin", "correct horse battery staple")
    service = DataSourceService(database, [root])
    source = service.create("local", "directory", str(root), "default", admin.user_id)
    service.save_manifest(source["id"], "stable", "default", "doc-1", "hash-1", ["c1", "c2"])
    assert service.get_manifest(source["id"], "stable")["chunk_ids"] == ["c1", "c2"]
    service.delete_manifest(source["id"], "stable")
    assert service.get_manifest(source["id"], "stable") is None


def test_source_index_references_exclude_source_being_deleted(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("references.admin", "correct horse battery staple")
    service = DataSourceService(database, [root])
    first = service.create("first", "directory", str(root), "default", admin.user_id)
    second = service.create("second", "directory", str(root), "default", admin.user_id)
    service.save_manifest(first["id"], "a", "default", "shared-doc", "h1", ["shared", "only-a"])
    service.save_manifest(second["id"], "b", "default", "shared-doc", "h2", ["shared", "only-b"])

    chunks, documents = service.index_references(first["id"], "default")

    assert chunks == {"shared", "only-b"}
    assert documents == {"shared-doc"}
    assert service.list_manifests(first["id"])[0]["chunk_ids"] == ["shared", "only-a"]


def test_source_index_references_can_exclude_only_one_manifest(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("manifest.reference", "correct horse battery staple")
    service = DataSourceService(database, [root])
    source = service.create("web", "web", "https://example.com", "default", admin.user_id)
    service.save_manifest(source["id"], "first", "default", "shared-doc", "h1", ["shared", "one"])
    service.save_manifest(source["id"], "second", "default", "shared-doc", "h2", ["shared", "two"])

    chunks, documents = service.index_references(
        source["id"], "default", stable_id="first",
    )

    assert chunks == {"shared", "two"}
    assert documents == {"shared-doc"}


def test_directory_scan_retries_discovered_and_failed_files(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    root = tmp_path / "documents"
    root.mkdir()
    document = root / "guide.md"
    document.write_text("retry me", encoding="utf-8")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("scan.retry", "correct horse battery staple")
    service = DataSourceService(database, [root])
    source = service.create("local", "directory", str(root), "default", admin.user_id)

    first = service.scan_directory(source["id"], (".md",), 1024)
    second = service.scan_directory(source["id"], (".md",), 1024)
    stable_id = first.added[0]["stable_id"]
    service.mark_status(source["id"], stable_id, "failed")
    third = service.scan_directory(source["id"], (".md",), 1024)
    service.mark_status(source["id"], stable_id, "indexed")
    fourth = service.scan_directory(source["id"], (".md",), 1024)

    assert len(first.added) == 1
    assert len(second.modified) == 1
    assert len(third.modified) == 1
    assert fourth.unchanged == 1
