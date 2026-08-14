"""Read-only data-source catalog and deterministic directory synchronization."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.production.database import ProductionDatabase


DEFAULT_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "models", "runtime", "temp", "cache", "logs", "qdrant", "backups",
}
DEFAULT_EXCLUDED_FILES = {".env", ".env.local", "secrets.yaml", "credentials.json"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SourceChangeSet:
    added: tuple[dict[str, Any], ...]
    modified: tuple[dict[str, Any], ...]
    deleted: tuple[dict[str, Any], ...]
    unchanged: int

    def to_dict(self) -> dict[str, Any]:
        return {"added": list(self.added), "modified": list(self.modified),
                "deleted": list(self.deleted), "unchanged": self.unchanged}


class DataSourceService:
    def __init__(self, database: ProductionDatabase, allowed_roots: Iterable[str | Path] = ()) -> None:
        self.database = database
        self.allowed_roots = tuple(Path(item).resolve() for item in allowed_roots)

    def create(
        self,
        name: str,
        kind: str,
        location: str,
        collection_name: str,
        created_by: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in {"directory", "web", "upload"}:
            raise ValueError("Unsupported data-source kind")
        if kind == "directory":
            location = str(self._validate_root(location))
        source_id = str(uuid.uuid4())
        timestamp = _now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO data_sources(id,name,kind,location,collection_name,config_json,created_by,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (source_id, name.strip(), kind, location, collection_name,
                 json.dumps(config or {}, ensure_ascii=False), created_by, timestamp, timestamp),
            )
        return self.get(source_id)

    def list(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM data_sources ORDER BY created_at DESC").fetchall()
        return [self._decode(row) for row in rows]

    def get(self, source_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM data_sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(source_id)
        return self._decode(row)

    def update(self, source_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "collection_name", "enabled", "config"}
        if not changes or not set(changes) <= allowed:
            raise ValueError("Unsupported data-source update")
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            column = "config_json" if key == "config" else key
            assignments.append(f"{column}=?")
            values.append(json.dumps(value, ensure_ascii=False) if key == "config" else value)
        values.extend([_now(), source_id])
        with self.database.transaction(immediate=True) as connection:
            if not connection.execute("SELECT 1 FROM data_sources WHERE id=?", (source_id,)).fetchone():
                raise KeyError(source_id)
            connection.execute(
                f"UPDATE data_sources SET {','.join(assignments)},updated_at=? WHERE id=?", values
            )
        return self.get(source_id)

    def delete(self, source_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            if not connection.execute("SELECT 1 FROM data_sources WHERE id=?", (source_id,)).fetchone():
                raise KeyError(source_id)
            connection.execute("DELETE FROM data_sources WHERE id=?", (source_id,))

    def scan_directory(
        self,
        source_id: str,
        allowed_extensions: Iterable[str],
        max_bytes: int,
    ) -> SourceChangeSet:
        source = self.get(source_id)
        if source["kind"] != "directory":
            raise ValueError("Data source is not a directory")
        root = self._validate_root(source["location"])
        extensions = {item.lower() for item in allowed_extensions}
        current: dict[str, dict[str, Any]] = {}
        for path in self._walk(root):
            if path.suffix.lower() not in extensions or path.stat().st_size > max_bytes:
                continue
            relative = path.relative_to(root).as_posix()
            stable_id = hashlib.sha256(f"{source_id}:{relative.lower()}".encode()).hexdigest()
            stat = path.stat()
            current[stable_id] = {
                "stable_id": stable_id, "path": str(path), "relative_path": relative,
                "content_hash": self._hash(path), "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM source_files WHERE source_id=?", (source_id,)).fetchall()
        previous = {row["stable_id"]: dict(row) for row in rows}
        added = [item for key, item in current.items() if key not in previous]
        modified = [
            item for key, item in current.items()
            if key in previous and (
                item["content_hash"] != previous[key]["content_hash"]
                or previous[key]["status"] != "indexed"
            )
        ]
        deleted = [item for key, item in previous.items() if key not in current]
        unchanged = len(current) - len(added) - len(modified)
        timestamp = _now()
        with self.database.transaction(immediate=True) as connection:
            for item in current.values():
                connection.execute(
                    """INSERT INTO source_files(source_id,stable_id,path,content_hash,size_bytes,modified_ns,status,last_seen_at)
                       VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source_id,stable_id) DO UPDATE SET
                       path=excluded.path,content_hash=excluded.content_hash,size_bytes=excluded.size_bytes,
                       modified_ns=excluded.modified_ns,last_seen_at=excluded.last_seen_at""",
                    (source_id, item["stable_id"], item["path"], item["content_hash"],
                     item["size_bytes"], item["modified_ns"], "discovered", timestamp),
                )
            for item in deleted:
                connection.execute(
                    "UPDATE source_files SET status='deleted',last_seen_at=? WHERE source_id=? AND stable_id=?",
                    (timestamp, source_id, item["stable_id"]),
                )
        return SourceChangeSet(tuple(added), tuple(modified), tuple(deleted), unchanged)

    def mark_status(self, source_id: str, stable_id: str, status: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE source_files SET status=? WHERE source_id=? AND stable_id=?",
                (status, source_id, stable_id),
            )

    def save_manifest(
        self,
        source_id: str,
        stable_id: str,
        collection_name: str,
        document_id: str,
        content_hash: str,
        chunk_ids: Iterable[str],
    ) -> None:
        """Persist the exact vector IDs produced by a source file."""
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO indexed_documents(
                   source_id,stable_id,collection_name,document_id,content_hash,chunk_ids_json,indexed_at)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT(source_id,stable_id) DO UPDATE SET
                   collection_name=excluded.collection_name,document_id=excluded.document_id,
                   content_hash=excluded.content_hash,chunk_ids_json=excluded.chunk_ids_json,
                   indexed_at=excluded.indexed_at""",
                (source_id, stable_id, collection_name, document_id, content_hash,
                 json.dumps(list(chunk_ids), ensure_ascii=False), _now()),
            )

    def get_manifest(self, source_id: str, stable_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM indexed_documents WHERE source_id=? AND stable_id=?",
                (source_id, stable_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["chunk_ids"] = json.loads(result.pop("chunk_ids_json"))
        return result

    def list_manifests(self, source_id: str) -> list[dict[str, Any]]:
        """Return exact index ownership records before a source is removed."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM indexed_documents WHERE source_id=? ORDER BY stable_id",
                (source_id,),
            ).fetchall()
        manifests: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["chunk_ids"] = json.loads(value.pop("chunk_ids_json"))
            manifests.append(value)
        return manifests

    def index_references(
        self, source_id: str, collection_name: str, stable_id: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """Return index IDs still owned outside the record being removed.

        Omitting ``stable_id`` excludes the whole source, which is appropriate
        when deleting a data source. Supplying it excludes only that manifest,
        preserving references shared by another page/file in the same source.
        """
        with self.database.connect() as connection:
            if stable_id is None:
                rows = connection.execute(
                    """SELECT document_id,chunk_ids_json FROM indexed_documents
                       WHERE source_id<>? AND collection_name=?""",
                    (source_id, collection_name),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT document_id,chunk_ids_json FROM indexed_documents
                       WHERE collection_name=?
                       AND NOT (source_id=? AND stable_id=?)""",
                    (collection_name, source_id, stable_id),
                ).fetchall()
        documents = {str(row["document_id"]) for row in rows}
        chunks = {
            str(chunk_id)
            for row in rows
            for chunk_id in json.loads(row["chunk_ids_json"])
        }
        return chunks, documents

    def delete_manifest(self, source_id: str, stable_id: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM indexed_documents WHERE source_id=? AND stable_id=?",
                (source_id, stable_id),
            )
            connection.execute(
                "DELETE FROM source_files WHERE source_id=? AND stable_id=?",
                (source_id, stable_id),
            )

    def _validate_root(self, value: str | Path) -> Path:
        root = Path(value).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("Directory data source does not exist")
        if self.allowed_roots and not any(root == item or item in root.parents for item in self.allowed_roots):
            raise PermissionError("Directory is outside configured read-only roots")
        return root

    @staticmethod
    def _walk(root: Path):
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in DEFAULT_EXCLUDED_DIRS and not name.startswith(".")]
            base = Path(directory)
            for name in filenames:
                if name in DEFAULT_EXCLUDED_FILES or name.startswith(".") or name.endswith(("~", ".tmp", ".part")):
                    continue
                path = base / name
                if not path.is_symlink():
                    yield path

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _decode(row) -> dict[str, Any]:
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        result["enabled"] = bool(result["enabled"])
        return result
