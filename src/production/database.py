"""SQLite WAL operational store and schema migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.production.review_attestation import REVIEW_ATTESTATION_FIELDS

SCHEMA_VERSION = 5


class ProductionDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','user')),
                active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                refresh_hash TEXT NOT NULL UNIQUE, expires_at TEXT NOT NULL,
                revoked_at TEXT, created_at TEXT NOT NULL, last_used_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS data_sources (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
                location TEXT NOT NULL, collection_name TEXT NOT NULL DEFAULT 'default',
                config_json TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1,
                created_by TEXT REFERENCES users(id), created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS source_files (
                source_id TEXT NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
                stable_id TEXT NOT NULL, path TEXT NOT NULL, content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, modified_ns INTEGER NOT NULL, status TEXT NOT NULL,
                last_seen_at TEXT NOT NULL, PRIMARY KEY(source_id, stable_id)
            )""",
            """CREATE TABLE IF NOT EXISTS indexed_documents (
                source_id TEXT NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
                stable_id TEXT NOT NULL, collection_name TEXT NOT NULL,
                document_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                chunk_ids_json TEXT NOT NULL, indexed_at TEXT NOT NULL,
                PRIMARY KEY(source_id, stable_id)
            )""",
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                payload_json TEXT NOT NULL, result_json TEXT, error TEXT,
                resource_key TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3, cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_by TEXT REFERENCES users(id), created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, heartbeat_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, created_at)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_resource_active ON jobs(resource_key) WHERE status IN ('queued','running') AND resource_key IS NOT NULL",
            """CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY, trace_type TEXT NOT NULL, user_id TEXT REFERENCES users(id),
                status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                total_elapsed_ms REAL NOT NULL DEFAULT 0, query_hash TEXT, query_text TEXT,
                response_text TEXT, prompt_text TEXT, prompt_id TEXT, prompt_version TEXT,
                prompt_hash TEXT, model_version TEXT, dataset_version TEXT, config_version TEXT,
                retrieval_method TEXT, input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0, citation_coverage REAL,
                faithfulness REAL, error_type TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
            )""",
            "CREATE INDEX IF NOT EXISTS idx_traces_time ON traces(started_at)",
            "CREATE INDEX IF NOT EXISTS idx_traces_user ON traces(user_id, started_at)",
            """CREATE TABLE IF NOT EXISTS trace_stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, stage TEXT NOT NULL, elapsed_ms REAL,
                data_json TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(trace_id, ordinal)
            )""",
            """CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id), action TEXT NOT NULL,
                resource_type TEXT, resource_id TEXT, detail_json TEXT NOT NULL DEFAULT '{}',
                ip_address TEXT, created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS deployment_events (
                id TEXT PRIMARY KEY, event_type TEXT NOT NULL, version TEXT NOT NULL,
                detail_json TEXT NOT NULL DEFAULT '{}', created_by TEXT REFERENCES users(id),
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY, dataset_name TEXT NOT NULL, dataset_version TEXT NOT NULL,
                split TEXT NOT NULL, status TEXT NOT NULL, metrics_json TEXT,
                config_json TEXT NOT NULL, report_path TEXT, created_by TEXT REFERENCES users(id),
                created_at TEXT NOT NULL, finished_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS golden_reviews (
                dataset_sha256 TEXT NOT NULL, query_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
                notes TEXT NOT NULL DEFAULT '', reviewer_id TEXT REFERENCES users(id),
                reviewed_at TEXT NOT NULL,
                query_verified INTEGER NOT NULL DEFAULT 0 CHECK(query_verified IN (0,1)),
                answer_verified INTEGER NOT NULL DEFAULT 0 CHECK(answer_verified IN (0,1)),
                evidence_verified INTEGER NOT NULL DEFAULT 0 CHECK(evidence_verified IN (0,1)),
                category_difficulty_verified INTEGER NOT NULL DEFAULT 0
                    CHECK(category_difficulty_verified IN (0,1)),
                split_verified INTEGER NOT NULL DEFAULT 0 CHECK(split_verified IN (0,1)),
                PRIMARY KEY(dataset_sha256, query_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_golden_reviews_status ON golden_reviews(dataset_sha256, status)",
            """CREATE TABLE IF NOT EXISTS private_golden_sets (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('draft','reviewed')) DEFAULT 'draft',
                created_by TEXT NOT NULL REFERENCES users(id), created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS private_golden_cases (
                set_id TEXT NOT NULL REFERENCES private_golden_sets(id) ON DELETE CASCADE,
                query_id TEXT NOT NULL, query_text TEXT NOT NULL, category TEXT NOT NULL,
                expected_document_ids_json TEXT NOT NULL, answer_key_points_json TEXT NOT NULL,
                answerable INTEGER NOT NULL CHECK(answerable IN (0,1)), source TEXT NOT NULL,
                split TEXT NOT NULL CHECK(split IN ('dev','final')), language TEXT NOT NULL,
                difficulty TEXT NOT NULL CHECK(difficulty IN ('easy','medium','hard')),
                review_status TEXT NOT NULL CHECK(review_status IN ('pending','approved','rejected')),
                review_notes TEXT NOT NULL DEFAULT '', reviewer_id TEXT REFERENCES users(id),
                reviewed_at TEXT,
                query_verified INTEGER NOT NULL DEFAULT 0 CHECK(query_verified IN (0,1)),
                answer_verified INTEGER NOT NULL DEFAULT 0 CHECK(answer_verified IN (0,1)),
                evidence_verified INTEGER NOT NULL DEFAULT 0 CHECK(evidence_verified IN (0,1)),
                category_difficulty_verified INTEGER NOT NULL DEFAULT 0
                    CHECK(category_difficulty_verified IN (0,1)),
                split_verified INTEGER NOT NULL DEFAULT 0 CHECK(split_verified IN (0,1)),
                created_by TEXT NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(set_id, query_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_private_golden_cases_review ON private_golden_cases(set_id, review_status)",
        ]
        with self.transaction(immediate=True) as connection:
            for statement in statements:
                connection.execute(statement)
            previous_version = connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
            self._ensure_attestation_columns(connection, "golden_reviews")
            self._ensure_attestation_columns(connection, "private_golden_cases")
            if previous_version is not None and int(previous_version) < SCHEMA_VERSION:
                incomplete = " OR ".join(f"{field}=0" for field in REVIEW_ATTESTATION_FIELDS)
                connection.execute(
                    f"""UPDATE golden_reviews SET status='pending'
                        WHERE status='approved' AND ({incomplete})"""
                )
                connection.execute(
                    f"""UPDATE private_golden_cases SET review_status='pending'
                        WHERE review_status='approved' AND ({incomplete})"""
                )
                connection.execute(
                    "UPDATE private_golden_sets SET status='draft' WHERE status='reviewed'"
                )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,)
            )

    @staticmethod
    def _ensure_attestation_columns(connection: sqlite3.Connection, table: str) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for field in REVIEW_ATTESTATION_FIELDS:
            if field not in existing:
                connection.execute(
                    f"""ALTER TABLE {table} ADD COLUMN {field} INTEGER NOT NULL DEFAULT 0
                        CHECK({field} IN (0,1))"""
                )

    def health(self) -> dict[str, object]:
        with self.connect() as connection:
            version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        return {"ok": version == SCHEMA_VERSION, "schema_version": version, "journal_mode": mode}
