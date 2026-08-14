"""Recoverable SQLite-backed job queue for the single-node worker."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from src.production.database import ProductionDatabase


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    TERMINAL = {"succeeded", "failed", "cancelled"}

    def __init__(self, database: ProductionDatabase) -> None:
        self.database = database

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        created_by: str | None = None,
        resource_key: str | None = None,
        max_attempts: int = 3,
    ) -> str:
        job_id = str(uuid.uuid4())
        try:
            with self.database.transaction(immediate=True) as connection:
                connection.execute(
                    """INSERT INTO jobs(id,kind,status,payload_json,resource_key,max_attempts,created_by,created_at)
                       VALUES (?,?, 'queued',?,?,?,?,?)""",
                    (job_id, kind, json.dumps(payload, ensure_ascii=False), resource_key,
                     max_attempts, created_by, _now()),
                )
        except Exception as exc:
            if "idx_jobs_resource_active" in str(exc) or "UNIQUE constraint" in str(exc):
                raise RuntimeError("An active job already owns this resource") from exc
            raise
        return job_id

    def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        del worker_id  # reserved for multi-worker diagnostics
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                """SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0
                   ORDER BY created_at LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            changed = connection.execute(
                """UPDATE jobs SET status='running', attempts=attempts+1,
                   started_at=COALESCE(started_at,?), heartbeat_at=?
                   WHERE id=? AND status='queued'""",
                (_now(), _now(), row["id"]),
            ).rowcount
            if not changed:
                return None
            updated = connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        return self._decode(updated)

    def heartbeat(self, job_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET heartbeat_at=? WHERE id=? AND status='running'", (_now(), job_id)
            )

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            status = "cancelled" if row["cancel_requested"] else "succeeded"
            connection.execute(
                "UPDATE jobs SET status=?,result_json=?,finished_at=?,heartbeat_at=? WHERE id=?",
                (status, json.dumps(result, ensure_ascii=False, default=str), _now(), _now(), job_id),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT attempts,max_attempts,cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["cancel_requested"]:
                status = "cancelled"
            elif row["attempts"] < row["max_attempts"]:
                status = "queued"
            else:
                status = "failed"
            connection.execute(
                """UPDATE jobs SET status=?,error=?,finished_at=CASE WHEN ? IN ('failed','cancelled') THEN ? ELSE NULL END,
                   heartbeat_at=? WHERE id=?""",
                (status, error[:4000], status, _now(), _now(), job_id),
            )

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] == "queued":
                connection.execute(
                    "UPDATE jobs SET cancel_requested=1,status='cancelled',finished_at=? WHERE id=?",
                    (_now(), job_id),
                )
            elif row["status"] == "running":
                connection.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        return self.get(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def recover_stale(self, stale_after_seconds: int = 120) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
        with self.database.transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT id,attempts,max_attempts,cancel_requested FROM jobs WHERE status='running' AND heartbeat_at<?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                status = "cancelled" if row["cancel_requested"] else (
                    "queued" if row["attempts"] < row["max_attempts"] else "failed"
                )
                connection.execute(
                    "UPDATE jobs SET status=?,error='Worker heartbeat expired',finished_at=? WHERE id=?",
                    (status, _now() if status in self.TERMINAL else None, row["id"]),
                )
        return len(rows)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row)

    @staticmethod
    def _decode(row) -> dict[str, Any]:
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        value["result"] = json.loads(value.pop("result_json")) if value.get("result_json") else None
        value["cancel_requested"] = bool(value["cancel_requested"])
        return value
