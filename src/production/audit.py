"""Audit and deployment event recording."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.production.database import ProductionDatabase


class AuditService:
    def __init__(self, database: ProductionDatabase) -> None:
        self.database = database

    def record(self, action: str, user_id: str | None, resource_type: str | None = None,
               resource_id: str | None = None, detail: dict[str, Any] | None = None,
               ip_address: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO audit_events(id,user_id,action,resource_type,resource_id,detail_json,ip_address,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (event_id, user_id, action, resource_type, resource_id,
                 json.dumps(detail or {}, ensure_ascii=False, default=str), ip_address,
                 datetime.now(timezone.utc).isoformat()),
            )
        return event_id

    def deployment(self, event_type: str, version: str, user_id: str | None,
                   detail: dict[str, Any] | None = None) -> str:
        event_id = str(uuid.uuid4())
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO deployment_events(id,event_type,version,detail_json,created_by,created_at) VALUES (?,?,?,?,?,?)",
                (event_id, event_type, version, json.dumps(detail or {}, ensure_ascii=False, default=str),
                 user_id, datetime.now(timezone.utc).isoformat()),
            )
        return event_id

    def deployment_if_changed(self, event_type: str, version: str,
                              detail: dict[str, Any] | None = None) -> str | None:
        """Record a system deployment event only when its version changed."""
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT version FROM deployment_events WHERE event_type=? ORDER BY created_at DESC LIMIT 1",
                (event_type,),
            ).fetchone()
            if row is not None and row["version"] == version:
                return None
            event_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO deployment_events(id,event_type,version,detail_json,created_by,created_at) VALUES (?,?,?,?,?,?)",
                (event_id, event_type, version,
                 json.dumps(detail or {}, ensure_ascii=False, default=str), None,
                 datetime.now(timezone.utc).isoformat()),
            )
        return event_id
