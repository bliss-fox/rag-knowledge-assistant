"""Lazy production dependency container."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.settings import Settings, load_settings, resolve_path
from src.production.audit import AuditService
from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.evidence_calibration import validate_calibration_artifact
from src.production.jobs import JobQueue
from src.production.prompts import PromptRegistry
from src.production.services import RAGApplicationService
from src.production.sources import DataSourceService
from src.production.trace_store import SQLiteTraceStore


@dataclass
class Runtime:
    settings: Settings
    database: ProductionDatabase
    auth: AuthService
    jobs: JobQueue
    traces: SQLiteTraceStore
    sources: DataSourceService
    audit: AuditService
    prompts: PromptRegistry
    rag: RAGApplicationService

    @classmethod
    def create(cls, settings: Settings | None = None) -> "Runtime":
        settings = settings or load_settings()
        calibration = validate_calibration_artifact(settings)
        database = ProductionDatabase(resolve_path(settings.observability.sqlite_path))
        secret = _load_or_create_secret(resolve_path("data/auth_secret"))
        auth = AuthService(
            database, secret=secret, access_minutes=settings.server.access_token_minutes,
            refresh_days=settings.server.refresh_token_days,
        )
        traces = SQLiteTraceStore(database, settings.observability.content_retention_days)
        prompts = PromptRegistry(resolve_path("config/prompts"))
        config_version = hashlib.sha256(
            json.dumps(asdict(settings), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        rag = RAGApplicationService(
            settings, trace_store=traces, prompts=prompts, config_version=config_version,
        )
        allowed_roots = [Path(r"D:\AI-KnowledgeBase\documents"), resolve_path("data/uploads")]
        audit = AuditService(database)
        audit.deployment_if_changed("config", config_version, {"source": "config/settings.yaml"})
        audit.deployment_if_changed("model", settings.llm.model, {
            "provider": settings.llm.provider, "embedding": settings.embedding.model,
            "reranker": settings.rerank.model,
        })
        audit.deployment_if_changed("index", settings.vector_store.index_version, {
            "prefix": settings.vector_store.collection_prefix,
        })
        if calibration is not None:
            calibration_path = resolve_path(settings.evidence.calibration_artifact or "")
            calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            audit.deployment_if_changed("evidence_calibration", calibration_sha256, {
                "artifact": str(calibration_path),
                "thresholds": calibration["thresholds"],
                "datasets": calibration["datasets"],
            })
        prompt_manifest = prompts.manifest()
        prompt_version = hashlib.sha256(
            json.dumps(prompt_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        audit.deployment_if_changed("prompts", prompt_version, {"prompts": prompt_manifest})
        return cls(settings, database, auth, JobQueue(database), traces,
                   DataSourceService(database, allowed_roots), audit, prompts, rag)

    def close(self) -> None:
        self.rag.close()


def _load_or_create_secret(path: Path) -> str:
    env = os.environ.get("RAG_AUTH_SECRET")
    if env:
        return env
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= 32:
            return value
    value = secrets.token_urlsafe(64)
    path.write_text(value, encoding="utf-8")
    return value
