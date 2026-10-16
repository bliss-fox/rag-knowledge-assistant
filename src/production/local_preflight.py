"""Non-business-data preflight for local production RAG dependencies.

The report deliberately contains only configuration identifiers, readiness
booleans, dimensions, token counts, devices, timings and stable error
categories.  Probe prompts, model response text, collection names and local
knowledge-base content are never returned.  The check never mutates an index,
database or knowledge source; first-time model loading may populate the
configured model cache.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from src.core.settings import Settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.reranker_factory import RerankerFactory

_EMBEDDING_PROBE = "Local dependency health check. 本地依赖健康检查。"
_LLM_PROBE = "Reply with OK only. /no_think"
_RERANK_QUERY = "哪个检索方案同时使用 BM25 关键词和向量语义搜索？"
_RERANK_CANDIDATES = [
    {
        "id": "relevant",
        "text": "混合检索同时使用 BM25 关键词搜索和稠密向量语义搜索。",
        "score": 0.0,
        "metadata": {},
    },
    {
        "id": "irrelevant",
        "text": "Bananas are commonly yellow when ripe.",
        "score": 0.0,
        "metadata": {},
    },
]


class LocalDependencyPreflight:
    """Exercise local dependencies without initializing or mutating an index."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client_factory: Callable[..., Any] | None = None,
        embedding_factory: Callable[[Settings, float], Any] | None = None,
        llm_factory: Callable[[Settings, float], Any] | None = None,
        reranker_factory: Callable[[Settings], Any] | None = None,
        http_timeout: float = 5.0,
        model_timeout: float = 180.0,
    ) -> None:
        if http_timeout <= 0 or model_timeout <= 0:
            raise ValueError("Preflight timeouts must be positive")
        self.settings = settings
        self.http_client_factory = http_client_factory or httpx.Client
        self.embedding_factory = embedding_factory or self._create_embedding
        self.llm_factory = llm_factory or self._create_llm
        self.reranker_factory = reranker_factory or RerankerFactory.create
        self.http_timeout = float(http_timeout)
        self.model_timeout = float(model_timeout)

    def run(self) -> dict[str, Any]:
        """Run every production dependency check and return a JSON-safe report."""
        ollama = self._check_ollama_inventory()
        qdrant = self._check_qdrant()

        model_status = ollama.get("models", {})
        if ollama["ok"] and model_status.get("embedding", {}).get("available"):
            embedding = self._check_embedding()
        else:
            embedding = self._blocked("prerequisite_failed")

        if ollama["ok"] and model_status.get("llm", {}).get("available"):
            llm = self._check_llm()
        else:
            llm = self._blocked("prerequisite_failed")

        reranker = self._check_reranker()
        checks = {
            "ollama": ollama,
            "embedding": embedding,
            "llm_generation": llm,
            "qdrant": qdrant,
            "cross_encoder": reranker,
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": all(bool(check["ok"]) for check in checks.values()),
            "configured": {
                "llm_provider": self.settings.llm.provider,
                "llm_model": self.settings.llm.model,
                "embedding_provider": self.settings.embedding.provider,
                "embedding_model": self.settings.embedding.model,
                "embedding_dimensions": self.settings.embedding.dimensions,
                "vector_store_provider": self.settings.vector_store.provider,
                "reranker_provider": self.settings.rerank.provider,
                "reranker_model": self.settings.rerank.model,
            },
            "checks": checks,
        }

    def _check_ollama_inventory(self) -> dict[str, Any]:
        started = perf_counter()
        roles = {
            "llm": {
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "base_url": self.settings.llm.base_url or "http://localhost:11434",
            },
            "embedding": {
                "provider": self.settings.embedding.provider,
                "model": self.settings.embedding.model,
                "base_url": self.settings.embedding.base_url or "http://localhost:11434",
            },
        }
        status = {
            role: {"model": item["model"], "available": False}
            for role, item in roles.items()
        }
        if any(item["provider"].lower() != "ollama" for item in roles.values()):
            return self._finish(
                started,
                ok=False,
                error_category="provider_not_local",
                models=status,
                endpoints_ready=0,
                endpoints_total=0,
            )

        endpoint_models: dict[str, set[str]] = {}
        endpoints = {str(item["base_url"]).rstrip("/") for item in roles.values()}
        ready = 0
        error_category: str | None = None
        for endpoint in endpoints:
            try:
                with self.http_client_factory(timeout=self.http_timeout) as client:
                    response = client.get(f"{endpoint}/api/tags")
                    response.raise_for_status()
                    payload = response.json()
                names = {
                    str(value)
                    for model in payload.get("models", [])
                    if isinstance(model, dict)
                    for value in (model.get("name"), model.get("model"))
                    if value
                }
                endpoint_models[endpoint] = names
                ready += 1
            except Exception as exc:  # report is intentionally sanitized
                error_category = error_category or self._categorize(exc)

        for role, item in roles.items():
            endpoint = str(item["base_url"]).rstrip("/")
            status[role]["available"] = item["model"] in endpoint_models.get(endpoint, set())

        all_available = all(bool(item["available"]) for item in status.values())
        if ready == len(endpoints) and not all_available:
            error_category = "model_missing"
        return self._finish(
            started,
            ok=ready == len(endpoints) and all_available,
            error_category=error_category,
            models=status,
            endpoints_ready=ready,
            endpoints_total=len(endpoints),
        )

    def _check_embedding(self) -> dict[str, Any]:
        started = perf_counter()
        expected = int(self.settings.embedding.dimensions)
        try:
            provider = self.embedding_factory(self.settings, self.model_timeout)
            vectors = provider.embed([_EMBEDDING_PROBE])
            vector = vectors[0] if len(vectors) == 1 else []
            actual = len(vector)
            numeric = all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in vector
            )
            ok = actual == expected and numeric
            return self._finish(
                started,
                ok=ok,
                error_category=None if ok else "dimension_mismatch",
                expected_dimensions=expected,
                actual_dimensions=actual,
            )
        except Exception as exc:
            return self._finish(
                started,
                ok=False,
                error_category=self._categorize(exc),
                expected_dimensions=expected,
                actual_dimensions=None,
            )

    def _check_llm(self) -> dict[str, Any]:
        started = perf_counter()
        try:
            provider = self.llm_factory(self.settings, self.model_timeout)
            response = provider.chat(
                [Message(role="user", content=_LLM_PROBE)],
                temperature=0.0,
                max_tokens=8,
            )
            usage = response.usage if isinstance(response.usage, dict) else {}
            model = str(response.model or self.settings.llm.model)
            return self._finish(
                started,
                ok=bool(model),
                error_category=None if model else "invalid_response",
                model=model,
                prompt_tokens=self._safe_count(usage.get("prompt_tokens")),
                completion_tokens=self._safe_count(usage.get("completion_tokens")),
            )
        except Exception as exc:
            return self._finish(
                started,
                ok=False,
                error_category=self._categorize(exc),
                model=self.settings.llm.model,
                prompt_tokens=None,
                completion_tokens=None,
            )

    def _check_qdrant(self) -> dict[str, Any]:
        started = perf_counter()
        if self.settings.vector_store.provider.lower() != "qdrant":
            return self._finish(
                started,
                ok=False,
                error_category="provider_not_qdrant",
                collection_count=None,
            )
        headers = {}
        if self.settings.vector_store.api_key:
            headers["api-key"] = self.settings.vector_store.api_key
        try:
            endpoint = self.settings.vector_store.url.rstrip("/")
            with self.http_client_factory(timeout=self.http_timeout) as client:
                response = client.get(f"{endpoint}/collections", headers=headers)
                response.raise_for_status()
                payload = response.json()
            collections = payload.get("result", {}).get("collections", [])
            if not isinstance(collections, list):
                raise ValueError("Invalid Qdrant collections response")
            return self._finish(
                started,
                ok=True,
                error_category=None,
                collection_count=len(collections),
            )
        except Exception as exc:
            return self._finish(
                started,
                ok=False,
                error_category=self._categorize(exc),
                collection_count=None,
            )

    def _check_reranker(self) -> dict[str, Any]:
        started = perf_counter()
        if not self.settings.rerank.enabled:
            return self._finish(
                started,
                ok=False,
                error_category="disabled",
                device=None,
                cache_present=False,
                ranking_probe_passed=False,
            )
        if self.settings.rerank.provider.lower() != "cross_encoder":
            return self._finish(
                started,
                ok=False,
                error_category="provider_not_cross_encoder",
                device=None,
                cache_present=False,
                ranking_probe_passed=False,
            )
        try:
            reranker = self.reranker_factory(self.settings)
            warmup = getattr(reranker, "warmup", None)
            if not callable(warmup):
                raise TypeError("Configured reranker does not support warmup")
            warmup()
            ranked = reranker.rerank(
                query=_RERANK_QUERY,
                candidates=[dict(item) for item in _RERANK_CANDIDATES],
                top_k=2,
            )
            scores_valid = len(ranked) == len(_RERANK_CANDIDATES) and all(
                isinstance(item.get("rerank_score"), (int, float))
                and not isinstance(item.get("rerank_score"), bool)
                and math.isfinite(float(item["rerank_score"]))
                for item in ranked
            )
            probe_passed = (
                scores_valid
                and ranked[0].get("id") == "relevant"
                and {item.get("id") for item in ranked} == {"relevant", "irrelevant"}
            )
            cache_dir = getattr(reranker, "cache_dir", None)
            cache_present = isinstance(cache_dir, Path) and cache_dir.exists()
            return self._finish(
                started,
                ok=probe_passed,
                error_category=None if probe_passed else "quality_probe_failed",
                device=str(getattr(reranker, "device", "unknown")),
                cache_present=cache_present,
                ranking_probe_passed=probe_passed,
            )
        except Exception as exc:
            return self._finish(
                started,
                ok=False,
                error_category=self._categorize(exc),
                device=None,
                cache_present=False,
                ranking_probe_passed=False,
            )

    @staticmethod
    def _create_embedding(settings: Settings, timeout: float) -> Any:
        return EmbeddingFactory.create(settings, timeout=timeout)

    @staticmethod
    def _create_llm(settings: Settings, timeout: float) -> Any:
        return LLMFactory.create(settings, timeout=timeout)

    @staticmethod
    def _safe_count(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _blocked(category: str) -> dict[str, Any]:
        return {"ok": False, "elapsed_ms": 0.0, "error_category": category}

    @staticmethod
    def _finish(
        started: float,
        *,
        ok: bool,
        error_category: str | None,
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "ok": bool(ok),
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error_category": error_category,
            **details,
        }

    @staticmethod
    def _categorize(exc: Exception) -> str:
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return "timeout"
        if isinstance(exc, (httpx.ConnectError, ConnectionError)):
            return "connection_error"
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            return "dependency_missing"
        if isinstance(exc, PermissionError):
            return "cache_permission_error"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in {401, 403}:
                return "authentication_error"
            if status == 404:
                return "endpoint_not_found"
            return "service_error"
        message = str(exc).lower()
        if "permissionerror" in message or "permission denied" in message:
            return "cache_permission_error"
        if "timed out" in message or "timeout" in message:
            return "timeout"
        if "model" in message and "not found" in message:
            return "model_missing"
        if "connect" in message or "refused" in message:
            return "connection_error"
        if isinstance(exc, (TypeError, ValueError, AttributeError)):
            return "invalid_response"
        return "inference_error"


def run_local_dependency_preflight(settings: Settings, **kwargs: Any) -> dict[str, Any]:
    """Convenience entry point shared by the CLI and future admin surfaces."""
    return LocalDependencyPreflight(settings, **kwargs).run()
