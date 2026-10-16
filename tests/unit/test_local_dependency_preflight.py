from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx

from scripts.preflight_local_dependencies import main
from src.core.settings import load_settings
from src.production.local_preflight import LocalDependencyPreflight


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("GET", "http://local.test")

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("failed", request=self.request, response=response)


class FakeHttpClient:
    def __init__(
        self,
        models: list[str] | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.models = models or []
        self.failure = failure

    def __enter__(self) -> FakeHttpClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
        del headers
        if self.failure:
            raise self.failure
        if url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": item} for item in self.models]})
        if url.endswith("/collections"):
            return FakeResponse({"result": {"collections": [{"name": "hidden"}]}})
        return FakeResponse({}, status_code=404)


class FakeEmbedding:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [[0.1] * self.dimensions]


class FakeLLM:
    def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        assert len(messages) == 1
        assert kwargs["max_tokens"] == 8
        return SimpleNamespace(
            content="PRIVATE MODEL RESPONSE",
            model="qwen3:8b",
            usage={"prompt_tokens": 7, "completion_tokens": 1},
        )


class FakeReranker:
    def __init__(self, cache_dir: Any) -> None:
        self.device = "cpu"
        self.cache_dir = cache_dir
        self.warmed = False

    def warmup(self) -> None:
        self.warmed = True

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> Any:
        assert self.warmed
        assert query
        assert top_k == 2
        ranked = sorted(candidates, key=lambda item: item["id"] != "relevant")
        return [{**item, "rerank_score": 1.0 - index} for index, item in enumerate(ranked)]


def _runner(tmp_path, *, dimensions: int = 1024, models: list[str] | None = None):
    settings = load_settings()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    client = FakeHttpClient(models or [settings.llm.model, settings.embedding.model])
    return LocalDependencyPreflight(
        settings,
        http_client_factory=lambda **_kwargs: client,
        embedding_factory=lambda _settings, _timeout: FakeEmbedding(dimensions),
        llm_factory=lambda _settings, _timeout: FakeLLM(),
        reranker_factory=lambda _settings: FakeReranker(cache_dir),
    )


def test_local_dependency_preflight_success_is_sanitized(tmp_path) -> None:
    report = _runner(tmp_path).run()

    assert report["ok"] is True
    assert report["checks"]["embedding"]["actual_dimensions"] == 1024
    assert report["checks"]["llm_generation"]["completion_tokens"] == 1
    assert report["checks"]["qdrant"]["collection_count"] == 1
    assert report["checks"]["cross_encoder"] == {
        "ok": True,
        "elapsed_ms": report["checks"]["cross_encoder"]["elapsed_ms"],
        "error_category": None,
        "device": "cpu",
        "cache_present": True,
        "ranking_probe_passed": True,
    }
    serialized = json.dumps(report, ensure_ascii=False)
    assert "PRIVATE MODEL RESPONSE" not in serialized
    assert "Local dependency health check" not in serialized
    assert "Bananas" not in serialized
    assert '"name": "hidden"' not in serialized


def test_local_dependency_preflight_fails_closed_for_missing_model(tmp_path) -> None:
    settings = load_settings()
    report = _runner(tmp_path, models=[settings.llm.model]).run()

    assert report["ok"] is False
    assert report["checks"]["ollama"]["error_category"] == "model_missing"
    assert report["checks"]["ollama"]["models"]["embedding"]["available"] is False
    assert report["checks"]["embedding"]["error_category"] == "prerequisite_failed"


def test_local_dependency_preflight_fails_closed_for_dimension_mismatch(tmp_path) -> None:
    report = _runner(tmp_path, dimensions=768).run()

    assert report["ok"] is False
    assert report["checks"]["embedding"]["expected_dimensions"] == 1024
    assert report["checks"]["embedding"]["actual_dimensions"] == 768
    assert report["checks"]["embedding"]["error_category"] == "dimension_mismatch"


def test_local_dependency_preflight_redacts_connection_exception(tmp_path) -> None:
    settings = load_settings()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    client = FakeHttpClient(failure=httpx.ConnectError("secret-host-and-token"))
    runner = LocalDependencyPreflight(
        settings,
        http_client_factory=lambda **_kwargs: client,
        embedding_factory=lambda _settings, _timeout: FakeEmbedding(1024),
        llm_factory=lambda _settings, _timeout: FakeLLM(),
        reranker_factory=lambda _settings: FakeReranker(cache_dir),
    )

    report = runner.run()

    assert report["ok"] is False
    assert report["checks"]["ollama"]["error_category"] == "connection_error"
    assert report["checks"]["qdrant"]["error_category"] == "connection_error"
    assert "secret-host-and-token" not in json.dumps(report)


def test_local_dependency_preflight_classifies_cache_permission_failure(tmp_path) -> None:
    settings = load_settings()
    client = FakeHttpClient(models=[settings.llm.model, settings.embedding.model])
    runner = LocalDependencyPreflight(
        settings,
        http_client_factory=lambda **_kwargs: client,
        embedding_factory=lambda _settings, _timeout: FakeEmbedding(1024),
        llm_factory=lambda _settings, _timeout: FakeLLM(),
        reranker_factory=lambda _settings: (_ for _ in ()).throw(
            RuntimeError("PermissionError: secret-cache-path")
        ),
    )

    report = runner.run()

    assert report["checks"]["cross_encoder"]["error_category"] == "cache_permission_error"
    assert "secret-cache-path" not in json.dumps(report)


def test_preflight_cli_returns_nonzero_for_failed_report(capsys) -> None:
    class FailedRunner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            return {"schema_version": 1, "ok": False, "checks": {}}

    exit_code = main([], runner_factory=FailedRunner)

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
