from dataclasses import replace
from types import SimpleNamespace

from src.core.settings import load_settings
from src.production.services import RAGApplicationService, RerankOutcome, SearchPipeline


def test_retrieval_method_marks_failed_reranker_fallback() -> None:
    details = SimpleNamespace(used_fallback=False)
    rerank = RerankOutcome(
        results=[],
        used_fallback=True,
        fallback_reason="reranker_initialization_failed: model unavailable",
        reranker_type="disabled",
    )

    assert RAGApplicationService._method(details, rerank) == "hybrid_rrf_rerank_fallback"


def test_retrieval_method_distinguishes_intentional_rerank_disable() -> None:
    details = SimpleNamespace(used_fallback=False)
    rerank = RerankOutcome(
        results=[],
        used_fallback=False,
        fallback_reason="disabled_by_request",
        reranker_type="disabled",
    )

    assert RAGApplicationService._method(details, rerank) == "bm25+dense+rrf"


def test_pipeline_does_not_construct_reranker_when_request_disables_it(monkeypatch) -> None:
    pipeline = SearchPipeline(hybrid=object(), reranker=None, vector_store=object())
    service = object.__new__(RAGApplicationService)
    service._pipelines = {}
    service.pipeline_factory = lambda _collection: pipeline
    service.settings = object()
    create = monkeypatch.setattr(
        "src.production.services.create_core_reranker",
        lambda _settings: (_ for _ in ()).throw(AssertionError("reranker initialized")),
    )

    result = service._pipeline("docs", initialize_reranker=False)

    assert result is pipeline
    assert result.reranker is None
    assert create is None


def test_rerank_applied_is_false_for_fallback_or_disabled() -> None:
    assert not RAGApplicationService._rerank_applied(
        RerankOutcome([], used_fallback=True, reranker_type="cross_encoder")
    )
    assert not RAGApplicationService._rerank_applied(
        RerankOutcome([], used_fallback=False, reranker_type="disabled")
    )
    assert RAGApplicationService._rerank_applied(
        RerankOutcome([], used_fallback=False, reranker_type="cross_encoder")
    )


def test_answer_refuses_before_retrieval_when_evidence_gate_is_pending() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        evidence=replace(settings.evidence, calibration_status="pending"),
    )

    class ExplodingLLM:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called before EvidenceGate calibration")

    service = RAGApplicationService(
        settings=settings,
        pipeline_factory=lambda _collection: (_ for _ in ()).throw(
            AssertionError("retrieval must not run before EvidenceGate calibration")
        ),
        llm=ExplodingLLM(),
    )

    result = service.answer("尚未校准时能否回答？")

    assert result["status"] == "refused"
    assert result["citations"] == []
    assert result["evidence"]["reason"] == "evidence_gate_uncalibrated"
    assert result["evidence"]["calibration_status"] == "pending"
