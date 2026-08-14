"""Single application service for search, answers, API, CLI and MCP."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import HybridSearchResult, create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import CoreReranker, create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import Settings, load_settings, resolve_path
from src.core.trace import TraceContext
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.production.evidence import EvidenceGate
from src.production.prompts import PromptRegistry
from src.production.trace_store import SQLiteTraceStore

REFUSAL_TEXT = "现有知识库证据不足，无法可靠回答。"


@dataclass
class SearchPipeline:
    hybrid: Any
    reranker: CoreReranker | None
    vector_store: Any


@dataclass
class RerankOutcome:
    """Normalized rerank result used by every delivery surface."""

    results: list[RetrievalResult]
    used_fallback: bool = False
    fallback_reason: str | None = None
    reranker_type: str = "disabled"


class RAGApplicationService:
    def __init__(
        self,
        settings: Settings | None = None,
        trace_store: SQLiteTraceStore | None = None,
        prompts: PromptRegistry | None = None,
        pipeline_factory: Any | None = None,
        llm: Any | None = None,
        config_version: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.trace_store = trace_store
        self.prompts = prompts or PromptRegistry(resolve_path("config/prompts"))
        self.pipeline_factory = pipeline_factory or self._build_pipeline
        self.llm = llm
        self.config_version = config_version
        evidence = self.settings.evidence
        self.evidence_gate = EvidenceGate(
            evidence.min_top_score, evidence.min_evidence_count, evidence.min_citation_coverage
        )
        self._pipelines: dict[str, SearchPipeline] = {}

    def search(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
        user_id: str | None = None,
        enable_rerank: bool = True,
    ) -> dict[str, Any]:
        self._validate_query(query)
        trace = self._new_trace(query, user_id)
        try:
            pipeline = self._pipeline(collection, initialize_reranker=enable_rerank)
            candidate_k = max(top_k or self.settings.rerank.top_k, self.settings.rerank.candidate_top_k)
            details: HybridSearchResult = pipeline.hybrid.search(
                query=query, top_k=candidate_k, trace=trace, return_details=True
            )
            fused = details.results
            rerank_result = self._rerank(
                pipeline, query, fused, top_k or self.settings.rerank.top_k, trace, enable_rerank
            )
            final = rerank_result.results
            status = "degraded" if details.used_fallback or rerank_result.used_fallback else "answered"
            trace.metadata.update({
                "status": status,
                "retrieval_method": self._method(details, rerank_result),
                "model_version": self.settings.rerank.model,
                "response": "",
            })
            return {
                "status": status,
                "query": query,
                "results": [self._result(item, rank) for rank, item in enumerate(final, 1)],
                "stages": {
                    "dense": [self._result(item, rank) for rank, item in enumerate(details.dense_results or [], 1)],
                    "sparse": [self._result(item, rank) for rank, item in enumerate(details.sparse_results or [], 1)],
                    "fusion": [self._result(item, rank) for rank, item in enumerate(fused, 1)],
                    "rerank": [self._result(item, rank) for rank, item in enumerate(final, 1)],
                },
                "degraded": status == "degraded",
                "degradation_reason": rerank_result.fallback_reason or details.dense_error or details.sparse_error,
                "retrieval_method": self._method(details, rerank_result),
                "rerank_applied": self._rerank_applied(rerank_result),
                "trace_id": trace.trace_id,
            }
        except Exception as exc:
            trace.metadata.update({"status": "error", "error_type": type(exc).__name__, "response": ""})
            raise
        finally:
            self._finish_trace(trace)

    def answer(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_query(query)
        trace = self._new_trace(query, user_id)
        try:
            # This guard belongs in the shared application service, not only in
            # FastAPI readiness.  CLI, MCP and evaluation workers call this
            # service directly and must never generate an answer with an
            # uncalibrated relevance threshold.
            if self.settings.evidence.calibration_status != "calibrated":
                decision = {
                    "accepted": False,
                    "score": 0.0,
                    "reason": "evidence_gate_uncalibrated",
                    "evidence_count": 0,
                    "citation_coverage": None,
                    "calibration_status": self.settings.evidence.calibration_status,
                }
                trace.record_stage("evidence_precheck", decision)
                return self._refused(trace, decision, [])

            pipeline = self._pipeline(collection, initialize_reranker=True)
            candidate_k = max(top_k or self.settings.rerank.top_k, self.settings.rerank.candidate_top_k)
            details: HybridSearchResult = pipeline.hybrid.search(query, top_k=candidate_k, trace=trace, return_details=True)
            rerank = self._rerank(
                pipeline, query, details.results,
                top_k or self.settings.rerank.top_k, trace, enable_rerank=True,
            )
            evidence = rerank.results
            precheck = self.evidence_gate.precheck(evidence)
            trace.record_stage("evidence_precheck", precheck.to_dict())
            if not precheck.accepted:
                return self._refused(trace, precheck.to_dict(), evidence)

            prompt = self.prompts.get("grounded_answer")
            context = self._format_context(evidence)
            rendered = prompt.render(question=query, context=context)
            trace.metadata.update({
                "prompt": rendered, "prompt_id": prompt.prompt_id,
                "prompt_version": prompt.version, "prompt_hash": prompt.sha256,
                "retrieval_method": self._method(details, rerank),
            })
            llm = self.llm or LLMFactory.create(self.settings)
            started = monotonic()
            response = llm.chat([Message(role="user", content=rendered)], trace=trace)
            elapsed = (monotonic() - started) * 1000
            trace.record_stage("generation", {
                "model": response.model, "usage": response.usage or {}, "response": response.content,
            }, elapsed_ms=elapsed)
            postcheck = self.evidence_gate.postcheck(response.content, evidence)
            trace.record_stage("evidence_postcheck", postcheck.to_dict())
            if not postcheck.accepted:
                return self._refused(trace, postcheck.to_dict(), evidence)
            status = "degraded" if details.used_fallback or rerank.used_fallback else "answered"
            citations = [self._citation(item, rank) for rank, item in enumerate(evidence, 1)]
            trace.metadata.update({
                "status": status, "response": response.content,
                "usage": response.usage or {}, "model_version": response.model,
                "citation_coverage": postcheck.citation_coverage,
            })
            return {
                "status": status, "answer": response.content, "citations": citations,
                "evidence": postcheck.to_dict(), "trace_id": trace.trace_id,
                "prompt_version": prompt.version, "model_version": response.model,
            }
        except Exception as exc:
            trace.metadata.update({"status": "error", "error_type": type(exc).__name__, "response": ""})
            return {
                "status": "error", "answer": "系统暂时无法完成回答。", "citations": [],
                "evidence": {"accepted": False, "reason": "system_error"},
                "trace_id": trace.trace_id, "prompt_version": None,
                "model_version": self.settings.llm.model,
            }
        finally:
            self._finish_trace(trace)

    def close(self) -> None:
        for pipeline in self._pipelines.values():
            pipeline.vector_store.close()
        self._pipelines.clear()

    def warmup(self, collection: str = "default") -> dict[str, Any]:
        """Initialize retrieval dependencies and preheat the configured reranker."""
        pipeline = self._pipeline(collection, initialize_reranker=True)
        assert pipeline.reranker is not None
        backend = getattr(pipeline.reranker, "_reranker", None)
        warmup = getattr(backend, "warmup", None)
        if callable(warmup):
            warmup()
        return {
            "collection": collection,
            "reranker": pipeline.reranker.reranker_type,
            "device": getattr(backend, "device", None),
            "reranker_configured": bool(self.settings.rerank.enabled),
            "reranker_ready": pipeline.reranker.is_enabled,
            "reranker_error": pipeline.reranker.initialization_error,
            "evidence_calibration": self.settings.evidence.calibration_status,
        }

    def _build_pipeline(self, collection: str) -> SearchPipeline:
        vector_store = VectorStoreFactory.create(self.settings, collection_name=collection)
        embedding = EmbeddingFactory.create(self.settings)
        dense = create_dense_retriever(self.settings, embedding_client=embedding, vector_store=vector_store)
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse = create_sparse_retriever(self.settings, bm25_indexer=bm25, vector_store=vector_store)
        sparse.default_collection = collection
        hybrid = create_hybrid_search(self.settings, QueryProcessor(), dense, sparse)
        return SearchPipeline(hybrid, None, vector_store)

    def _pipeline(
        self, collection: str, *, initialize_reranker: bool = True,
    ) -> SearchPipeline:
        if collection not in self._pipelines:
            self._pipelines[collection] = self.pipeline_factory(collection)
        pipeline = self._pipelines[collection]
        if initialize_reranker and pipeline.reranker is None:
            pipeline.reranker = create_core_reranker(self.settings)
        return pipeline

    @staticmethod
    def _rerank(
        pipeline: SearchPipeline,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
        trace: TraceContext,
        enabled: bool,
    ) -> Any:
        if not enabled or pipeline.reranker is None or not pipeline.reranker.is_enabled:
            if not enabled:
                reason = "disabled_by_request"
                used_fallback = False
            elif pipeline.reranker is not None and pipeline.reranker.initialization_error:
                reason = f"reranker_initialization_failed: {pipeline.reranker.initialization_error}"
                used_fallback = True
            else:
                reason = "disabled_by_configuration"
                used_fallback = False
            trace.record_stage("rerank", {
                "used_fallback": used_fallback,
                "fallback_reason": reason,
                "input_order": [item.chunk_id for item in results],
                "output_order": [item.chunk_id for item in results[:top_k]],
            })
            return RerankOutcome(
                results=results[:top_k],
                used_fallback=used_fallback,
                fallback_reason=reason,
            )
        return pipeline.reranker.rerank(query=query, results=results, top_k=top_k, trace=trace)

    def _new_trace(self, query: str, user_id: str | None) -> TraceContext:
        trace = TraceContext(trace_type="query")
        trace.metadata.update({
            "query": query, "user_id": user_id, "status": "running",
            "config_version": self.config_version,
            "dataset_version": self.settings.vector_store.index_version,
            "model_version": self.settings.llm.model,
        })
        return trace

    def _finish_trace(self, trace: TraceContext) -> None:
        trace.finish()
        if self.trace_store:
            self.trace_store.save(trace.to_dict())

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > self.settings.server.max_query_length:
            raise ValueError("Query exceeds configured length limit")

    def _refused(self, trace: TraceContext, decision: dict[str, object], evidence: list[RetrievalResult]) -> dict[str, Any]:
        trace.metadata.update({"status": "refused", "response": REFUSAL_TEXT,
                               "citation_coverage": decision.get("citation_coverage")})
        prompt = self.prompts.get("grounded_answer")
        return {
            "status": "refused", "answer": REFUSAL_TEXT,
            "citations": [self._citation(item, rank) for rank, item in enumerate(evidence, 1)],
            "evidence": decision, "trace_id": trace.trace_id,
            "prompt_version": prompt.version, "model_version": self.settings.llm.model,
        }

    @staticmethod
    def _format_context(results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[{index}] chunk_id={item.chunk_id} source={item.metadata.get('source_uri') or item.metadata.get('source_path','unknown')}\n{item.text}"
            for index, item in enumerate(results, 1)
        )

    @staticmethod
    def _result(item: RetrievalResult, rank: int) -> dict[str, Any]:
        return {"rank": rank, "chunk_id": item.chunk_id, "score": round(float(item.score), 6),
                "text": item.text, "metadata": item.metadata}

    @staticmethod
    def _citation(item: RetrievalResult, rank: int) -> dict[str, Any]:
        metadata = item.metadata
        return {
            "index": rank, "document_id": metadata.get("document_id") or metadata.get("source_ref") or metadata.get("doc_hash"),
            "chunk_id": item.chunk_id, "source": metadata.get("source_uri") or metadata.get("source_path", "unknown"),
            "page": metadata.get("page") or metadata.get("page_num"), "excerpt": item.text[:500],
            "score": round(float(item.score), 6),
        }

    @staticmethod
    def _method(details: HybridSearchResult, rerank: Any) -> str:
        if rerank.used_fallback:
            return "hybrid_rrf_rerank_fallback"
        if rerank.reranker_type in {"disabled", "none"}:
            return "bm25+dense+rrf"
        if details.used_fallback:
            return "single_route_fallback+rerank"
        return f"bm25+dense+rrf+{rerank.reranker_type}"

    @staticmethod
    def _rerank_applied(rerank: RerankOutcome) -> bool:
        return not rerank.used_fallback and rerank.reranker_type not in {"disabled", "none"}
