#!/usr/bin/env python
"""Stage K: Comprehensive retrieval benchmark across 4 modes.

Evaluates Dense / Sparse / Hybrid / Hybrid+Rerank on 21 golden QA pairs and
produces Hit@1, Hit@5, MRR@10 metrics for each mode, then writes a comparison
JSON to data/eval_results/.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --collection benchmark --top-k 10
    python scripts/run_benchmark.py --skip-ingest
    python scripts/run_benchmark.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def configure_windows_console() -> None:
    """Use UTF-8 for CLI output without mutating streams when imported in tests."""
    if sys.platform != "win32":
        return
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Source-level evaluator (no exact chunk-id needed)
# ---------------------------------------------------------------------------


class SourceMatchEvaluator:
    """Evaluates retrieval by checking source-file name matching.

    Computes Hit@1, Hit@5, Hit@10, MRR@10 based on whether any
    retrieved chunk's source_path contains one of the expected source names.
    """

    def evaluate(
        self,
        retrieved_chunks: list[Any],
        expected_sources: list[str],
        top_k: int = 10,
    ) -> dict[str, float]:
        if not expected_sources:
            return {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr@10": 0.0}

        hits = [self._is_relevant(c, expected_sources) for c in retrieved_chunks[:top_k]]

        def hit_at(k: int) -> float:
            return 1.0 if any(hits[:k]) else 0.0

        def mrr_at(k: int) -> float:
            for rank, h in enumerate(hits[:k], start=1):
                if h:
                    return 1.0 / rank
            return 0.0

        return {
            "hit@1": hit_at(1),
            "hit@5": hit_at(5),
            "hit@10": hit_at(10),
            "mrr@10": mrr_at(10),
        }

    def _is_relevant(self, chunk: Any, expected_sources: list[str]) -> bool:
        source = self._get_source(chunk)
        return any(exp in source for exp in expected_sources)

    def _get_source(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(
                chunk.get(
                    "source_path",
                    chunk.get("source", chunk.get("metadata", {}).get("source_path", "")),
                )
            )
        meta = getattr(chunk, "metadata", {}) or {}
        return str(meta.get("source_path", meta.get("source", getattr(chunk, "source_path", ""))))


# ---------------------------------------------------------------------------
# Test-set loader
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    id: str
    query: str
    expected_sources: list[str]
    reference_answer: str = ""
    category: str = "factual"
    language: str = "en"


def validate_test_cases(cases: list[TestCase]) -> None:
    """Reject malformed benchmark inputs before any metric is computed."""
    if not cases:
        raise ValueError("Benchmark test set contains no cases")
    seen: set[str] = set()
    for case in cases:
        if not case.id or case.id in seen:
            raise ValueError(f"Missing or duplicate benchmark case id: {case.id!r}")
        seen.add(case.id)
        if not case.query.strip():
            raise ValueError(f"Benchmark case {case.id} has an empty query")
        if "\ufffd" in case.query or "\ufffd" in case.reference_answer:
            raise ValueError(f"Benchmark case {case.id} contains replacement characters")
        if not case.expected_sources:
            raise ValueError(f"Benchmark case {case.id} has no expected source")


def load_test_cases(path: Path) -> list[TestCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for tc in data.get("test_cases", []):
        cases.append(
            TestCase(
                id=tc.get("id", ""),
                query=tc["query"],
                expected_sources=tc.get("expected_sources", []),
                reference_answer=tc.get("reference_answer", ""),
                category=tc.get("category", "factual"),
                language=tc.get("language", "en"),
            )
        )
    validate_test_cases(cases)
    return cases


# ---------------------------------------------------------------------------
# Component builder
# ---------------------------------------------------------------------------


def build_components(settings: Any, collection: str):
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    query_processor = QueryProcessor()
    return query_processor, dense_retriever, sparse_retriever


def build_search(mode: str, settings: Any, query_processor, dense_retriever, sparse_retriever):
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

    rrf_k = getattr(getattr(settings, "retrieval", None), "rrf_k", 60)
    fusion = RRFFusion(k=rrf_k)

    if mode == "dense":
        cfg = HybridSearchConfig(enable_dense=True, enable_sparse=False)
    elif mode == "sparse":
        cfg = HybridSearchConfig(enable_dense=False, enable_sparse=True)
    else:
        cfg = None  # use settings defaults

    return HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Ingestion helper
# ---------------------------------------------------------------------------


def ingest_documents(settings: Any, collection: str, doc_dir: Path) -> int:
    """Ingest all PDFs/TXTs in doc_dir into the given collection. Returns total chunk count."""
    from src.ingestion.pipeline import IngestionPipeline

    pdf_files = [
        f
        for f in sorted(doc_dir.glob("*"))
        if f.suffix.lower() in (".pdf", ".txt", ".md") and not f.name.startswith(".")
    ]
    if not pdf_files:
        print(f"  ⚠️  No documents found in {doc_dir}")
        return 0

    pipeline = IngestionPipeline(settings=settings, collection=collection, force=True)
    total = 0
    for f in pdf_files:
        try:
            result = pipeline.run(str(f))
            chunks_added = getattr(result, "chunk_count", 0) or 0
            total += chunks_added
            status = "✅" if getattr(result, "success", True) else "⚠️ "
            print(f"  {status} {f.name}: {chunks_added} chunks")
        except Exception as e:
            print(f"  ⚠️  Failed to ingest {f.name}: {e}")
    return total


def check_collection_has_data(settings: Any, collection: str) -> int:
    """Returns number of chunks in collection, or 0 on error."""
    try:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vs = VectorStoreFactory.create(settings, collection_name=collection)
        count = vs.count() if hasattr(vs, "count") else -1
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Single-mode benchmark
# ---------------------------------------------------------------------------


@dataclass
class ModeResult:
    mode: str
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    degraded_queries: int = 0
    error_queries: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


def run_mode(
    mode: str,
    search,
    test_cases: list[TestCase],
    evaluator: SourceMatchEvaluator,
    top_k: int,
    repeats: int = 1,
    reranker=None,
) -> ModeResult:
    result = ModeResult(mode=mode)
    agg: dict[str, list[float]] = {
        "hit@1": [],
        "hit@5": [],
        "hit@10": [],
        "mrr@10": [],
        "latency": [],
    }

    for repeat in range(1, repeats + 1):
        for tc in test_cases:
            t0 = time.monotonic()
            used_fallback = False
            fallback_reason = None
            error = None
            try:
                search_result = search.search(
                    query=tc.query,
                    top_k=top_k * 2 if reranker else top_k,
                    return_details=True,
                )
                chunks = search_result.results
                used_fallback = bool(search_result.used_fallback)
                fallback_reason = search_result.dense_error or search_result.sparse_error

                if reranker and chunks:
                    try:
                        rerank_res = reranker.rerank(query=tc.query, results=chunks, top_k=top_k)
                        chunks = rerank_res.results
                        if rerank_res.used_fallback:
                            used_fallback = True
                            fallback_reason = rerank_res.fallback_reason
                    except Exception as exc:
                        used_fallback = True
                        fallback_reason = f"reranker exception: {exc}"
                        chunks = chunks[:top_k]
                else:
                    chunks = chunks[:top_k]

            except Exception as exc:
                chunks = []
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = (time.monotonic() - t0) * 1000.0
            metrics = evaluator.evaluate(chunks, tc.expected_sources, top_k=top_k)

            agg["hit@1"].append(metrics["hit@1"])
            agg["hit@5"].append(metrics["hit@5"])
            agg["hit@10"].append(metrics["hit@10"])
            agg["mrr@10"].append(metrics["mrr@10"])
            agg["latency"].append(latency_ms)

            result.per_query.append(
                {
                    "id": tc.id,
                    "repeat": repeat,
                    "query": tc.query[:60],
                    "category": tc.category,
                    "language": tc.language,
                    "retrieved_sources": [_get_source(c) for c in chunks[:3]],
                    **{k: round(v, 4) for k, v in metrics.items()},
                    "latency_ms": round(latency_ms, 1),
                    "degraded": used_fallback,
                    "fallback_reason": fallback_reason,
                    "error": error,
                }
            )

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result.hit_at_1 = avg(agg["hit@1"])
    result.hit_at_5 = avg(agg["hit@5"])
    result.hit_at_10 = avg(agg["hit@10"])
    result.mrr_at_10 = avg(agg["mrr@10"])
    result.avg_latency_ms = avg(agg["latency"])
    result.latency_p50_ms = statistics.median(agg["latency"]) if agg["latency"] else 0.0
    if agg["latency"]:
        ordered = sorted(agg["latency"])
        position = (len(ordered) - 1) * 0.95
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result.latency_p95_ms = ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )
    result.degraded_queries = sum(bool(item["degraded"]) for item in result.per_query)
    result.error_queries = sum(bool(item["error"]) for item in result.per_query)
    return result


def _get_source(chunk: Any) -> str:
    if isinstance(chunk, dict):
        meta = chunk.get("metadata", {}) or {}
        return str(chunk.get("source_path", meta.get("source_path", "")))
    meta = getattr(chunk, "metadata", {}) or {}
    return str(meta.get("source_path", meta.get("source", "")))


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------


def print_table(results: list[ModeResult]) -> None:
    print()
    print("=" * 80)
    print("  BENCHMARK RESULTS — Modular RAG MCP Server")
    print("=" * 80)
    header = f"  {'Mode':<22} {'Hit@1':>7} {'Hit@5':>7} {'MRR@10':>8} {'P50':>9} {'P95':>9}"
    print(header)
    print("  " + "-" * 58)
    for r in results:
        print(
            f"  {r.mode:<22} {r.hit_at_1:>6.1%} {r.hit_at_5:>6.1%}"
            f" {r.mrr_at_10:>8.4f} {r.latency_p50_ms:>7.0f}ms {r.latency_p95_ms:>7.0f}ms"
        )
    print("=" * 80)

    # Compute improvement over dense baseline
    baseline = next((r for r in results if r.mode.lower() == "dense only"), None)
    hybrid = next(
        (r for r in results if "hybrid" in r.mode.lower() and "rerank" not in r.mode.lower()), None
    )
    if baseline and hybrid and baseline.hit_at_5 > 0:
        rel = (hybrid.hit_at_5 - baseline.hit_at_5) / baseline.hit_at_5 * 100
        print(f"\n  Hybrid vs Dense — Hit@5 delta: {rel:+.1f}%")

    print()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_results(
    results: list[ModeResult],
    out_path: Path,
    *,
    metadata: dict[str, Any],
    unavailable_modes: list[dict[str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_date": time.strftime("%Y-%m-%d"),
        "metadata": metadata,
        "unavailable_modes": unavailable_modes,
        "modes": [
            {
                "mode": r.mode,
                "hit@1": round(r.hit_at_1, 4),
                "hit@5": round(r.hit_at_5, 4),
                "hit@10": round(r.hit_at_10, 4),
                "mrr@10": round(r.mrr_at_10, 4),
                "avg_latency_ms": round(r.avg_latency_ms, 1),
                "latency_p50_ms": round(r.latency_p50_ms, 1),
                "latency_p95_ms": round(r.latency_p95_ms, 1),
                "degraded_queries": r.degraded_queries,
                "error_queries": r.error_queries,
                "per_query": r.per_query,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage K: multi-mode retrieval benchmark")
    p.add_argument("--collection", default="benchmark", help="ChromaDB collection name")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--test-set", default="tests/fixtures/golden_test_set.json")
    p.add_argument("--skip-ingest", action="store_true", help="Skip document ingestion")
    p.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    p.add_argument("--output", default="data/eval_results/benchmark_comparison.json")
    return p.parse_args()


def main() -> int:
    configure_windows_console()
    args = parse_args()

    # 1. Load settings
    try:
        from src.core.settings import load_settings

        settings = load_settings()
        print("✅ Settings loaded")
    except Exception as e:
        print(f"❌ Settings error: {e}", file=sys.stderr)
        return 2

    collection = args.collection
    top_k = args.top_k
    if top_k < 1 or args.repeats < 1:
        print("top-k and repeats must be positive integers", file=sys.stderr)
        return 2

    # 2. Optionally ingest documents
    doc_dir = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents"
    if not args.skip_ingest:
        existing = check_collection_has_data(settings, collection)
        if existing > 0:
            print(f"✅ Collection '{collection}' already has {existing} chunks — skipping ingest")
        else:
            print(f"\n📥 Ingesting documents from {doc_dir} into collection '{collection}'...")
            n = ingest_documents(settings, collection, doc_dir)
            if n == 0:
                print("⚠️  No chunks ingested — evaluation will produce zero metrics")
    else:
        print(f"⏭️  Skipping ingest (--skip-ingest). Using existing collection '{collection}'")

    # 3. Load test cases
    test_set_path = PROJECT_ROOT / args.test_set
    try:
        test_cases = load_test_cases(test_set_path)
        print(f"\n📋 Loaded {len(test_cases)} test cases from {test_set_path.name}")
    except Exception as e:
        print(f"❌ Failed to load test set: {e}", file=sys.stderr)
        return 1

    # 4. Build retrieval components (shared across modes)
    try:
        query_processor, dense_retriever, sparse_retriever = build_components(settings, collection)
        print("✅ Retrieval components initialized")
    except Exception as e:
        print(f"❌ Component initialization failed: {e}", file=sys.stderr)
        return 1

    evaluator = SourceMatchEvaluator()

    # 5. Try to build reranker (optional)
    reranker = None
    reranker_unavailable_reason = None
    try:
        runtime_cache = PROJECT_ROOT / "data" / "runtime" / "huggingface"
        runtime_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(runtime_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(runtime_cache / "hub")
        from src.core.query_engine.reranker import create_core_reranker

        reranker = create_core_reranker(settings=settings)
        if reranker and not getattr(reranker, "is_enabled", False):
            reranker_unavailable_reason = (
                "configured reranker could not be initialized; core fell back to none"
            )
            reranker = None
        if reranker:
            print("✅ Reranker initialized")
    except Exception as exc:
        reranker_unavailable_reason = f"{type(exc).__name__}: {exc}"

    # 6. Run 4 modes
    mode_configs = [
        ("Dense Only", "dense"),
        ("Sparse Only", "sparse"),
        ("Hybrid (RRF)", "hybrid"),
    ]

    results: list[ModeResult] = []
    print()

    for label, mode_key in mode_configs:
        print(f"⏱️  Running mode: {label} ...")
        search = build_search(
            mode_key, settings, query_processor, dense_retriever, sparse_retriever
        )
        r = run_mode(label, search, test_cases, evaluator, top_k, args.repeats)
        results.append(r)
        print(
            f"   Hit@1={r.hit_at_1:.1%}  Hit@5={r.hit_at_5:.1%}  "
            f"MRR@10={r.mrr_at_10:.4f}  "
            f"P50/P95={r.latency_p50_ms:.0f}/{r.latency_p95_ms:.0f}ms"
        )

    # Do not label the RRF fallback as reranked; that would invalidate the ablation.
    if reranker is not None:
        print("⏱️  Running mode: Hybrid+Rerank ...")
        hybrid_search = build_search(
            "hybrid", settings, query_processor, dense_retriever, sparse_retriever
        )
        r_rerank = run_mode(
            "Hybrid+Rerank",
            hybrid_search,
            test_cases,
            evaluator,
            top_k,
            args.repeats,
            reranker=reranker,
        )
        results.append(r_rerank)
        print(
            f"   Hit@1={r_rerank.hit_at_1:.1%}  Hit@5={r_rerank.hit_at_5:.1%}  "
            f"MRR@10={r_rerank.mrr_at_10:.4f}  "
            f"P50/P95={r_rerank.latency_p50_ms:.0f}/{r_rerank.latency_p95_ms:.0f}ms  "
            f"degraded={r_rerank.degraded_queries}"
        )
    else:
        if reranker_unavailable_reason is None:
            reranker_unavailable_reason = "configured reranker is disabled"
        print(f"  Hybrid+Rerank: N/A ({reranker_unavailable_reason})")

    # 7. Print summary
    print_table(results)

    # 8. Save JSON
    out_path = PROJECT_ROOT / args.output
    corpus_paths = [
        PROJECT_ROOT / item
        for item in json.loads(test_set_path.read_text(encoding="utf-8")).get("documents_used", [])
    ]
    metadata = {
        "status": "legacy_candidate_set_not_formal_baseline",
        "collection": collection,
        "physical_collection": getattr(dense_retriever.vector_store, "collection_name", collection),
        "vector_store_provider": settings.vector_store.provider,
        "vector_store_url": settings.vector_store.url,
        "embedding_provider": settings.embedding.provider,
        "embedding_model": settings.embedding.model,
        "embedding_dimensions": settings.embedding.dimensions,
        "reranker_provider": settings.rerank.provider,
        "reranker_model": settings.rerank.model,
        "rrf_k": settings.retrieval.rrf_k,
        "top_k": top_k,
        "repeats": args.repeats,
        "query_count": len(test_cases),
        "measurement_count_per_mode": len(test_cases) * args.repeats,
        "test_set": str(test_set_path.relative_to(PROJECT_ROOT)),
        "test_set_sha256": _sha256(test_set_path),
        "corpus": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
            }
            for path in corpus_paths
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    unavailable_modes = []
    if reranker is None:
        unavailable_modes.append(
            {
                "mode": "Hybrid+Rerank",
                "reason": reranker_unavailable_reason or "unavailable",
            }
        )
    save_results(
        results,
        out_path,
        metadata=metadata,
        unavailable_modes=unavailable_modes,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "modes": [
                        {k: v for k, v in asdict(r).items() if k != "per_query"} for r in results
                    ]
                },
                indent=2,
            )
        )

    return 1 if any(result.error_queries for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
