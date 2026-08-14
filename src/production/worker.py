"""Persistent single-node worker for ingestion, source sync and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from src.core.trace import TraceContext
from src.ingestion.pipeline import IngestionPipeline
from src.production.runtime import Runtime


def validate_evaluation_privacy(visibility: str, llm_provider: str) -> None:
    """Fail closed before loading retrieval or judge services for private datasets."""
    if visibility == "private" and llm_provider.lower() != "ollama":
        raise RuntimeError(
            "Private evaluation requires the local Ollama LLM provider; "
            "refusing to transmit private evaluation content"
        )


class Worker:
    def __init__(self, runtime: Runtime, poll_seconds: float = 1.0) -> None:
        self.runtime = runtime
        self.poll_seconds = poll_seconds
        self.worker_id = f"{socket.gethostname()}:{uuid.uuid4()}"
        self.handlers: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
            "ingest_file": self._ingest_file,
            "sync_source": self._sync_source,
            "delete_source": self._delete_source,
            "evaluation": self._evaluation,
            "purge_trace_content": self._purge_trace_content,
        }

    def run_once(self) -> bool:
        job = self.runtime.jobs.claim_next(self.worker_id)
        if job is None:
            return False
        try:
            handler = self.handlers.get(job["kind"])
            if handler is None:
                raise ValueError(f"Unsupported job kind: {job['kind']}")
            result = handler(job["payload"], job["id"])
            self.runtime.jobs.finish(job["id"], result)
        except Exception as exc:
            self.runtime.jobs.fail(job["id"], f"{type(exc).__name__}: {exc}")
        return True

    def run_forever(self) -> None:
        self.runtime.jobs.recover_stale()
        while True:
            if not self.run_once():
                time.sleep(self.poll_seconds)

    def _ingest_file(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        path = Path(payload["path"])
        collection = str(payload.get("collection", "default"))
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        pipeline = IngestionPipeline(
            self.runtime.settings,
            collection=collection,
            force=bool(payload.get("force", False)),
        )
        trace = TraceContext(trace_type="ingestion")
        trace.metadata.update({"source_path": str(path), "job_id": job_id, "status": "running"})
        try:
            result = pipeline.run(str(path), trace=trace, on_progress=lambda *_: self._heartbeat(job_id))
            trace.metadata["status"] = "succeeded" if result.success else "error"
            if not result.success:
                trace.metadata["error_type"] = "IngestionError"
                raise RuntimeError(result.error or "Ingestion failed")
            return result.to_dict()
        finally:
            trace.finish()
            self.runtime.traces.save(trace.to_dict())
            pipeline.close()

    def _sync_source(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        source_id = str(payload["source_id"])
        source = self.runtime.sources.get(source_id)
        if source["kind"] == "web":
            return self._sync_web(source, job_id)
        ingestion = self.runtime.settings.ingestion
        changes = self.runtime.sources.scan_directory(
            source_id,
            ingestion.allowed_extensions if ingestion else (".pdf", ".md", ".txt"),
            (ingestion.max_file_size_mb if ingestion else 100) * 1024 * 1024,
        )
        processed = 0
        failed: list[dict[str, str]] = []
        cancelled = False
        for item in (*changes.added, *changes.modified):
            if self.runtime.jobs.is_cancel_requested(job_id):
                cancelled = True
                break
            try:
                old_manifest = self.runtime.sources.get_manifest(source_id, item["stable_id"])
                result = self._ingest_file({
                    "path": item["path"], "collection": source["collection_name"],
                    # Source manifests own idempotency. Force avoids the global
                    # ingestion history suppressing indexing into another collection.
                    "force": True,
                }, job_id)
                document_id = str(
                    result.get("stages", {}).get("loading", {}).get("doc_id")
                    or result.get("doc_id") or item["content_hash"]
                )
                if old_manifest:
                    stale_ids = sorted(set(old_manifest["chunk_ids"]) - set(result.get("vector_ids", [])))
                    self._delete_manifest_entries(
                        source_id,
                        old_manifest,
                        chunk_ids=stale_ids,
                        remove_bm25=old_manifest["document_id"] != document_id,
                    )
                self.runtime.sources.save_manifest(
                    source_id, item["stable_id"], source["collection_name"],
                    document_id, item["content_hash"],
                    result.get("vector_ids", []),
                )
                self.runtime.sources.mark_status(source_id, item["stable_id"], "indexed")
                processed += 1
            except Exception as exc:
                self.runtime.sources.mark_status(source_id, item["stable_id"], "failed")
                failed.append({"path": item["path"], "error": str(exc)})
        deleted = 0
        if not cancelled:
            for item in changes.deleted:
                manifest = self.runtime.sources.get_manifest(source_id, item["stable_id"])
                if manifest:
                    self._delete_manifest_entries(source_id, manifest)
                self.runtime.sources.delete_manifest(source_id, item["stable_id"])
                deleted += 1
        return {"changes": changes.to_dict(), "processed": processed, "failed": failed,
                "deleted": deleted, "cancelled": cancelled}

    def _delete_indexed_document(
        self,
        manifest: dict[str, Any],
        chunk_ids: list[str] | None = None,
        remove_bm25: bool = True,
    ) -> None:
        """Delete only IDs recorded by our manifest from dense and sparse indexes."""
        pipeline = self.runtime.rag._pipeline(manifest["collection_name"])
        ids_to_delete = manifest.get("chunk_ids") if chunk_ids is None else chunk_ids
        if ids_to_delete:
            pipeline.vector_store.delete(ids_to_delete)
        from src.core.settings import resolve_path
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        if remove_bm25:
            BM25Indexer(index_dir=str(resolve_path(
                f"data/db/bm25/{manifest['collection_name']}"
            ))).remove_document(manifest["document_id"], manifest["collection_name"])

    def _delete_manifest_entries(
        self,
        source_id: str,
        manifest: dict[str, Any],
        chunk_ids: list[str] | None = None,
        remove_bm25: bool = True,
    ) -> tuple[int, bool]:
        """Delete manifest entries only when no other manifest owns them."""
        shared_chunks, shared_documents = self.runtime.sources.index_references(
            source_id,
            manifest["collection_name"],
            stable_id=manifest["stable_id"],
        )
        candidates = manifest.get("chunk_ids", []) if chunk_ids is None else chunk_ids
        exclusive_chunks = [item for item in candidates if item not in shared_chunks]
        remove_document = remove_bm25 and manifest["document_id"] not in shared_documents
        self._delete_indexed_document(
            manifest,
            chunk_ids=exclusive_chunks,
            remove_bm25=remove_document,
        )
        return len(exclusive_chunks), remove_document

    def _delete_source(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        """Remove only index entries exclusively owned by this source, then its catalog row."""
        source_id = str(payload["source_id"])
        source = self.runtime.sources.get(source_id)
        manifests = self.runtime.sources.list_manifests(source_id)
        deleted_chunks = 0
        deleted_documents = 0
        for manifest in manifests:
            self._heartbeat(job_id)
            shared_chunks, shared_documents = self.runtime.sources.index_references(
                source_id, manifest["collection_name"],
            )
            exclusive_chunks = [
                chunk_id for chunk_id in manifest["chunk_ids"]
                if chunk_id not in shared_chunks
            ]
            remove_bm25 = manifest["document_id"] not in shared_documents
            self._delete_indexed_document(
                manifest, chunk_ids=exclusive_chunks, remove_bm25=remove_bm25,
            )
            deleted_chunks += len(exclusive_chunks)
            deleted_documents += int(remove_bm25)
        self.runtime.sources.delete(source_id)
        deleted_snapshots = 0
        if source["kind"] == "web":
            deleted_snapshots = self._delete_web_snapshots(source_id)
        return {
            "source_id": source_id,
            "source_name": source["name"],
            "manifests": len(manifests),
            "deleted_chunks": deleted_chunks,
            "deleted_documents": deleted_documents,
            "deleted_snapshots": deleted_snapshots,
        }

    @staticmethod
    def _delete_web_snapshots(source_id: str) -> int:
        """Delete only this source's application-owned normalized snapshots."""
        from src.core.settings import resolve_path

        snapshots_root = resolve_path("data/uploads/web").resolve()
        target = (snapshots_root / source_id).resolve()
        if target.parent != snapshots_root:
            raise RuntimeError("Refusing to delete snapshots outside the web upload root")
        if not target.exists():
            return 0
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError("Refusing to delete an unsafe web snapshot target")
        file_count = sum(1 for item in target.rglob("*") if item.is_file())
        shutil.rmtree(target)
        return file_count

    def _sync_web(self, source: dict[str, Any], job_id: str) -> dict[str, Any]:
        from src.libs.loader.web_loader import WebLoader
        config = source["config"]
        crawler = self.runtime.settings.web_crawler
        loader = WebLoader(
            max_depth=int(config.get("max_depth", crawler.max_depth)),
            max_pages=int(config.get("max_pages", crawler.max_pages)),
            max_response_bytes=crawler.max_response_mb * 1024 * 1024,
            timeout=crawler.timeout_seconds,
            allowed_path_prefixes=config.get("allowed_path_prefixes", []),
        )
        crawl = loader.crawl_with_report(source["location"])
        # Persist normalized web snapshots inside the application's upload area so the
        # standard ingestion pipeline remains the only indexing path.
        from src.core.settings import resolve_path

        root = resolve_path("data/uploads/web") / source["id"]
        root.mkdir(parents=True, exist_ok=True)
        current: dict[str, dict[str, Any]] = {}
        for document in crawl.documents:
            source_url = str(document.metadata["source_uri"])
            stable_id = hashlib.sha256(f"{source['id']}:{source_url}".encode()).hexdigest()
            content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            current[stable_id] = {
                "stable_id": stable_id,
                "source_url": source_url,
                "content_hash": content_hash,
                "document": document,
                "path": root / f"{stable_id}.md",
            }

        previous = {
            item["stable_id"]: item for item in self.runtime.sources.list_manifests(source["id"])
        }
        added = [item for key, item in current.items() if key not in previous]
        modified = [
            item for key, item in current.items()
            if key in previous and item["content_hash"] != previous[key]["content_hash"]
        ]
        unchanged = len(current) - len(added) - len(modified)
        deleted_manifests = (
            [item for key, item in previous.items() if key not in current]
            if crawl.complete else []
        )

        processed = 0
        failed: list[dict[str, str]] = []
        for item in (*added, *modified):
            self._heartbeat(job_id)
            path = item["path"]
            document = item["document"]
            header = f"---\nsource_url: {json.dumps(item['source_url'])}\n---\n\n"
            try:
                path.write_text(header + document.text, encoding="utf-8")
                old_manifest = previous.get(item["stable_id"])
                result = self._ingest_file({
                    "path": str(path),
                    "collection": source["collection_name"],
                    # The web manifest already filters unchanged pages. Force is
                    # required when identical content belongs in another collection.
                    "force": True,
                }, job_id)
                document_id = str(
                    result.get("stages", {}).get("loading", {}).get("doc_id")
                    or result.get("doc_id")
                    or item["content_hash"]
                )
                vector_ids = list(result.get("vector_ids", []))
                if old_manifest:
                    stale_ids = sorted(set(old_manifest["chunk_ids"]) - set(vector_ids))
                    self._delete_manifest_entries(
                        source["id"],
                        old_manifest,
                        chunk_ids=stale_ids,
                        remove_bm25=old_manifest["document_id"] != document_id,
                    )
                self.runtime.sources.save_manifest(
                    source["id"],
                    item["stable_id"],
                    source["collection_name"],
                    document_id,
                    item["content_hash"],
                    vector_ids,
                )
                processed += 1
            except Exception as exc:
                if self.runtime.jobs.is_cancel_requested(job_id):
                    raise
                failed.append({"url": item["source_url"], "error": str(exc)})

        deleted = 0
        for manifest in deleted_manifests:
            self._heartbeat(job_id)
            self._delete_manifest_entries(source["id"], manifest)
            self.runtime.sources.delete_manifest(source["id"], manifest["stable_id"])
            snapshot = root / f"{manifest['stable_id']}.md"
            if snapshot.exists():
                snapshot.unlink()
            deleted += 1

        changes = {
            "added": [self._web_change(item) for item in added],
            "modified": [self._web_change(item) for item in modified],
            "deleted": [self._web_deleted_change(item) for item in deleted_manifests],
            "unchanged": unchanged,
        }
        result = {
            **crawl.to_dict(),
            "changes": changes,
            "processed": processed,
            "failed": failed,
            "deleted": deleted,
            "deletion_skipped_reason": None if crawl.complete else crawl.truncation_reason,
        }
        return result

    @staticmethod
    def _web_change(item: dict[str, Any]) -> dict[str, str]:
        return {
            "stable_id": str(item["stable_id"]),
            "source_url": str(item["source_url"]),
            "content_hash": str(item["content_hash"]),
            "path": str(item["path"]),
        }

    @staticmethod
    def _web_deleted_change(manifest: dict[str, Any]) -> dict[str, str]:
        return {
            "stable_id": str(manifest["stable_id"]),
            "document_id": str(manifest["document_id"]),
            "content_hash": str(manifest["content_hash"]),
        }

    def _evaluation(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.libs.llm.base_llm import Message
        from src.libs.llm.llm_factory import LLMFactory
        from src.production.evaluation import (
            AblationEvaluator,
            AnswerQualityEvaluator,
            load_golden_dataset,
            parse_judge_scores,
        )

        evaluation_id = str(payload.get("evaluation_id") or job_id)
        path = Path(payload["dataset_path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            with self.runtime.database.transaction(immediate=True) as connection:
                connection.execute(
                    "UPDATE evaluations SET status='running' WHERE id=?",
                    (evaluation_id,),
                )
            dataset = load_golden_dataset(path, split=str(payload.get("split", "final")))
            llm_provider = self.runtime.settings.llm.provider.lower()
            validate_evaluation_privacy(dataset.visibility, llm_provider)
            pipeline = self.runtime.rag._pipeline(str(payload.get("collection", "evaluation")))
            query_processor = QueryProcessor()
            k = int(payload.get("top_k", 5))
            repeats = int(payload.get("repeats", 5))

            def normalized(results: list[Any]) -> list[dict[str, Any]]:
                return [{
                    "document_id": str(
                        item.metadata.get("document_id")
                        or item.metadata.get("source_path")
                        or item.metadata.get("source")
                        or item.chunk_id
                    ),
                    "chunk_id": item.chunk_id,
                    "score": float(item.score),
                } for item in results]

            def bm25(query: str, limit: int) -> list[dict[str, Any]]:
                self._heartbeat(job_id)
                processed = query_processor.process(query)
                if not processed.keywords:
                    return []
                return normalized(pipeline.hybrid.sparse_retriever.retrieve(
                    processed.keywords, top_k=limit,
                ))

            def dense(query: str, limit: int) -> list[dict[str, Any]]:
                self._heartbeat(job_id)
                return normalized(pipeline.hybrid.dense_retriever.retrieve(query, top_k=limit))

            def hybrid(query: str, limit: int) -> list[dict[str, Any]]:
                self._heartbeat(job_id)
                return normalized(pipeline.hybrid.search(query, top_k=limit))

            variants: dict[str, Callable[[str, int], list[dict[str, Any]]]] = {
                "bm25": bm25,
                "dense": dense,
                "hybrid_rrf": hybrid,
            }
            unavailable: dict[str, str] = {}
            if pipeline.reranker.is_enabled:
                def hybrid_cross_encoder(query: str, limit: int) -> list[dict[str, Any]]:
                    self._heartbeat(job_id)
                    candidates = pipeline.hybrid.search(
                        query, top_k=max(limit, self.runtime.settings.rerank.candidate_top_k)
                    )
                    reranked = pipeline.reranker.rerank(query, candidates, top_k=limit)
                    if reranked.used_fallback:
                        raise RuntimeError(f"reranker degraded: {reranked.fallback_reason}")
                    return normalized(reranked.results)

                variants["hybrid_cross_encoder"] = hybrid_cross_encoder
            else:
                unavailable["hybrid_cross_encoder"] = "configured reranker is unavailable"

            report = AblationEvaluator(dataset, k=k, repeats=repeats).run(variants)
            report["unavailable_variants"] = unavailable
            report["collection"] = str(payload.get("collection", "evaluation"))

            judge_prompt = self.runtime.prompts.get("grounded_answer_quality_judge")
            judge_llm = self.runtime.rag.llm or LLMFactory.create(self.runtime.settings)

            def answer_case(case: Any) -> dict[str, Any]:
                self._heartbeat(job_id)
                return self.runtime.rag.answer(
                    case.query,
                    str(payload.get("collection", "evaluation")),
                    k,
                    user_id=str(payload.get("created_by") or "evaluation-worker"),
                )

            def judge_case(case: Any, response: dict[str, Any]) -> dict[str, float]:
                self._heartbeat(job_id)
                rendered = judge_prompt.render(
                    query=case.query,
                    key_points=json.dumps(case.answer_key_points, ensure_ascii=False),
                    answer=str(response.get("answer", "")),
                    citations=json.dumps(response.get("citations", []), ensure_ascii=False),
                    visibility=dataset.visibility,
                )
                judged = judge_llm.chat(
                    [Message(role="user", content=rendered)],
                    temperature=0.0,
                )
                return parse_judge_scores(judged.content)

            answer_report = AnswerQualityEvaluator(dataset).run(answer_case, judge_case)
            report["answer_quality"] = answer_report
            report["judge"] = {
                "provider": llm_provider,
                "model": self.runtime.settings.llm.model,
                "temperature": 0.0,
                "prompt_id": judge_prompt.prompt_id,
                "prompt_version": judge_prompt.version,
                "prompt_sha256": judge_prompt.sha256,
                "status": "failed" if answer_report["judge_errors"] else "completed",
            }
            from src.core.settings import resolve_path

            report_dir = resolve_path("outputs/evaluations")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{evaluation_id}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics = {item["variant"]: {
                key: item[key] for key in (
                    "hit_at_k", "recall_at_k", "mrr", "ndcg_at_k",
                    "latency_p50_ms", "latency_p95_ms",
                )
            } for item in report["variants"]}
            metrics["answer_quality"] = answer_report["metrics"]
            with self.runtime.database.transaction(immediate=True) as connection:
                connection.execute(
                    """UPDATE evaluations SET dataset_name=?,dataset_version=?,status='completed',
                       metrics_json=?,report_path=?,finished_at=CURRENT_TIMESTAMP WHERE id=?""",
                    (dataset.name, dataset.version, json.dumps(metrics, ensure_ascii=False),
                     str(report_path), evaluation_id),
                )
            return {"evaluation_id": evaluation_id, "report_path": str(report_path),
                    "metrics": metrics, "query_count": len(dataset.cases),
                    "unavailable_variants": unavailable,
                    "unavailable_answer_metrics": answer_report["unavailable_metrics"]}
        except Exception:
            with self.runtime.database.transaction(immediate=True) as connection:
                connection.execute(
                    "UPDATE evaluations SET status='failed',finished_at=CURRENT_TIMESTAMP WHERE id=?",
                    (evaluation_id,),
                )
            raise

    def _purge_trace_content(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        del payload, job_id
        return {"purged": self.runtime.traces.purge_content()}

    def _heartbeat(self, job_id: str) -> None:
        if self.runtime.jobs.is_cancel_requested(job_id):
            raise RuntimeError("Job cancellation requested")
        self.runtime.jobs.heartbeat(job_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    runtime = Runtime.create()
    try:
        worker = Worker(runtime)
        worker.run_once() if args.once else worker.run_forever()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
