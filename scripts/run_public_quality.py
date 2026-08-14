#!/usr/bin/env python
"""Build an isolated public corpus and produce final-split answers for CI judging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.settings import resolve_path
from src.ingestion.pipeline import IngestionPipeline
from src.production.evaluation import (
    load_golden_dataset,
    require_quality_gate_eligible,
    retrieval_metrics,
)
from src.production.runtime import Runtime


def collect_public_sources(dataset_path: Path, dataset: object) -> list[Path]:
    """Resolve every public evidence file, including multi-hop and no-answer context."""
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    corpus = {
        str(item["document_id"]): resolve_path(str(item["path"]))
        for item in raw_dataset.get("corpus", [])
    }
    source_paths: set[Path] = set()
    repository_root = resolve_path(".")
    for case in dataset.cases:
        if corpus:
            missing = set(case.expected_document_ids) - corpus.keys()
            if missing:
                raise ValueError(
                    f"{case.query_id}: corpus is missing expected documents {sorted(missing)}"
                )
            source_paths.update(corpus[document_id] for document_id in case.expected_document_ids)
            # An unanswerable SQuAD-style case must retain its related passage; otherwise
            # refusal evaluation becomes an unrealistically easy empty-context test.
            if case.source != "manual-negative":
                source = resolve_path(case.source)
                if source not in corpus.values():
                    raise ValueError(f"{case.query_id}: source is not declared in public corpus")
                source_paths.add(source)
        elif case.answerable and case.source != "manual-negative":
            if len(case.expected_document_ids) > 1:
                raise ValueError(
                    f"{case.query_id}: legacy public dataset cannot represent multi-hop corpus"
                )
            source_paths.add(resolve_path(case.source))
    for path in sorted(source_paths):
        if repository_root != path and repository_root not in path.parents:
            raise ValueError(f"Public source is outside repository: {path}")
        if not path.is_file():
            raise ValueError(f"Public source does not exist: {path}")
    return sorted(source_paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/public_golden.json")
    parser.add_argument("--collection", default="public_ci_final")
    parser.add_argument("--output", default="artifacts/public-answers.json")
    args = parser.parse_args()
    dataset_path = resolve_path(args.dataset)
    try:
        require_quality_gate_eligible(dataset_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = load_golden_dataset(dataset_path, split="final")
    if dataset.visibility != "public":
        raise SystemExit("Only a repository public dataset may run in cloud CI")
    try:
        source_paths = collect_public_sources(dataset_path, dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime = Runtime.create()
    pipeline = IngestionPipeline(runtime.settings, collection=args.collection, force=True)
    try:
        for path in source_paths:
            result = pipeline.run(str(path))
            if not result.success:
                raise RuntimeError(result.error or f"Failed to ingest {path}")
    finally:
        pipeline.close()

    answers = []
    recalls = []
    try:
        for case in dataset.cases:
            result = runtime.rag.answer(case.query, args.collection, 5)
            document_ids = [str(item["document_id"]) for item in result["citations"]]
            if case.answerable:
                recalls.append(
                    retrieval_metrics(document_ids, case.expected_document_ids, 5)["recall_at_k"]
                )
            answers.append({
                "query_id": case.query_id, "answer": result["answer"],
                "status": result["status"], "citations": result["citations"],
                "trace_id": result["trace_id"], "prompt_version": result["prompt_version"],
                "model_version": result["model_version"],
            })
    finally:
        runtime.close()
    payload = {
        "answers": answers,
        "metrics": {"recall_at_5": sum(recalls) / len(recalls) if recalls else 0.0},
        "parameters": {"collection": args.collection, "top_k": 5,
                       "dataset_version": dataset.version, "dataset_sha256": dataset.sha256},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
