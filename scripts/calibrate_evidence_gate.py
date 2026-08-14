#!/usr/bin/env python
"""Calibrate EvidenceGate from human-reviewed public and private dev splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.core.settings import load_settings, resolve_path
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible
from src.production.evidence_calibration import build_calibration_artifact
from src.production.runtime import Runtime


def _dataset_metadata(path: Path) -> tuple[dict[str, object], object]:
    require_quality_gate_eligible(path)
    dataset = load_golden_dataset(path, split="dev")
    return ({
        "path": path.relative_to(resolve_path(".")).as_posix(),
        "name": dataset.name,
        "version": dataset.version,
        "visibility": dataset.visibility,
        "split": "dev",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_count": len(dataset.cases),
    }, dataset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", default="evaluation/public_golden.json")
    parser.add_argument("--private", default="evaluation/private_golden.json")
    parser.add_argument("--public-collection", default="public_evidence_calibration")
    parser.add_argument("--private-collection", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-correct-refusal-rate", type=float, default=0.80)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.20)
    parser.add_argument("--output", default="evaluation/evidence_calibration.json")
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")

    public_path = resolve_path(args.public)
    private_path = resolve_path(args.private)
    public_meta, public_dataset = _dataset_metadata(public_path)
    private_meta, private_dataset = _dataset_metadata(private_path)
    if public_meta["visibility"] != "public" or private_meta["visibility"] != "private":
        raise SystemExit("--public and --private datasets have incorrect visibility")

    settings = load_settings()
    calibration_settings = replace(
        settings,
        evidence=replace(
            settings.evidence, calibration_status="pending", calibration_artifact=None,
        ),
    )
    runtime = Runtime.create(calibration_settings)
    observations: list[dict[str, object]] = []
    try:
        for metadata, dataset, collection in (
            (public_meta, public_dataset, args.public_collection),
            (private_meta, private_dataset, args.private_collection),
        ):
            for case in dataset.cases:
                response = runtime.rag.search(case.query, collection, args.top_k)
                results = response["results"]
                retrieved_ids = {
                    str(item["metadata"].get("document_id")
                        or item["metadata"].get("source_ref")
                        or item["metadata"].get("doc_hash")
                        or item["chunk_id"])
                    for item in results
                }
                observations.append({
                    "query_id": case.query_id,
                    "dataset_sha256": metadata["sha256"],
                    "visibility": metadata["visibility"],
                    "answerable": case.answerable,
                    "supported": bool(set(case.expected_document_ids) & retrieved_ids)
                    if case.answerable else False,
                    "top_score": max((float(item["score"]) for item in results), default=0.0),
                    "evidence_count": len(results),
                    "degraded": bool(response.get("degraded")),
                })
    finally:
        runtime.close()

    if any(row["degraded"] for row in observations):
        raise SystemExit("Calibration cannot use degraded retrieval observations")
    artifact = build_calibration_artifact(
        observations, [public_meta, private_meta], settings,
        min_correct_refusal_rate=args.min_correct_refusal_rate,
        max_false_refusal_rate=args.max_false_refusal_rate,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": output.as_posix(), "status": artifact["status"],
        "thresholds": artifact["thresholds"], "selection": artifact["selection"],
    }, ensure_ascii=False, indent=2))
    if artifact["status"] != "calibrated":
        raise SystemExit("Calibration constraints were not met; settings remain pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
