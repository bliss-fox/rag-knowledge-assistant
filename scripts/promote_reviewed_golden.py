#!/usr/bin/env python
"""Promote a reviewed candidate into the canonical public golden dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.core.settings import resolve_path
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible


def promote(reviewed: Path, output: Path, version: str) -> dict[str, object]:
    payload = json.loads(reviewed.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "golden_set_reviewed_candidate":
        raise ValueError("input is not a reviewed golden-set candidate")
    if payload.get("review_status") != "all_cases_approved":
        raise ValueError("reviewed candidate is not fully approved")
    cases = payload.get("cases", [])
    records = payload.get("human_review", {}).get("records", [])
    if not cases or len(records) != len(cases):
        raise ValueError("reviewed candidate has incomplete review records")
    query_ids = [str(case.get("query_id")) for case in cases]
    review_ids = [str(record.get("query_id")) for record in records]
    if len(set(query_ids)) != len(query_ids) or set(query_ids) != set(review_ids):
        raise ValueError("review records do not match candidate query IDs")
    if any(record.get("status") != "approved" for record in records):
        raise ValueError("reviewed candidate contains a non-approved review")
    if any(not case.get("candidate_metadata", {}).get("human_reviewed") for case in cases):
        raise ValueError("reviewed candidate contains an unreviewed case")
    promoted = {
        **payload,
        "artifact_type": "golden_set",
        "name": "modular-rag-public-golden",
        "version": version,
        "visibility": "public",
        "review_status": "human_reviewed",
        "eligible_for_quality_gate": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_from": reviewed.relative_to(resolve_path(".")).as_posix(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        load_golden_dataset(temporary)
        require_quality_gate_eligible(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    dataset = load_golden_dataset(output)
    return {
        "output": output.as_posix(), "version": dataset.version,
        "case_count": len(dataset.cases), "sha256": dataset.sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--output", default="evaluation/public_golden.json")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    result = promote(resolve_path(args.reviewed), resolve_path(args.output), args.version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
