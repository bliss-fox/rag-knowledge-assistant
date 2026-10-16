from __future__ import annotations

import json

from scripts.check_public_quality_readiness import check_readiness
from scripts.quality_gate import THRESHOLDS


def _dataset(*, eligible: bool = True) -> dict:
    return {
        "artifact_type": "golden_set",
        "name": "reviewed-public",
        "version": "1.0.0",
        "visibility": "public",
        "review_status": "human_reviewed",
        "eligible_for_quality_gate": eligible,
        "human_review": {"attestation_version": 1, "complete_attestation_count": 2},
        "cases": [
            {
                "query_id": "final-answerable", "query": "q", "category": "fact",
                "expected_document_ids": ["doc-1"], "answer_key_points": ["a"],
                "answerable": True, "source": "public.md", "split": "final",
                "language": "en", "difficulty": "easy",
            },
            {
                "query_id": "final-unanswerable", "query": "missing", "category": "none",
                "expected_document_ids": [], "answer_key_points": [], "answerable": False,
                "source": "manual-negative", "split": "final", "language": "en",
                "difficulty": "hard",
            },
        ],
    }


def test_public_quality_readiness_accepts_reviewed_final_split_and_frozen_baseline(
    tmp_path,
) -> None:
    dataset = tmp_path / "public.json"
    baseline = tmp_path / "baseline.json"
    dataset.write_text(json.dumps(_dataset()), encoding="utf-8")
    baseline.write_text(json.dumps({
        "status": "frozen", "metrics": {name: 0.95 for name in THRESHOLDS},
    }), encoding="utf-8")

    assert check_readiness(dataset, baseline) == []


def test_public_quality_readiness_fails_closed_for_current_pending_assets(tmp_path) -> None:
    dataset = tmp_path / "public.json"
    baseline = tmp_path / "baseline.json"
    dataset.write_text(json.dumps(_dataset(eligible=False)), encoding="utf-8")
    baseline.write_text(json.dumps({
        "status": "pending_reproducible_freeze", "metrics": {},
    }), encoding="utf-8")

    failures = check_readiness(dataset, baseline)

    assert any("dataset is not ready" in item for item in failures)
    assert any("not been reproducibly frozen" in item for item in failures)
    assert all(any(name in item for item in failures) for name in THRESHOLDS)
