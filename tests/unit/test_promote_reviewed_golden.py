from __future__ import annotations

import json

import pytest

from scripts.promote_reviewed_golden import promote
from src.production.evaluation import require_quality_gate_eligible


def _reviewed(path, status="approved"):
    payload = {
        "artifact_type": "golden_set_reviewed_candidate",
        "name": "candidate", "version": "1", "visibility": "public",
        "review_status": "all_cases_approved", "eligible_for_quality_gate": False,
        "corpus": [],
        "cases": [{
            "query_id": "q1", "query": "Question", "category": "fact",
            "expected_document_ids": ["doc1"], "answer_key_points": ["answer"],
            "answerable": True, "source": "source.md", "split": "dev",
            "language": "en", "difficulty": "easy",
            "candidate_metadata": {"human_reviewed": True},
        }],
        "human_review": {"records": [{"query_id": "q1", "status": status}]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_promote_reviewed_candidate_into_explicitly_eligible_dataset(tmp_path, monkeypatch):
    reviewed = tmp_path / "reviewed.json"
    output = tmp_path / "public.json"
    _reviewed(reviewed)
    monkeypatch.setattr("scripts.promote_reviewed_golden.resolve_path", lambda value: tmp_path)
    result = promote(reviewed, output, "2.0.0")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["case_count"] == 1
    assert payload["artifact_type"] == "golden_set"
    assert payload["review_status"] == "human_reviewed"
    assert payload["eligible_for_quality_gate"] is True
    require_quality_gate_eligible(output)


def test_promote_rejects_nonapproved_review(tmp_path, monkeypatch):
    reviewed = tmp_path / "reviewed.json"
    _reviewed(reviewed, status="rejected")
    monkeypatch.setattr("scripts.promote_reviewed_golden.resolve_path", lambda value: tmp_path)
    with pytest.raises(ValueError, match="non-approved"):
        promote(reviewed, tmp_path / "public.json", "2.0.0")
