from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_public_golden_candidates import CandidateBuilder, SOURCES, _stable_rows


def test_deterministic_selection_does_not_depend_on_input_order():
    frame = pd.DataFrame([{"id": str(index)} for index in range(20)])
    forward = [row["id"] for row in _stable_rows(frame, 5, salt="fixed")]
    reverse = [row["id"] for row in _stable_rows(frame.iloc[::-1], 5, salt="fixed")]
    assert forward == reverse


def test_candidate_validation_rejects_unreviewed_pack_as_formal_gate_input(tmp_path):
    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")
    builder.cases = [{
        "query_id": f"candidate-{index}", "query": "q", "category": "无答案",
        "expected_document_ids": [], "answer_key_points": [], "answerable": False,
        "source": "candidate", "split": "dev" if index % 2 == 0 else "final",
        "language": "zh", "difficulty": "hard",
        "candidate_metadata": {"human_reviewed": False},
    } for index in range(100)]
    target = builder.write()
    payload = target.read_text(encoding="utf-8")
    assert '"eligible_for_quality_gate": false' in payload
    assert payload.count('"human_reviewed": false') == 100


def test_all_upstream_licenses_are_explicitly_allowlisted():
    assert {source["license"] for source in SOURCES.values()} == {"cc-by-sa-4.0"}


def test_candidate_pack_requires_exactly_one_hundred_cases(tmp_path):
    builder = CandidateBuilder(Path(tmp_path))
    with pytest.raises(ValueError, match="must contain 100 cases"):
        builder.write()
