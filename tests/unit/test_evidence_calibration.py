from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import EvidenceSettings, SettingsError, load_settings
from src.production.evidence_calibration import (
    CALIBRATION_SAMPLE_REQUIREMENTS,
    build_calibration_artifact,
    calibrate_top_score,
    retrieval_config_sha256,
    validate_calibration_artifact,
)


def _write_reviewed_dataset(path: Path, visibility: str, count: int = 20) -> dict[str, Any]:
    cases = []
    for index in range(count):
        answerable = index < count // 2
        cases.append({
            "query_id": f"{visibility}-{index}",
            "query": f"{visibility} query {index}?",
            "category": "fact" if answerable else "unanswerable",
            "expected_document_ids": [f"{visibility}-doc-{index}"] if answerable else [],
            "answer_key_points": [f"answer-{index}"] if answerable else [],
            "answerable": answerable,
            "source": "reviewed-test-fixture",
            "split": "dev",
            "language": "en",
            "difficulty": "easy" if answerable else "hard",
        })
    payload = {
        "artifact_type": "golden_set",
        "name": f"{visibility}-golden",
        "version": "1.0.0",
        "visibility": visibility,
        "review_status": "human_reviewed",
        "eligible_for_quality_gate": True,
        "human_review": {"attestation_version": 1, "complete_attestation_count": count},
        "cases": cases,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": str(path),
        "name": payload["name"],
        "version": payload["version"],
        "visibility": visibility,
        "split": "dev",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_count": count,
    }


def _observations(metadata: dict[str, Any], count: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"{metadata['visibility']}-{index}",
            "dataset_sha256": metadata["sha256"],
            "visibility": metadata["visibility"],
            "supported": index < count // 2,
            "top_score": 0.8 if index < count // 2 else 0.2,
            "degraded": False,
        }
        for index in range(count)
    ]


def _valid_artifact(tmp_path: Path) -> tuple[dict[str, Any], Any, Path]:
    public = _write_reviewed_dataset(tmp_path / "public.json", "public")
    private = _write_reviewed_dataset(tmp_path / "private.json", "private")
    rows = _observations(public) + _observations(private)
    artifact_path = tmp_path / "calibration.json"
    evidence = EvidenceSettings(
        min_top_score=0.8,
        min_evidence_count=1,
        min_citation_coverage=0.95,
        calibration_status="calibrated",
        calibration_artifact=str(artifact_path),
    )
    settings = replace(load_settings(), evidence=evidence)
    artifact = build_calibration_artifact(rows, [public, private], settings)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact, settings, artifact_path


def test_calibration_selects_threshold_meeting_refusal_constraints():
    result = calibrate_top_score([
        {"supported": True, "top_score": 0.91},
        {"supported": True, "top_score": 0.82},
        {"supported": False, "top_score": 0.25},
        {"supported": False, "top_score": 0.31},
    ])
    assert result["criteria_met"] is True
    assert result["threshold"] == pytest.approx(0.82)
    assert result["correct_refusal_rate"] == 1.0
    assert result["false_refusal_rate"] == 0.0


def test_calibration_requires_positive_and_negative_cases():
    with pytest.raises(ValueError, match="both supported and unsupported"):
        calibrate_top_score([{"supported": True, "top_score": 0.9}])


@pytest.mark.parametrize(
    ("constraint", "value"),
    [("min_correct_refusal_rate", -0.01), ("max_false_refusal_rate", 1.01)],
)
def test_calibration_rejects_constraints_outside_unit_interval(constraint, value):
    kwargs = {constraint: value}
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_top_score([
            {"supported": True, "top_score": 0.9},
            {"supported": False, "top_score": 0.2},
        ], **kwargs)


def test_small_perfect_sample_remains_incomplete():
    settings = load_settings()
    datasets = [
        {"visibility": "public", "sha256": "a" * 64},
        {"visibility": "private", "sha256": "b" * 64},
    ]
    rows = [
        {
            "query_id": f"{visibility}-{label}",
            "visibility": visibility,
            "dataset_sha256": datasets[index]["sha256"],
            "supported": supported,
            "top_score": 0.9 if supported else 0.1,
        }
        for index, visibility in enumerate(("public", "private"))
        for label, supported in (("supported", True), ("unsupported", False))
    ]
    artifact = build_calibration_artifact(rows, datasets, settings)
    assert artifact["selection"]["criteria_met"] is True
    assert artifact["sample_requirements"] == CALIBRATION_SAMPLE_REQUIREMENTS
    assert artifact["sample_requirements_met"] is False
    assert artifact["status"] == "incomplete"


def test_artifact_requires_minimum_positive_and_negative_samples_per_visibility():
    settings = load_settings()
    datasets = [
        {"visibility": "public", "sha256": "a" * 64},
        {"visibility": "private", "sha256": "b" * 64},
    ]
    rows = []
    for dataset in datasets:
        rows.extend({
            "query_id": f"{dataset['visibility']}-{case_index}",
            "visibility": dataset["visibility"],
            "dataset_sha256": dataset["sha256"],
            "supported": case_index < 4,
            "top_score": 0.9 if case_index < 4 else 0.1,
        } for case_index in range(20))
    artifact = build_calibration_artifact(rows, datasets, settings)
    assert artifact["status"] == "incomplete"
    assert artifact["visibility_coverage"]["public"] == {
        "supported": 4,
        "unsupported": 16,
        "observation_count": 20,
    }


def test_calibrated_settings_require_matching_reviewed_datasets(tmp_path):
    artifact, settings, _ = _valid_artifact(tmp_path)
    assert artifact["status"] == "calibrated"
    assert validate_calibration_artifact(settings)["status"] == "calibrated"

    changed = EvidenceSettings(**{**settings.evidence.__dict__, "min_top_score": 0.81})
    with pytest.raises(SettingsError, match="do not match"):
        validate_calibration_artifact(replace(settings, evidence=changed))


def test_validator_recomputes_visibility_coverage(tmp_path):
    artifact, settings, artifact_path = _valid_artifact(tmp_path)
    artifact["visibility_coverage"]["public"]["supported"] = 999
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError, match="coverage was modified"):
        validate_calibration_artifact(settings)


def test_validator_rejects_observation_dataset_sha_mismatch(tmp_path):
    artifact, settings, artifact_path = _valid_artifact(tmp_path)
    artifact["observations"][0]["dataset_sha256"] = "f" * 64
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError, match="observations are invalid"):
        validate_calibration_artifact(settings)


def test_validator_rejects_duplicate_visibility_query_id(tmp_path):
    artifact, settings, artifact_path = _valid_artifact(tmp_path)
    artifact["observations"][1]["query_id"] = artifact["observations"][0]["query_id"]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError, match="observations are invalid"):
        validate_calibration_artifact(settings)


def test_validator_requires_full_dev_split_coverage(tmp_path):
    public = _write_reviewed_dataset(tmp_path / "public.json", "public")
    private = _write_reviewed_dataset(tmp_path / "private.json", "private")
    rows = _observations(public) + _observations(private)
    rows[0]["query_id"] = "public-invented"
    artifact_path = tmp_path / "calibration.json"
    evidence = EvidenceSettings(
        min_top_score=0.8,
        min_evidence_count=1,
        min_citation_coverage=0.95,
        calibration_status="calibrated",
        calibration_artifact=str(artifact_path),
    )
    settings = replace(load_settings(), evidence=evidence)
    artifact = build_calibration_artifact(rows, [public, private], settings)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError, match="full dev split"):
        validate_calibration_artifact(settings)


def test_validator_rejects_tampered_selection(tmp_path):
    artifact, settings, artifact_path = _valid_artifact(tmp_path)
    artifact["selection"]["supported_count"] = 999
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SettingsError, match="selection cannot be reproduced"):
        validate_calibration_artifact(settings)


def test_retrieval_config_hash_changes_with_reranker_model():
    settings = load_settings()
    changed = replace(settings, rerank=replace(settings.rerank, model="different-model"))
    assert retrieval_config_sha256(settings) != retrieval_config_sha256(changed)


def test_pending_calibration_needs_no_artifact():
    assert validate_calibration_artifact(load_settings()) is None
