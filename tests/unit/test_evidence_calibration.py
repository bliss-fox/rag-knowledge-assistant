from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.core.settings import EvidenceSettings, SettingsError, load_settings
from src.production.evidence_calibration import (
    build_calibration_artifact,
    calibrate_top_score,
    retrieval_config_sha256,
    validate_calibration_artifact,
)


def test_calibration_selects_threshold_meeting_refusal_constraints():
    result = calibrate_top_score([
        {"supported": True, "top_score": .91},
        {"supported": True, "top_score": .82},
        {"supported": False, "top_score": .25},
        {"supported": False, "top_score": .31},
    ])
    assert result["criteria_met"] is True
    assert result["threshold"] == pytest.approx(.82)
    assert result["correct_refusal_rate"] == 1.0
    assert result["false_refusal_rate"] == 0.0


def test_calibration_requires_positive_and_negative_cases():
    with pytest.raises(ValueError, match="both supported and unsupported"):
        calibrate_top_score([{"supported": True, "top_score": .9}])


def test_artifact_requires_both_labels_in_each_visibility():
    settings = load_settings()
    rows = [
        {"visibility": "public", "supported": True, "top_score": .9},
        {"visibility": "public", "supported": False, "top_score": .2},
        {"visibility": "private", "supported": True, "top_score": .8},
    ]
    artifact = build_calibration_artifact(rows, [
        {"visibility": "public"}, {"visibility": "private"},
    ], settings)
    assert artifact["status"] == "incomplete"
    assert artifact["visibility_coverage"]["private"]["unsupported"] == 0


def test_calibrated_settings_require_matching_reviewed_datasets(tmp_path, monkeypatch):
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"
    base = {
        "artifact_type": "golden_set", "version": "1", "review_status": "human_reviewed",
        "eligible_for_quality_gate": True, "cases": [],
    }
    public.write_text(
        json.dumps({**base, "name": "public", "visibility": "public"}), encoding="utf-8",
    )
    private.write_text(
        json.dumps({**base, "name": "private", "visibility": "private"}), encoding="utf-8",
    )
    artifact = tmp_path / "calibration.json"
    evidence = EvidenceSettings(
        min_top_score=.7, min_evidence_count=1, min_citation_coverage=.95,
        calibration_status="calibrated", calibration_artifact=str(artifact),
    )
    settings = replace(load_settings(), evidence=evidence)
    artifact.write_text(json.dumps({
        "artifact_type": "evidence_gate_calibration", "status": "calibrated",
        "thresholds": {
            "min_top_score": .7, "min_evidence_count": 1, "min_citation_coverage": .95,
        },
        "datasets": [
            {"path": str(public), "visibility": "public", "split": "dev",
             "sha256": hashlib.sha256(public.read_bytes()).hexdigest()},
            {"path": str(private), "visibility": "private", "split": "dev",
             "sha256": hashlib.sha256(private.read_bytes()).hexdigest()},
        ],
        "selection": {
            "constraints": {"min_correct_refusal_rate": .8, "max_false_refusal_rate": .2},
        },
        "runtime": {"retrieval_config_sha256": retrieval_config_sha256(settings)},
        "observations": [
            {"supported": True, "top_score": .7},
            {"supported": False, "top_score": .2},
        ],
        "visibility_coverage": {
            "public": {"supported": 1, "unsupported": 1},
            "private": {"supported": 1, "unsupported": 1},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        "src.production.evidence_calibration.resolve_path", lambda value: Path(value),
    )
    assert validate_calibration_artifact(settings)["status"] == "calibrated"

    changed = EvidenceSettings(**{**evidence.__dict__, "min_top_score": .8})
    with pytest.raises(SettingsError, match="do not match"):
        validate_calibration_artifact(replace(settings, evidence=changed))


def test_pending_calibration_needs_no_artifact():
    assert validate_calibration_artifact(load_settings()) is None
