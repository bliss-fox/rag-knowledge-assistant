"""Reproducible EvidenceGate threshold calibration and artifact validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from src.core.settings import Settings, SettingsError, resolve_path
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible

REQUIRED_VISIBILITIES = ("public", "private")
CALIBRATION_SAMPLE_REQUIREMENTS = {
    "min_observations_per_visibility": 20,
    "min_supported_per_visibility": 5,
    "min_unsupported_per_visibility": 5,
}


def _validate_rate(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _dataset_sha_by_visibility(datasets: list[dict[str, Any]]) -> dict[str, str]:
    if not isinstance(datasets, list):
        raise ValueError("calibration datasets must be a list")
    result: dict[str, str] = {}
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("each calibration dataset must be an object")
        visibility = item.get("visibility")
        if visibility not in REQUIRED_VISIBILITIES:
            raise ValueError("calibration dataset visibility must be public or private")
        if visibility in result:
            raise ValueError(f"duplicate calibration dataset visibility: {visibility}")
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256.lower())
        ):
            raise ValueError(f"calibration dataset {visibility} needs a SHA-256 digest")
        result[visibility] = sha256.lower()
    if set(result) != set(REQUIRED_VISIBILITIES):
        raise ValueError("EvidenceGate calibration requires public and private dev datasets")
    return result


def _validate_and_measure_observations(
    rows: list[dict[str, Any]], datasets: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    dataset_shas = _dataset_sha_by_visibility(datasets)
    coverage = {
        visibility: {"supported": 0, "unsupported": 0, "observation_count": 0}
        for visibility in REQUIRED_VISIBILITIES
    }
    seen: set[tuple[str, str]] = set()
    if not isinstance(rows, list):
        raise ValueError("calibration observations must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each calibration observation must be an object")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("each calibration observation needs a non-empty query_id")
        visibility = row.get("visibility")
        if visibility not in REQUIRED_VISIBILITIES:
            raise ValueError("calibration observation visibility must be public or private")
        key = (visibility, query_id)
        if key in seen:
            raise ValueError(f"duplicate calibration observation: {visibility}/{query_id}")
        seen.add(key)
        dataset_sha256 = row.get("dataset_sha256")
        if not isinstance(dataset_sha256, str) or dataset_sha256.lower() != dataset_shas[visibility]:
            raise ValueError(
                f"calibration observation dataset SHA-256 mismatch for {visibility}/{query_id}"
            )
        if not isinstance(row.get("supported"), bool):
            raise ValueError("each calibration observation needs a boolean supported label")
        try:
            score = float(row["top_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each calibration observation needs a numeric top_score") from exc
        if not math.isfinite(score):
            raise ValueError("calibration scores must be finite")
        if row.get("degraded") is True:
            raise ValueError("calibration cannot use degraded retrieval observations")
        label = "supported" if row["supported"] else "unsupported"
        coverage[visibility][label] += 1
        coverage[visibility]["observation_count"] += 1
    return coverage


def _sample_requirements_met(coverage: dict[str, dict[str, int]]) -> bool:
    return all(
        coverage[visibility]["observation_count"]
        >= CALIBRATION_SAMPLE_REQUIREMENTS["min_observations_per_visibility"]
        and coverage[visibility]["supported"]
        >= CALIBRATION_SAMPLE_REQUIREMENTS["min_supported_per_visibility"]
        and coverage[visibility]["unsupported"]
        >= CALIBRATION_SAMPLE_REQUIREMENTS["min_unsupported_per_visibility"]
        for visibility in REQUIRED_VISIBILITIES
    )


def calibrate_top_score(
    rows: Iterable[dict[str, Any]], *, min_correct_refusal_rate: float = 0.80,
    max_false_refusal_rate: float = 0.20,
) -> dict[str, Any]:
    """Choose a conservative threshold from labeled dev retrieval observations."""
    min_correct_refusal_rate = _validate_rate(
        "min_correct_refusal_rate", min_correct_refusal_rate,
    )
    max_false_refusal_rate = _validate_rate(
        "max_false_refusal_rate", max_false_refusal_rate,
    )
    observations = list(rows)
    if not observations:
        raise ValueError("calibration observations are empty")
    for row in observations:
        score = float(row["top_score"])
        if not math.isfinite(score):
            raise ValueError("calibration scores must be finite")
        if not isinstance(row.get("supported"), bool):
            raise ValueError("each calibration observation needs a boolean supported label")
    positive = sum(bool(row["supported"]) for row in observations)
    negative = len(observations) - positive
    if not positive or not negative:
        raise ValueError("calibration requires both supported and unsupported dev cases")

    scores = sorted({float(row["top_score"]) for row in observations})
    upper = math.nextafter(scores[-1], math.inf)
    candidates = scores + [upper]
    evaluated = []
    for threshold in candidates:
        true_positive = sum(
            bool(row["supported"]) and float(row["top_score"]) >= threshold
            for row in observations
        )
        true_negative = sum(
            not bool(row["supported"]) and float(row["top_score"]) < threshold
            for row in observations
        )
        false_refusal_rate = 1.0 - true_positive / positive
        correct_refusal_rate = true_negative / negative
        balanced_accuracy = ((true_positive / positive) + correct_refusal_rate) / 2
        evaluated.append({
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy,
            "correct_refusal_rate": correct_refusal_rate,
            "false_refusal_rate": false_refusal_rate,
        })
    feasible = [
        row for row in evaluated
        if row["correct_refusal_rate"] >= min_correct_refusal_rate
        and row["false_refusal_rate"] <= max_false_refusal_rate
    ]
    pool = feasible or evaluated
    # Prefer the higher threshold when quality is tied: pre-generation gating is fail-closed.
    selected = max(pool, key=lambda row: (row["balanced_accuracy"], row["threshold"]))
    return {
        **selected,
        "criteria_met": bool(feasible),
        "supported_count": positive,
        "unsupported_count": negative,
        "observation_count": len(observations),
        "constraints": {
            "min_correct_refusal_rate": min_correct_refusal_rate,
            "max_false_refusal_rate": max_false_refusal_rate,
        },
    }


def build_calibration_artifact(
    rows: list[dict[str, Any]], datasets: list[dict[str, Any]], settings: Any,
    *, min_correct_refusal_rate: float = 0.80, max_false_refusal_rate: float = 0.20,
) -> dict[str, Any]:
    coverage = _validate_and_measure_observations(rows, datasets)
    result = calibrate_top_score(
        rows, min_correct_refusal_rate=min_correct_refusal_rate,
        max_false_refusal_rate=max_false_refusal_rate,
    )
    requirements_met = _sample_requirements_met(coverage)
    complete = result["criteria_met"] and requirements_met
    return {
        "artifact_type": "evidence_gate_calibration",
        "version": "1.1.0",
        "status": "calibrated" if complete else "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "dev_split_top_score_balanced_accuracy",
        "thresholds": {
            "min_top_score": result["threshold"],
            "min_evidence_count": settings.evidence.min_evidence_count,
            "min_citation_coverage": settings.evidence.min_citation_coverage,
        },
        "selection": result,
        "visibility_coverage": coverage,
        "sample_requirements": dict(CALIBRATION_SAMPLE_REQUIREMENTS),
        "sample_requirements_met": requirements_met,
        "datasets": datasets,
        "runtime": {
            "retrieval_config_sha256": retrieval_config_sha256(settings),
            "embedding": {
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
            },
            "reranker": {"provider": settings.rerank.provider, "model": settings.rerank.model},
            "index_version": settings.vector_store.index_version,
        },
        # Store labels and scores, never queries or private evidence text.
        "observations": rows,
    }


def retrieval_config_sha256(settings: Settings) -> str:
    """Hash every setting that can change retrieval-score distributions."""
    payload = {
        "embedding": asdict(settings.embedding),
        "vector_store": asdict(settings.vector_store),
        "retrieval": asdict(settings.retrieval),
        "rerank": asdict(settings.rerank),
        "ingestion": asdict(settings.ingestion) if settings.ingestion else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_calibration_artifact(settings: Settings) -> dict[str, Any] | None:
    """Validate configured calibration provenance; pending configurations stay fail-closed."""
    evidence = settings.evidence
    if evidence.calibration_status == "pending":
        return None
    if not evidence.calibration_artifact:
        raise SettingsError("calibrated EvidenceGate requires evidence.calibration_artifact")
    path = resolve_path(evidence.calibration_artifact)
    if not path.is_file():
        raise SettingsError(f"EvidenceGate calibration artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SettingsError(f"invalid EvidenceGate calibration artifact: {path}") from exc
    if payload.get("artifact_type") != "evidence_gate_calibration":
        raise SettingsError("invalid EvidenceGate calibration artifact type")
    if payload.get("version") != "1.1.0":
        raise SettingsError("unsupported EvidenceGate calibration artifact version")
    if payload.get("status") != "calibrated":
        raise SettingsError("EvidenceGate calibration artifact is incomplete")
    datasets = payload.get("datasets")
    observations = payload.get("observations")
    try:
        recomputed_coverage = _validate_and_measure_observations(observations, datasets)
    except (TypeError, ValueError) as exc:
        raise SettingsError("EvidenceGate calibration observations are invalid") from exc
    if payload.get("visibility_coverage") != recomputed_coverage:
        raise SettingsError("EvidenceGate calibration visibility coverage was modified")
    if payload.get("sample_requirements") != CALIBRATION_SAMPLE_REQUIREMENTS:
        raise SettingsError("EvidenceGate calibration sample requirements are invalid")
    if not _sample_requirements_met(recomputed_coverage):
        raise SettingsError("EvidenceGate calibration has insufficient dev observations")
    if payload.get("sample_requirements_met") is not True:
        raise SettingsError("EvidenceGate calibration sample requirements are not satisfied")
    constraints = payload.get("selection", {}).get("constraints", {})
    try:
        recomputed = calibrate_top_score(
            observations,
            min_correct_refusal_rate=constraints["min_correct_refusal_rate"],
            max_false_refusal_rate=constraints["max_false_refusal_rate"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SettingsError("EvidenceGate calibration observations are invalid") from exc
    if not recomputed["criteria_met"]:
        raise SettingsError("EvidenceGate calibration constraints are not met")
    selection = payload.get("selection", {})
    for field_name in (
        "threshold", "balanced_accuracy", "correct_refusal_rate", "false_refusal_rate",
        "criteria_met", "supported_count", "unsupported_count", "observation_count",
        "constraints",
    ):
        if selection.get(field_name) != recomputed[field_name]:
            raise SettingsError("EvidenceGate calibration selection cannot be reproduced")
    thresholds = payload.get("thresholds", {})
    expected = {
        "min_top_score": evidence.min_top_score,
        "min_evidence_count": evidence.min_evidence_count,
        "min_citation_coverage": evidence.min_citation_coverage,
    }
    if thresholds != expected:
        raise SettingsError("EvidenceGate settings do not match the calibration artifact")
    if float(thresholds["min_top_score"]) != float(recomputed["threshold"]):
        raise SettingsError("EvidenceGate calibration threshold cannot be reproduced")
    runtime = payload.get("runtime", {})
    if runtime.get("retrieval_config_sha256") != retrieval_config_sha256(settings):
        raise SettingsError("retrieval configuration changed after EvidenceGate calibration")
    observed_query_ids: dict[str, set[str]] = {
        visibility: set() for visibility in REQUIRED_VISIBILITIES
    }
    for row in observations:
        observed_query_ids[row["visibility"]].add(row["query_id"])
    for item in datasets:
        if item.get("split") != "dev":
            raise SettingsError("EvidenceGate calibration may only use dev splits")
        dataset_path = resolve_path(str(item.get("path", "")))
        if not dataset_path.is_file():
            raise SettingsError(f"calibration dataset not found: {dataset_path}")
        raw = dataset_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise SettingsError(f"calibration dataset changed: {dataset_path}")
        try:
            require_quality_gate_eligible(dataset_path)
            dataset = load_golden_dataset(dataset_path, split="dev")
        except ValueError as exc:
            raise SettingsError(
                f"calibration dataset is not human-reviewed: {dataset_path}"
            ) from exc
        if dataset.visibility != item["visibility"]:
            raise SettingsError(f"calibration dataset visibility changed: {dataset_path}")
        if item.get("name") != dataset.name or item.get("version") != dataset.version:
            raise SettingsError(f"calibration dataset identity changed: {dataset_path}")
        expected_query_ids = {case.query_id for case in dataset.cases}
        if item.get("case_count") != len(expected_query_ids):
            raise SettingsError(f"calibration dataset case count changed: {dataset_path}")
        if observed_query_ids[dataset.visibility] != expected_query_ids:
            raise SettingsError(
                f"calibration observations do not cover the full dev split: {dataset_path}"
            )
    return payload
