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
from src.production.evaluation import require_quality_gate_eligible


def calibrate_top_score(
    rows: Iterable[dict[str, Any]], *, min_correct_refusal_rate: float = 0.80,
    max_false_refusal_rate: float = 0.20,
) -> dict[str, Any]:
    """Choose a conservative threshold from labeled dev retrieval observations."""
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
    result = calibrate_top_score(
        rows, min_correct_refusal_rate=min_correct_refusal_rate,
        max_false_refusal_rate=max_false_refusal_rate,
    )
    visibilities = {str(item["visibility"]) for item in datasets}
    coverage = {
        visibility: {
            "supported": sum(
                row["visibility"] == visibility and bool(row["supported"]) for row in rows
            ),
            "unsupported": sum(
                row["visibility"] == visibility and not bool(row["supported"]) for row in rows
            ),
        }
        for visibility in visibilities
    }
    complete = (
        result["criteria_met"]
        and visibilities == {"public", "private"}
        and all(values["supported"] and values["unsupported"] for values in coverage.values())
    )
    return {
        "artifact_type": "evidence_gate_calibration",
        "version": "1.0.0",
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
    if payload.get("status") != "calibrated":
        raise SettingsError("EvidenceGate calibration artifact is incomplete")
    coverage = payload.get("visibility_coverage", {})
    if any(
        not coverage.get(visibility, {}).get("supported")
        or not coverage.get(visibility, {}).get("unsupported")
        for visibility in ("public", "private")
    ):
        raise SettingsError(
            "EvidenceGate calibration needs supported and unsupported cases in each dataset"
        )
    observations = payload.get("observations")
    constraints = payload.get("selection", {}).get("constraints", {})
    try:
        recomputed = calibrate_top_score(
            observations,
            min_correct_refusal_rate=float(constraints["min_correct_refusal_rate"]),
            max_false_refusal_rate=float(constraints["max_false_refusal_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SettingsError("EvidenceGate calibration observations are invalid") from exc
    if not recomputed["criteria_met"]:
        raise SettingsError("EvidenceGate calibration constraints are not met")
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
    datasets = payload.get("datasets", [])
    if {item.get("visibility") for item in datasets} != {"public", "private"}:
        raise SettingsError("EvidenceGate calibration requires public and private dev datasets")
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
        except ValueError as exc:
            raise SettingsError(
                f"calibration dataset is not human-reviewed: {dataset_path}"
            ) from exc
    return payload
