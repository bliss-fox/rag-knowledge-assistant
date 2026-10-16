"""Structured human-review attestations shared by public and private goldens."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REVIEW_ATTESTATION_FIELDS = (
    "query_verified",
    "answer_verified",
    "evidence_verified",
    "category_difficulty_verified",
    "split_verified",
)


def normalize_attestation(value: Mapping[str, Any] | None) -> dict[str, bool]:
    source = value or {}
    return {
        field: (
            source.get(field) is True
            or (type(source.get(field)) is int and source.get(field) == 1)
        )
        for field in REVIEW_ATTESTATION_FIELDS
    }


def attestation_complete(value: Mapping[str, Any] | None) -> bool:
    normalized = normalize_attestation(value)
    return all(normalized.values())


def require_complete_attestation(status: str, value: Mapping[str, Any] | None) -> None:
    if status == "approved" and not attestation_complete(value):
        missing = [
            field for field, checked in normalize_attestation(value).items() if not checked
        ]
        raise ValueError(
            "approved reviews require every verification attestation; missing: "
            + ", ".join(missing)
        )
