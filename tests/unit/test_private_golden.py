from __future__ import annotations

import json

import pytest

from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible
from src.production.private_golden import PrivateGoldenService

_ATTESTATION = {
    "query_verified": True,
    "answer_verified": True,
    "evidence_verified": True,
    "category_difficulty_verified": True,
    "split_verified": True,
}


def _service(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("private.admin", "correct horse battery staple")
    return PrivateGoldenService(database, tmp_path), database, admin.user_id


def _case(index: int) -> dict:
    unanswerable = index == 1
    return {
        "query": f"私有问题 {index}",
        "category": "无答案" if unanswerable else "精确事实",
        "expected_document_ids": [] if unanswerable else [f"private-doc-{index}"],
        "answer_key_points": [] if unanswerable else [f"答案要点 {index}"],
        "answerable": not unanswerable,
        "source": "local-only",
        "split": "dev" if index < 25 else "final",
        "language": "zh",
        "difficulty": "hard" if unanswerable else "medium",
    }


def _approve(service, set_id: str, query_id: str, admin_id: str, notes: str = "ok"):
    return service.review_case(
        set_id, query_id, "approved", notes, admin_id, attestation=_ATTESTATION,
    )


def test_private_golden_requires_individual_review_and_exports_local_versioned_artifact(
    tmp_path,
) -> None:
    service, _, admin_id = _service(tmp_path)
    dataset = service.create_set("internal-knowledge-golden", "1.0.0", admin_id)
    assert dataset["case_count"] == 0
    assert dataset["export_ready"] is False

    for index in range(50):
        query_id = f"private-{index:03d}"
        saved = service.put_case(dataset["id"], query_id, _case(index), admin_id)
        assert saved["review_status"] == "pending"
        _approve(service, dataset["id"], query_id, admin_id, "人工核对")

    summary = service.get_set(dataset["id"])
    assert summary["case_count"] == 50
    assert summary["review_counts"] == {"pending": 0, "approved": 50, "rejected": 0}
    assert summary["split_counts"] == {"dev": 25, "final": 25}
    assert summary["answerability_counts"] == {"answerable": 49, "unanswerable": 1}
    assert summary["export_ready"] is True

    exported = service.export(dataset["id"], admin_id)
    output = tmp_path / "evaluation" / "private_golden.json"
    assert exported["case_count"] == 50
    assert exported["path"] == "evaluation/private_golden.json"
    assert exported["eligible_for_quality_gate"] is True
    assert service.get_set(dataset["id"])["status"] == "reviewed"
    require_quality_gate_eligible(output)
    loaded = load_golden_dataset(output)
    assert loaded.visibility == "private" and len(loaded.cases) == 50
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["local_only"] is True
    assert payload["review_status"] == "human_reviewed"
    assert payload["human_review"]["attestation_version"] == 1
    assert payload["human_review"]["complete_attestation_count"] == 50

    changed = _case(0)
    changed["query"] = "修改后的私有问题"
    service.put_case(dataset["id"], "private-000", changed, admin_id)
    assert service.get_set(dataset["id"])["export_ready"] is False
    _approve(service, dataset["id"], "private-000", admin_id, "重新核对")
    with pytest.raises(ValueError, match="increment its version"):
        service.export(dataset["id"], admin_id)
    service.update_set(dataset["id"], version="1.0.1")
    replacement = service.export(dataset["id"], admin_id)
    assert replacement["version"] == "1.0.1"
    assert service.export(dataset["id"], admin_id)["sha256"] == replacement["sha256"]

    service.update_set(dataset["id"], name="internal-knowledge-golden")
    assert service.get_set(dataset["id"])["status"] == "draft"
    assert service.export(dataset["id"], admin_id)["sha256"] == replacement["sha256"]
    assert service.get_set(dataset["id"])["status"] == "reviewed"


def test_private_golden_metadata_and_canonical_export_require_version_increase(tmp_path) -> None:
    service, _, admin_id = _service(tmp_path)
    dataset = service.create_set("private-a", "1.0.0", admin_id)
    for index in range(50):
        query_id = f"a-{index:03d}"
        service.put_case(dataset["id"], query_id, _case(index), admin_id)
        _approve(service, dataset["id"], query_id, admin_id)
    service.export(dataset["id"], admin_id)

    service.update_set(dataset["id"], name="renamed-without-version-change")
    with pytest.raises(ValueError, match="metadata changed"):
        service.export(dataset["id"], admin_id)

    other = service.create_set("private-b", "1.0.0", admin_id)
    for index in range(50):
        query_id = f"b-{index:03d}"
        service.put_case(other["id"], query_id, _case(index), admin_id)
        _approve(service, other["id"], query_id, admin_id)
    with pytest.raises(ValueError, match="increment the new dataset version"):
        service.export(other["id"], admin_id)
    service.update_set(other["id"], version="1.0.1")
    assert service.export(other["id"], admin_id)["version"] == "1.0.1"


def test_private_golden_case_validation_and_export_requirements(tmp_path) -> None:
    service, _, admin_id = _service(tmp_path)
    dataset = service.create_set("private", "1.0.0", admin_id)
    invalid = _case(0)
    invalid["answer_key_points"] = []
    with pytest.raises(ValueError, match="answer key points"):
        service.put_case(dataset["id"], "missing-key-points", invalid, admin_id)
    invalid = _case(1)
    invalid["expected_document_ids"] = ["should-not-exist"]
    with pytest.raises(ValueError, match="must not contain"):
        service.put_case(dataset["id"], "bad-unanswerable", invalid, admin_id)
    with pytest.raises(ValueError, match="not exportable"):
        service.export(dataset["id"], admin_id)


def test_private_golden_edits_reset_approval_and_list_hides_content(tmp_path) -> None:
    service, _, admin_id = _service(tmp_path)
    dataset = service.create_set("private", "1.0.0", admin_id)
    service.put_case(dataset["id"], "private-001", _case(0), admin_id)
    _approve(service, dataset["id"], "private-001", admin_id)
    summary_row = service.list_cases(dataset["id"], include_content=False)[0]
    assert "query" not in summary_row
    assert "expected_document_ids" not in summary_row
    assert "answer_key_points" not in summary_row
    service.put_case(dataset["id"], "private-001", _case(0), admin_id)
    detail = service.get_case(dataset["id"], "private-001")
    assert detail["review_status"] == "pending"
    assert detail["reviewer_id"] is None
    assert detail["attestation_complete"] is False
    service.delete_case(dataset["id"], "private-001")
    with pytest.raises(KeyError):
        service.get_case(dataset["id"], "private-001")


def test_private_approval_requires_complete_attestation(tmp_path) -> None:
    service, _, admin_id = _service(tmp_path)
    dataset = service.create_set("private", "1.0.0", admin_id)
    service.put_case(dataset["id"], "private-001", _case(0), admin_id)
    incomplete = {**_ATTESTATION, "answer_verified": False}

    with pytest.raises(ValueError, match="answer_verified"):
        service.review_case(
            dataset["id"], "private-001", "approved", "incomplete", admin_id,
            attestation=incomplete,
        )
    pending = service.review_case(
        dataset["id"], "private-001", "pending", "continue", admin_id,
        attestation=incomplete,
    )
    assert pending["attestation_complete"] is False
    assert service.get_set(dataset["id"])["export_ready"] is False
