from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.production.auth import AuthService
from src.production.database import ProductionDatabase
from src.production.golden_review import GoldenReviewService


def _candidate(root: Path) -> tuple[Path, str]:
    corpus = root / "evaluation" / "candidates" / "public_corpus"
    corpus.mkdir(parents=True)
    content = b"# Evidence\n\nThe supported answer is 60.\n"
    document_id = "doc_" + hashlib.sha256(content).hexdigest()[:16]
    document = corpus / "doc.md"
    document.write_bytes(content)
    payload = {
        "artifact_type": "golden_set_candidate", "name": "candidate", "version": "1",
        "eligible_for_quality_gate": False,
        "corpus": [{"document_id": document_id,
                    "path": document.relative_to(root).as_posix(), "title": "Evidence"}],
        "cases": [{
            "query_id": "q-1", "query": "What is supported?", "category": "fact",
            "expected_document_ids": [document_id], "answer_key_points": ["60"],
            "answerable": True, "source": document.relative_to(root).as_posix(),
            "split": "dev", "language": "en", "difficulty": "easy",
        }],
    }
    path = root / "evaluation" / "candidates" / "public_golden_candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, document_id


def test_review_is_bound_to_candidate_hash_and_exposes_evidence(tmp_path):
    path, document_id = _candidate(tmp_path)
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("review.admin", "correct horse battery staple")
    service = GoldenReviewService(database, tmp_path)
    summary = service.summary(str(path.relative_to(tmp_path)))
    assert summary["counts"] == {"pending": 0, "approved": 0, "rejected": 0, "unreviewed": 1}
    detail = service.get_case("q-1", str(path.relative_to(tmp_path)))
    assert detail["evidence"][0]["document_id"] == document_id
    assert "supported answer" in detail["evidence"][0]["content"]

    service.review(
        "q-1", "approved", "verified", admin.user_id, summary["dataset_sha256"],
        str(path.relative_to(tmp_path)),
    )
    approved = service.summary(str(path.relative_to(tmp_path)))
    assert approved["all_approved"] is True
    assert approved["counts"]["approved"] == 1

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate file changed"):
        service.review(
            "q-1", "approved", "stale", admin.user_id, summary["dataset_sha256"],
            str(path.relative_to(tmp_path)),
        )


def test_review_path_cannot_escape_repository(tmp_path):
    database = ProductionDatabase(tmp_path / "production.db")
    service = GoldenReviewService(database, tmp_path)
    outside = tmp_path.parent / "outside-candidate.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="outside repository"):
        service.summary(str(outside))


def test_reviewed_export_requires_all_approved_and_remains_gate_ineligible(tmp_path):
    path, _ = _candidate(tmp_path)
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("review.admin", "correct horse battery staple")
    service = GoldenReviewService(database, tmp_path)
    relative = str(path.relative_to(tmp_path))
    digest = service.summary(relative)["dataset_sha256"]

    with pytest.raises(ValueError, match="1 remain"):
        service.export_reviewed(digest, admin.user_id, relative)

    service.review("q-1", "approved", "evidence verified", admin.user_id, digest, relative)
    exported = service.export_reviewed(digest, admin.user_id, relative)
    output = tmp_path / exported["path"]
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exported["case_count"] == 1
    assert exported["eligible_for_quality_gate"] is False
    assert payload["artifact_type"] == "golden_set_reviewed_candidate"
    assert payload["source_candidate"]["sha256"] == digest
    assert payload["human_review"]["approved_case_count"] == 1
    assert payload["human_review"]["records"][0]["status"] == "approved"
    assert payload["eligible_for_quality_gate"] is False


def test_reviewed_export_rejects_stale_candidate_hash(tmp_path):
    path, _ = _candidate(tmp_path)
    database = ProductionDatabase(tmp_path / "production.db")
    auth = AuthService(database, secret="x" * 64)
    admin = auth.bootstrap_admin("review.admin", "correct horse battery staple")
    service = GoldenReviewService(database, tmp_path)
    digest = service.summary(str(path.relative_to(tmp_path)))["dataset_sha256"]
    service.review(
        "q-1", "approved", "verified", admin.user_id, digest,
        str(path.relative_to(tmp_path)),
    )
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate file changed"):
        service.export_reviewed(digest, admin.user_id, str(path.relative_to(tmp_path)))
