from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_public_golden_candidates as candidate_module
from scripts.build_public_golden_candidates import (
    SOURCES,
    CandidateBuilder,
    _download_verified,
    _stable_rows,
)


def test_deterministic_selection_does_not_depend_on_input_order():
    frame = pd.DataFrame([{"id": str(index)} for index in range(20)])
    forward = [row["id"] for row in _stable_rows(frame, 5, salt="fixed")]
    reverse = [row["id"] for row in _stable_rows(frame.iloc[::-1], 5, salt="fixed")]
    assert forward == reverse


def test_deterministic_selection_supports_sources_without_an_id_column():
    frame = pd.DataFrame([{"query": f"question-{index}"} for index in range(20)])
    forward = [row["query"] for row in _stable_rows(frame, 5, salt="fixed")]
    reverse = [row["query"] for row in _stable_rows(frame.iloc[::-1], 5, salt="fixed")]
    assert forward == reverse


def test_candidate_validation_rejects_unreviewed_pack_as_formal_gate_input(tmp_path):
    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")
    stale = builder.corpus_root / "old-source" / "stale.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale generated document", encoding="utf-8")
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
    assert not stale.exists()


def test_all_upstream_licenses_are_explicitly_allowlisted():
    assert {source["license"] for source in SOURCES.values()} == {
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "odc-by-1.0",
    }
    assert SOURCES["multihop_rag"]["attribution"]
    assert SOURCES["multihop_rag"]["revision"] == (
        "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
    )
    assert SOURCES["xquad_zh"]["config"] == "xquad.zh"
    for source in SOURCES.values():
        assert source["attribution"]
        assert source["license_url"].startswith("https://")
        for artifact in [source["dataset_card"], *source["artifacts"]]:
            assert source["revision"] in artifact["url"]
            assert len(artifact["sha256"]) == 64
            assert artifact["size_bytes"] > 0


class _DownloadResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_verified_download_accepts_only_the_pinned_payload(monkeypatch):
    payload = b"verified public fixture"
    artifact = {
        "url": "https://huggingface.co/datasets/example/data/resolve/abc/data.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    monkeypatch.setattr(
        candidate_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _DownloadResponse(payload),
    )

    assert _download_verified(artifact) == payload


def test_verified_download_fails_closed_on_content_drift(monkeypatch):
    expected = b"expected"
    actual = b"tampered"
    artifact = {
        "url": "https://huggingface.co/datasets/example/data/resolve/abc/data.json",
        "sha256": hashlib.sha256(expected).hexdigest(),
        "size_bytes": len(actual),
    }
    monkeypatch.setattr(
        candidate_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _DownloadResponse(actual),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _download_verified(artifact)


def test_verified_download_rejects_non_huggingface_hosts():
    artifact = {
        "url": "https://example.com/data.json",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    with pytest.raises(ValueError, match="untrusted candidate artifact URL"):
        _download_verified(artifact)


def test_candidate_pack_requires_exactly_one_hundred_cases(tmp_path):
    builder = CandidateBuilder(Path(tmp_path))
    with pytest.raises(ValueError, match="must contain 100 cases"):
        builder.write()


def test_multihop_candidates_preserve_upstream_types_and_evidence_mapping(tmp_path):
    corpus = pd.DataFrame([{
        "title": "Evidence document",
        "author": "Author",
        "source": "Publisher",
        "published_at": "2024-02-03T00:00:00+00:00",
        "category": "technology",
        "url": "https://example.test/evidence",
        "body": "The complete source document.",
    }])
    rows = []
    for question_type in (
        "inference_query", "comparison_query", "temporal_query", "null_query",
    ):
        for index in range(6):
            answerable = question_type != "null_query"
            rows.append({
                "query": f"{question_type} question {index}",
                "answer": "Supported answer" if answerable else "Insufficient information.",
                "question_type": question_type,
                "evidence_list": ([{
                    "url": "https://example.test/evidence",
                    "fact": "Supporting fact",
                }] if answerable else []),
            })

    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")
    builder.add_multihop(pd.DataFrame(rows), corpus)

    assert len(builder.cases) == 24
    assert {case["category"] for case in builder.cases} == {
        "多文档组合", "跨文档比较", "时间条件", "无答案",
    }
    assert {
        case["candidate_metadata"]["upstream_question_type"]
        for case in builder.cases
    } == {
        "inference_query", "comparison_query", "temporal_query", "null_query",
    }
    assert all(
        case["candidate_metadata"]["category_assignment"] == "upstream_question_type"
        for case in builder.cases
    )
    null_cases = [case for case in builder.cases if not case["answerable"]]
    assert len(null_cases) == 6
    assert all(not case["expected_document_ids"] for case in null_cases)
    assert all(not case["answer_key_points"] for case in null_cases)
    assert all(case["source"] == "manual-negative" for case in null_cases)
    known_document_ids = set(builder.documents)
    assert all(
        set(case["expected_document_ids"]) <= known_document_ids
        for case in builder.cases
    )


def test_multihop_candidate_rejects_evidence_missing_from_corpus(tmp_path):
    questions = pd.DataFrame([{
        "query": f"inference question {index}",
        "answer": "answer",
        "question_type": "inference_query",
        "evidence_list": [{"url": "https://example.test/missing"}],
    } for index in range(6)])
    corpus = pd.DataFrame([{
        "title": "Other",
        "url": "https://example.test/other",
        "body": "Other document",
    }])
    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")

    with pytest.raises(ValueError, match="evidence URL is absent from corpus"):
        builder.add_multihop(questions, corpus)


def test_ragbench_finqa_maps_sentence_keys_to_expected_documents(tmp_path):
    frame = pd.DataFrame([{
        "id": f"finqa-{index}",
        "question": f"Calculate metric {index}",
        "documents": ["Narrative", '[["year", "value"], ["2024", "42"]]', "Footer"],
        "response": "The result is 42.",
        "generation_model_name": "gpt-3.5-turbo-0125",
        "adherence_score": True,
        "unsupported_response_sentence_keys": [],
        "all_relevant_sentence_keys": ["1a", "2b"],
        "completeness_score": 1.0,
    } for index in range(8)])
    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")

    builder.add_ragbench_finqa(frame)

    assert len(builder.cases) == 8
    assert all(case["category"] == "表格" for case in builder.cases)
    assert all(len(case["expected_document_ids"]) == 2 for case in builder.cases)
    assert all(
        case["candidate_metadata"]["category_assignment"]
        == "upstream_dataset_config"
        for case in builder.cases
    )
    assert all(
        case["candidate_metadata"]["answer_key_origin"]
        == "model_response_requires_human_review"
        for case in builder.cases
    )


def test_ragbench_finqa_rejects_out_of_range_evidence_index(tmp_path):
    frame = pd.DataFrame([{
        "id": f"finqa-{index}",
        "question": f"Calculate metric {index}",
        "documents": ["Only document"],
        "response": "Answer",
        "generation_model_name": "gpt-3.5-turbo-0125",
        "adherence_score": True,
        "unsupported_response_sentence_keys": [],
        "all_relevant_sentence_keys": ["2a"],
        "completeness_score": 1.0,
    } for index in range(8)])
    builder = CandidateBuilder(tmp_path / "evaluation" / "candidates")

    with pytest.raises(ValueError, match="evidence index is out of range"):
        builder.add_ragbench_finqa(frame)


def test_checked_in_candidate_pack_has_expected_coverage_and_no_stale_corpus():
    repository_root = Path(__file__).resolve().parents[2]
    candidate_path = (
        repository_root / "evaluation" / "candidates" / "public_golden_candidate.json"
    )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert payload["version"] == "2026.09.02.3"
    assert payload["eligible_for_quality_gate"] is False
    assert payload["review_status"] == "requires_human_review"
    assert len(payload["cases"]) == 100
    assert {case["category"] for case in payload["cases"]} >= {
        "精确事实", "无答案", "多文档组合", "跨文档比较", "时间条件", "表格",
    }
    assert all(
        case["candidate_metadata"]["human_reviewed"] is False
        for case in payload["cases"]
    )

    sources = {source["dataset"]: source for source in payload["sources"]}
    xquad = sources["google/xquad"]
    assert xquad["config"] == "xquad.zh"
    assert xquad["license"] == "cc-by-sa-4.0"
    assert xquad["artifacts"][0]["sha256"] == (
        "6d9033e82b2a1dbf67b66025e84fab9a44e13dc91401e4914a9529db784146f6"
    )
    multihop = sources["yixuantt/MultiHopRAG"]
    assert multihop["revision"] == "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
    assert multihop["license"] == "odc-by-1.0"
    assert multihop["attribution"]
    assert {artifact["role"] for artifact in multihop["artifacts"]} == {
        "questions", "corpus",
    }
    ragbench = sources["galileo-ai/ragbench"]
    assert ragbench["config"] == "finqa"
    assert ragbench["revision"] == "97808f3e5fd16ede40bbff6c2949af8139b2eb7b"
    assert ragbench["license"] == "cc-by-4.0"
    assert ragbench["artifacts"][0]["sha256"] == (
        "b55bd373741fbe65933191690267421183f898c8253f5db2a4a4703360ecf94d"
    )

    manifest_paths = {repository_root / item["path"] for item in payload["corpus"]}
    actual_paths = set(
        (repository_root / "evaluation" / "candidates" / "public_corpus").rglob("*.md")
    )
    assert manifest_paths == actual_paths
    known_document_ids = {item["document_id"] for item in payload["corpus"]}
    assert all(
        set(case["expected_document_ids"]) <= known_document_ids
        for case in payload["cases"]
    )
