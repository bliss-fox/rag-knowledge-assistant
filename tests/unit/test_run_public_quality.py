from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_public_quality import collect_public_sources
from src.production.evaluation import load_golden_dataset


def test_collect_public_sources_includes_all_multihop_and_unanswerable_context(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("scripts.run_public_quality.resolve_path", lambda value: (
        tmp_path if value == "." else Path(value)
    ))
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    negative = tmp_path / "negative.md"
    for path in (first, second, negative):
        path.write_text(path.stem, encoding="utf-8")
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps({
        "name": "public", "version": "1", "visibility": "public",
        "corpus": [
            {"document_id": "doc1", "path": str(first)},
            {"document_id": "doc2", "path": str(second)},
            {"document_id": "doc3", "path": str(negative)},
        ],
        "cases": [
            {
                "query_id": "multi", "query": "multi", "category": "multi",
                "expected_document_ids": ["doc1", "doc2"], "answer_key_points": ["x"],
                "answerable": True, "source": str(first), "split": "final",
                "language": "en", "difficulty": "hard",
            },
            {
                "query_id": "negative", "query": "negative", "category": "unanswerable",
                "expected_document_ids": [], "answer_key_points": [], "answerable": False,
                "source": str(negative), "split": "final", "language": "en",
                "difficulty": "hard",
            },
        ],
    }), encoding="utf-8")
    dataset = load_golden_dataset(dataset_path, "final")
    assert collect_public_sources(dataset_path, dataset) == sorted([first, second, negative])


def test_collect_public_sources_rejects_missing_multihop_document(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_public_quality.resolve_path", lambda value: (
        tmp_path if value == "." else Path(value)
    ))
    source = tmp_path / "source.md"
    source.write_text("source", encoding="utf-8")
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps({
        "name": "public", "version": "1", "visibility": "public", "corpus": [],
        "cases": [{
            "query_id": "q", "query": "q", "category": "multi",
            "expected_document_ids": ["doc1", "doc2"], "answer_key_points": ["x"],
            "answerable": True, "source": str(source), "split": "final",
            "language": "en", "difficulty": "hard",
        }],
    }), encoding="utf-8")
    dataset = load_golden_dataset(dataset_path, "final")
    # Without a corpus manifest, the legacy single-source format cannot satisfy multi-hop labels.
    with pytest.raises(ValueError, match="legacy public dataset"):
        collect_public_sources(dataset_path, dataset)
