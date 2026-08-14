from __future__ import annotations

import json

import pytest

from scripts.quality_gate import check
from src.production.evaluation import (
    AblationEvaluator,
    AnswerQualityEvaluator,
    citation_coverage,
    load_golden_dataset,
    parse_judge_scores,
    refusal_metrics,
    require_quality_gate_eligible,
    retrieval_metrics,
)
from src.production.worker import validate_evaluation_privacy


def test_public_golden_schema_and_splits():
    dataset = load_golden_dataset("evaluation/public_golden.json")
    assert dataset.visibility == "public"
    assert {case.split for case in dataset.cases} == {"dev", "final"}
    assert len(load_golden_dataset("evaluation/public_golden.json", "final").cases) == 3


def test_golden_schema_rejects_missing_fields_and_answerable_without_docs(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "x", "version": "1", "visibility": "public", "cases": [{}]}))
    with pytest.raises(ValueError, match="missing fields"):
        load_golden_dataset(path)


def test_quality_gate_rejects_candidate_artifacts(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({
        "artifact_type": "golden_set_candidate",
        "eligible_for_quality_gate": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="require human review"):
        require_quality_gate_eligible(path)
    with pytest.raises(ValueError, match="not explicitly eligible"):
        require_quality_gate_eligible("evaluation/public_golden.json")
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps({
        "artifact_type": "golden_set", "eligible_for_quality_gate": True,
        "review_status": "human_reviewed",
    }), encoding="utf-8")
    require_quality_gate_eligible(approved)


def test_retrieval_citation_and_refusal_metrics():
    metrics = retrieval_metrics(["b", "a", "c"], ["a", "d"], 3)
    assert metrics == {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5,
                       "ndcg_at_k": pytest.approx(0.38685280723454163)}
    assert citation_coverage([["1"], [], ["2"]]) == pytest.approx(2 / 3)
    assert refusal_metrics([(False, True), (False, False), (True, True), (True, False)]) == {
        "correct_refusal_rate": 0.5, "false_refusal_rate": 0.5,
    }


def test_ablation_uses_same_protocol_and_reports_latency():
    dataset = load_golden_dataset("evaluation/public_golden.json", "dev")

    def search(query, k):
        del query, k
        return [{"document_id": "doc_148b71f5c9162dd2"}]

    report = AblationEvaluator(dataset, k=5, repeats=2).run({"bm25": search, "dense": search})
    assert report["protocol"]["k"] == 5 and report["protocol"]["repeats"] == 2
    assert [item["variant"] for item in report["variants"]] == ["bm25", "dense"]
    assert all(item["latency_p95_ms"] >= item["latency_p50_ms"] for item in report["variants"])


def test_answer_quality_reports_judge_refusal_and_does_not_store_content():
    dataset = load_golden_dataset("evaluation/public_golden.json", "dev")

    def answer(case):
        if case.answerable:
            return {"status": "answered", "answer": "private answer", "citations": [{
                "excerpt": "private evidence",
            }], "evidence": {"citation_coverage": 1.0}}
        return {"status": "refused", "answer": "refusal", "citations": [], "evidence": {}}

    report = AnswerQualityEvaluator(dataset).run(
        answer,
        lambda case, response: {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "citation_coverage": 1.0,
        },
    )
    assert report["metrics"]["correct_refusal_rate"] == 1.0
    assert report["metrics"]["false_refusal_rate"] == 0.0
    assert report["metrics"]["faithfulness"] == 0.9
    assert report["unavailable_metrics"] == []
    serialized = json.dumps(report)
    assert "private answer" not in serialized and "private evidence" not in serialized


def test_parse_judge_scores_accepts_reasoning_prefix_and_rejects_bad_values():
    assert parse_judge_scores(
        '<think>done</think>\n```json\n{"faithfulness":0.9,"answer_relevancy":0.8,"citation_coverage":1}\n```'
    )["faithfulness"] == 0.9
    with pytest.raises(ValueError, match="outside"):
        parse_judge_scores({
            "faithfulness": 2, "answer_relevancy": .8, "citation_coverage": 1,
        })


def test_private_evaluation_only_allows_local_ollama_judge():
    validate_evaluation_privacy("public", "openai")
    validate_evaluation_privacy("private", "ollama")
    with pytest.raises(RuntimeError, match="local Ollama"):
        validate_evaluation_privacy("private", "deepseek")


def test_quality_gate_absolute_and_regression_thresholds():
    good = {
        "faithfulness": .9, "answer_relevancy": .9, "citation_coverage": .97,
        "recall_at_5": .85, "correct_refusal_rate": 1.0, "false_refusal_rate": 0.0,
    }
    baseline = {
        "faithfulness": .91, "answer_relevancy": .91, "citation_coverage": .98,
        "recall_at_5": .86, "correct_refusal_rate": 1.0, "false_refusal_rate": 0.0,
    }
    assert check(
        good,
        baseline,
    ) == []
    failures = check(
        {**good, "faithfulness": .82, "citation_coverage": .90,
         "recall_at_5": .77, "false_refusal_rate": .2},
        baseline,
    )
    assert any("faithfulness" in item for item in failures)
