from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.core.settings import load_settings
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import ChatResponse
from src.production.database import ProductionDatabase
from src.production.prompts import PromptRegistry
from src.production.worker import Worker


class FakeJobs:
    def __init__(self):
        self.heartbeats = 0

    def is_cancel_requested(self, job_id):
        return False

    def heartbeat(self, job_id):
        self.heartbeats += 1


class FakeJudge:
    def chat(self, messages, **kwargs):
        assert "Expected key points" in messages[0].content
        assert kwargs["temperature"] == 0.0
        return ChatResponse(
            content=json.dumps({
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "citation_coverage": 1.0,
            }),
            model="fake-local-judge",
        )


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query, top_k):
        return [self.result][:top_k]


class FakeHybrid:
    def __init__(self, result):
        self.result = result
        self.sparse_retriever = FakeRetriever(result)
        self.dense_retriever = FakeRetriever(result)

    def search(self, query, top_k):
        return [self.result][:top_k]


class FakeReranker:
    is_enabled = True

    def rerank(self, query, candidates, top_k):
        return SimpleNamespace(results=candidates[:top_k], used_fallback=False)


class FakeRAG:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.llm = FakeJudge()

    def _pipeline(self, collection):
        return self.pipeline

    def answer(self, query, collection, top_k, user_id=None):
        if "没有" in query:
            return {
                "status": "refused", "answer": "private refusal", "citations": [],
                "evidence": {"citation_coverage": None},
            }
        return {
            "status": "answered", "answer": "private generated answer [1]",
            "citations": [{"document_id": "doc-1", "excerpt": "private excerpt"}],
            "evidence": {"citation_coverage": 1.0},
        }


def test_worker_evaluation_persists_retrieval_and_answer_metrics_without_content(tmp_path, monkeypatch):
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps({
        "name": "worker-test", "version": "1.0.0", "visibility": "public",
        "cases": [
            {
                "query_id": "answerable", "query": "可回答问题", "category": "fact",
                "expected_document_ids": ["doc-1"], "answer_key_points": ["point"],
                "answerable": True, "source": "fixture", "split": "dev",
                "language": "zh", "difficulty": "easy",
            },
            {
                "query_id": "negative", "query": "没有答案的问题", "category": "no-answer",
                "expected_document_ids": [], "answer_key_points": [], "answerable": False,
                "source": "manual-negative", "split": "dev", "language": "zh",
                "difficulty": "easy",
            },
        ],
    }, ensure_ascii=False), encoding="utf-8")
    report_root = tmp_path / "outputs" / "evaluations"
    original_resolve = __import__("src.core.settings", fromlist=["resolve_path"]).resolve_path

    def resolve_path(value):
        if str(value) == "outputs/evaluations":
            return report_root
        return original_resolve(value)

    monkeypatch.setattr("src.core.settings.resolve_path", resolve_path)
    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider="ollama", model="fake-local"))
    result = RetrievalResult(
        chunk_id="chunk-1", score=0.9, text="private retrieved text",
        metadata={"document_id": "doc-1", "source_path": "fixture.pdf"},
    )
    pipeline = SimpleNamespace(hybrid=FakeHybrid(result), reranker=FakeReranker())
    database = ProductionDatabase(tmp_path / "production.db")
    evaluation_id = "eval-1"
    with database.transaction(immediate=True) as connection:
        connection.execute(
            """INSERT INTO evaluations(id,dataset_name,dataset_version,split,status,config_json,created_at)
               VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (evaluation_id, "pending", "pending", "dev", "queued", "{}"),
        )
    runtime = SimpleNamespace(
        settings=settings,
        database=database,
        jobs=FakeJobs(),
        rag=FakeRAG(pipeline),
        prompts=PromptRegistry("config/prompts"),
    )
    result_payload = Worker(runtime)._evaluation({
        "evaluation_id": evaluation_id,
        "dataset_path": str(dataset_path),
        "split": "dev",
        "collection": "test",
        "top_k": 5,
        "repeats": 2,
        "created_by": "admin-1",
    }, "job-1")
    report_path = Path(result_payload["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["answer_quality"]["metrics"] == {
        "correct_refusal_rate": 1.0,
        "false_refusal_rate": 0.0,
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "citation_coverage": 1.0,
    }
    assert report["judge"]["prompt_id"] == "grounded_answer_quality_judge"
    assert report["judge"]["status"] == "completed"
    assert result_payload["unavailable_answer_metrics"] == []
    serialized = report_path.read_text(encoding="utf-8")
    assert "private generated answer" not in serialized
    assert "private excerpt" not in serialized
    assert "private retrieved text" not in serialized
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,dataset_version,metrics_json FROM evaluations WHERE id=?",
            (evaluation_id,),
        ).fetchone()
    assert row["status"] == "completed" and row["dataset_version"] == "1.0.0"
    assert "answer_quality" in json.loads(row["metrics_json"])
