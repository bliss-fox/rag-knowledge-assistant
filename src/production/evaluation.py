"""Versioned golden-set validation and reproducible offline evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.production.trace_store import percentile


REQUIRED_CASE_FIELDS = {
    "query_id", "query", "category", "expected_document_ids", "answer_key_points",
    "answerable", "source", "split", "language", "difficulty",
}
ALLOWED_SPLITS = {"dev", "final"}


@dataclass(frozen=True)
class GoldenCase:
    query_id: str
    query: str
    category: str
    expected_document_ids: tuple[str, ...]
    answer_key_points: tuple[str, ...]
    answerable: bool
    source: str
    split: str
    language: str
    difficulty: str


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    version: str
    visibility: str
    cases: tuple[GoldenCase, ...]
    sha256: str


@dataclass
class VariantReport:
    variant: str
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    failures: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)


ANSWER_SCORE_FIELDS = ("faithfulness", "answer_relevancy", "citation_coverage")


def load_golden_dataset(path: str | Path, split: str | None = None) -> GoldenDataset:
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    for field_name in ("name", "version", "visibility", "cases"):
        if field_name not in data:
            raise ValueError(f"{path}: missing dataset field {field_name}")
    if data["visibility"] not in {"public", "private"}:
        raise ValueError("visibility must be public or private")
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for index, item in enumerate(data["cases"]):
        missing = REQUIRED_CASE_FIELDS - item.keys()
        if missing:
            raise ValueError(f"case[{index}] missing fields: {sorted(missing)}")
        if item["query_id"] in seen:
            raise ValueError(f"duplicate query_id: {item['query_id']}")
        if item["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"invalid split: {item['split']}")
        if bool(item["answerable"]) and not item["expected_document_ids"]:
            raise ValueError(f"{item['query_id']}: answerable case needs expected_document_ids")
        seen.add(str(item["query_id"]))
        case = GoldenCase(
            query_id=str(item["query_id"]), query=str(item["query"]),
            category=str(item["category"]),
            expected_document_ids=tuple(str(value) for value in item["expected_document_ids"]),
            answer_key_points=tuple(str(value) for value in item["answer_key_points"]),
            answerable=bool(item["answerable"]), source=str(item["source"]),
            split=str(item["split"]), language=str(item["language"]),
            difficulty=str(item["difficulty"]),
        )
        if split is None or case.split == split:
            cases.append(case)
    if split is not None and split not in ALLOWED_SPLITS:
        raise ValueError(f"invalid split: {split}")
    if not cases:
        raise ValueError("golden dataset selection is empty")
    return GoldenDataset(str(data["name"]), str(data["version"]), str(data["visibility"]),
                         tuple(cases), hashlib.sha256(raw).hexdigest())


def require_quality_gate_eligible(path: str | Path) -> None:
    """Allow only datasets that explicitly record completed human review."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("artifact_type") == "golden_set_candidate":
        raise ValueError("golden-set candidates require human review before quality-gate use")
    if data.get("eligible_for_quality_gate") is not True:
        raise ValueError("dataset is not explicitly eligible for quality-gate use")
    if data.get("review_status") != "human_reviewed":
        raise ValueError("dataset does not record completed human review")


def retrieval_metrics(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(retrieved)[:k]
    expected = set(relevant)
    if not expected:
        return {"hit_at_k": 1.0 if not ranked else 0.0, "recall_at_k": 1.0 if not ranked else 0.0,
                "mrr": 0.0, "ndcg_at_k": 1.0 if not ranked else 0.0}
    hits = [1 if item in expected else 0 for item in ranked]
    first = next((index for index, value in enumerate(hits, 1) if value), None)
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(hits, 1))
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(expected), k) + 1))
    return {
        "hit_at_k": float(any(hits)),
        "recall_at_k": len(set(ranked) & expected) / len(expected),
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def citation_coverage(claim_citations: Iterable[Iterable[str]]) -> float:
    claims = [set(item) for item in claim_citations]
    return sum(bool(item) for item in claims) / len(claims) if claims else 1.0


def refusal_metrics(rows: Iterable[tuple[bool, bool]]) -> dict[str, float]:
    values = list(rows)
    unanswerable = [refused for answerable, refused in values if not answerable]
    answerable = [refused for can_answer, refused in values if can_answer]
    return {
        "correct_refusal_rate": sum(unanswerable) / len(unanswerable) if unanswerable else 0.0,
        "false_refusal_rate": sum(answerable) / len(answerable) if answerable else 0.0,
    }


def parse_judge_scores(value: str | dict[str, Any]) -> dict[str, float]:
    """Parse one strict judge result, tolerating fenced or reasoning-prefixed JSON."""
    if isinstance(value, dict):
        payload = value
    else:
        payload = None
        decoder = json.JSONDecoder()
        for index, character in enumerate(value):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(value[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and all(name in candidate for name in ANSWER_SCORE_FIELDS):
                payload = candidate
                break
        if payload is None:
            raise ValueError("Judge response does not contain the required JSON object")
    scores = {name: float(payload[name]) for name in ANSWER_SCORE_FIELDS}
    invalid = {name: score for name, score in scores.items() if not 0.0 <= score <= 1.0}
    if invalid:
        raise ValueError(f"Judge returned scores outside [0, 1]: {invalid}")
    return scores


class AnswerQualityEvaluator:
    """Evaluate generated answers without persisting questions, answers or excerpts."""

    def __init__(self, dataset: GoldenDataset) -> None:
        self.dataset = dataset

    def run(
        self,
        answer: Callable[[GoldenCase], dict[str, Any]],
        judge: Callable[[GoldenCase, dict[str, Any]], dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        refusal_rows: list[tuple[bool, bool]] = []
        latencies: list[float] = []
        scores: dict[str, list[float]] = {name: [] for name in ANSWER_SCORE_FIELDS}
        judge_errors: list[dict[str, str]] = []
        status_counts: dict[str, int] = {}
        for case in self.dataset.cases:
            started = time.perf_counter()
            try:
                response = answer(case)
            except Exception as exc:
                response = {"status": "error", "evidence": {}, "citations": []}
                answer_error = type(exc).__name__
            else:
                answer_error = None
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            status = str(response.get("status", "error"))
            status_counts[status] = status_counts.get(status, 0) + 1
            refused = status == "refused"
            refusal_rows.append((case.answerable, refused))
            row: dict[str, Any] = {
                "query_id": case.query_id,
                "category": case.category,
                "answerable": case.answerable,
                "status": status,
                "latency_ms": elapsed,
                "error": answer_error,
            }
            if case.answerable:
                if refused or status == "error":
                    item_scores = {name: 0.0 for name in ANSWER_SCORE_FIELDS}
                elif judge is None:
                    coverage = response.get("evidence", {}).get("citation_coverage")
                    item_scores = ({"citation_coverage": float(coverage)}
                                   if coverage is not None else {})
                else:
                    try:
                        item_scores = parse_judge_scores(judge(case, response))
                    except Exception as exc:
                        item_scores = {}
                        judge_errors.append({
                            "query_id": case.query_id,
                            "error": type(exc).__name__,
                        })
                row["scores"] = item_scores
                for name, value in item_scores.items():
                    scores[name].append(float(value))
            rows.append(row)
        metrics: dict[str, float] = {
            **refusal_metrics(refusal_rows),
            **{name: _mean(values) for name, values in scores.items() if values},
        }
        if judge_errors:
            for name in ANSWER_SCORE_FIELDS:
                metrics.pop(name, None)
        unavailable = [name for name, values in scores.items() if not values]
        if judge_errors:
            unavailable = list(ANSWER_SCORE_FIELDS)
        return {
            "metrics": metrics,
            "unavailable_metrics": unavailable,
            "judge_errors": judge_errors,
            "status_counts": status_counts,
            "answerable_count": sum(case.answerable for case in self.dataset.cases),
            "judged_count": sum(1 for row in rows if row.get("scores")),
            "latency_p50_ms": percentile(latencies, .5),
            "latency_p95_ms": percentile(latencies, .95),
            "cases": rows,
        }


class AblationEvaluator:
    """Compare retrieval variants using identical cases, k, and repetition count."""

    def __init__(self, dataset: GoldenDataset, k: int = 5, repeats: int = 5) -> None:
        if k < 1 or repeats < 1:
            raise ValueError("k and repeats must be positive")
        self.dataset = dataset
        self.k = k
        self.repeats = repeats

    def run(self, variants: dict[str, Callable[[str, int], list[dict[str, Any]]]]) -> dict[str, Any]:
        reports: list[VariantReport] = []
        for name, search in variants.items():
            rows: list[dict[str, Any]] = []
            latencies: list[float] = []
            for case in self.dataset.cases:
                result: list[dict[str, Any]] = []
                error: str | None = None
                samples: list[float] = []
                for _ in range(self.repeats):
                    started = time.perf_counter()
                    try:
                        result = search(case.query, self.k)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        result = []
                    samples.append((time.perf_counter() - started) * 1000)
                latencies.extend(samples)
                document_ids = [str(item.get("document_id") or item.get("source") or "") for item in result]
                metrics = retrieval_metrics(document_ids, case.expected_document_ids, self.k)
                rows.append({"query_id": case.query_id, "category": case.category,
                             "document_ids": document_ids, "latency_ms": samples, "error": error, **metrics})
            reports.append(VariantReport(
                variant=name,
                hit_at_k=_mean(row["hit_at_k"] for row in rows),
                recall_at_k=_mean(row["recall_at_k"] for row in rows),
                mrr=_mean(row["mrr"] for row in rows),
                ndcg_at_k=_mean(row["ndcg_at_k"] for row in rows),
                latency_p50_ms=percentile(latencies, .5), latency_p95_ms=percentile(latencies, .95),
                failures=[{"query_id": row["query_id"], "category": row["category"],
                           "error": row["error"] or "retrieval_miss"}
                          for row in rows if row["error"] or not row["hit_at_k"]],
                queries=rows,
            ))
        return {
            "dataset": {"name": self.dataset.name, "version": self.dataset.version,
                        "sha256": self.dataset.sha256, "query_count": len(self.dataset.cases)},
            "protocol": {"k": self.k, "repeats": self.repeats, "python": sys.version,
                         "platform": platform.platform()},
            "variants": [asdict(report) for report in reports],
        }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
