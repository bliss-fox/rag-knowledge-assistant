"""Structured trace persistence, redaction, retention and percentile metrics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from src.production.database import ProductionDatabase


_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|authorization|password|token)(\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
]

_CONTENT_KEYS = {
    "answer", "claim", "content", "context", "excerpt", "preview", "prompt",
    "query", "question", "response", "sentence", "summary", "text", "chunks",
}
_CONTENT_SUFFIXES = ("_text", "_preview", "_excerpt", "_content")


def redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    result = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            result = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class SQLiteTraceStore:
    def __init__(self, database: ProductionDatabase, retention_days: int = 30) -> None:
        self.database = database
        self.retention_days = retention_days

    def save(self, trace: dict[str, Any]) -> None:
        trace_id = trace["trace_id"]
        metadata = dict(trace.get("metadata", {}))
        query = redact_text(str(metadata.pop("query", ""))) or None
        response = redact_text(metadata.pop("response", None))
        prompt = redact_text(metadata.pop("prompt", None))
        usage = metadata.pop("usage", {}) or {}
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO traces(
                   id,trace_type,user_id,status,started_at,finished_at,total_elapsed_ms,
                   query_hash,query_text,response_text,prompt_text,prompt_id,prompt_version,prompt_hash,
                   model_version,dataset_version,config_version,retrieval_method,input_tokens,output_tokens,
                   citation_coverage,faithfulness,error_type,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (trace_id, trace.get("trace_type", "query"), metadata.pop("user_id", None),
                 metadata.pop("status", "succeeded"), trace.get("started_at"), trace.get("finished_at"),
                 float(trace.get("total_elapsed_ms", 0)), hashlib.sha256((query or "").encode()).hexdigest(),
                 query, response, prompt, metadata.pop("prompt_id", None), metadata.pop("prompt_version", None),
                 metadata.pop("prompt_hash", None), metadata.pop("model_version", None),
                 metadata.pop("dataset_version", None), metadata.pop("config_version", None),
                 metadata.pop("retrieval_method", None), int(usage.get("prompt_tokens", 0)),
                 int(usage.get("completion_tokens", 0)), metadata.pop("citation_coverage", None),
                 metadata.pop("faithfulness", None), metadata.pop("error_type", None),
                 json.dumps(metadata, ensure_ascii=False, default=str)),
            )
            connection.execute("DELETE FROM trace_stages WHERE trace_id=?", (trace_id,))
            for ordinal, stage in enumerate(trace.get("stages", [])):
                data = self._redact(stage.get("data", {}))
                connection.execute(
                    "INSERT INTO trace_stages(trace_id,ordinal,stage,elapsed_ms,data_json,created_at) VALUES (?,?,?,?,?,?)",
                    (trace_id, ordinal, stage.get("stage", "unknown"), stage.get("elapsed_ms"),
                     json.dumps(data, ensure_ascii=False, default=str), stage.get("timestamp") or trace.get("started_at")),
                )

    def get(self, trace_id: str, user_id: str | None = None, is_admin: bool = False) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
            if row is None or (not is_admin and row["user_id"] != user_id):
                return None
            stages = connection.execute(
                "SELECT stage,elapsed_ms,data_json,created_at FROM trace_stages WHERE trace_id=? ORDER BY ordinal",
                (trace_id,),
            ).fetchall()
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["stages"] = [
            {"stage": item["stage"], "elapsed_ms": item["elapsed_ms"],
             "data": json.loads(item["data_json"]), "timestamp": item["created_at"]}
            for item in stages
        ]
        return result

    def list(
        self,
        user_id: str | None,
        is_admin: bool,
        limit: int = 100,
        *,
        start: str | None = None,
        end: str | None = None,
        prompt_version: str | None = None,
        model_version: str | None = None,
        dataset_version: str | None = None,
        config_version: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM traces"
        params: list[Any] = []
        clauses: list[str] = []
        if not is_admin:
            clauses.append("user_id=?")
            params.append(user_id)
        for column, value in (
            ("started_at>=?", start), ("started_at<?", end),
            ("prompt_version=?", prompt_version), ("model_version=?", model_version),
            ("dataset_version=?", dataset_version), ("config_version=?", config_version),
        ):
            if value:
                clauses.append(column)
                params.append(value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(min(max(limit, 1), 1000))
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def metrics(self, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        clauses = ["trace_type='query'"]
        params: list[Any] = []
        if start:
            clauses.append("started_at>=?")
            params.append(start)
        if end:
            clauses.append("started_at<?")
            params.append(end)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT status,total_elapsed_ms,input_tokens,output_tokens,citation_coverage,faithfulness,error_type FROM traces WHERE " + " AND ".join(clauses),
                params,
            ).fetchall()
            stage_rows = connection.execute(
                """SELECT stage,elapsed_ms FROM trace_stages ts JOIN traces t ON t.id=ts.trace_id
                   WHERE """ + " AND ".join(
                       clause if clause.startswith("t.") else f"t.{clause}" for clause in clauses
                   ) + " AND elapsed_ms IS NOT NULL",
                params,
            ).fetchall()
        count = len(rows)
        statuses: dict[str, int] = {}
        errors: dict[str, int] = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
            if row["error_type"]:
                errors[row["error_type"]] = errors.get(row["error_type"], 0) + 1
        by_stage: dict[str, list[float]] = {}
        for row in stage_rows:
            by_stage.setdefault(row["stage"], []).append(row["elapsed_ms"])
        return {
            "request_count": count,
            "status_counts": statuses,
            "failure_rate": (statuses.get("error", 0) / count) if count else 0.0,
            "refusal_rate": (statuses.get("refused", 0) / count) if count else 0.0,
            "degraded_rate": (statuses.get("degraded", 0) / count) if count else 0.0,
            "latency_ms": {"p50": percentile((r["total_elapsed_ms"] for r in rows), .5),
                           "p95": percentile((r["total_elapsed_ms"] for r in rows), .95)},
            "stage_latency_ms": {name: {"p50": percentile(values, .5), "p95": percentile(values, .95)} for name, values in by_stage.items()},
            "tokens": {"input": sum(r["input_tokens"] for r in rows), "output": sum(r["output_tokens"] for r in rows)},
            "citation_coverage": self._average(r["citation_coverage"] for r in rows),
            "faithfulness": self._average(r["faithfulness"] for r in rows),
            "error_counts": errors,
        }

    def time_series(self, start: str, end: str, bucket: str = "hour") -> list[dict[str, Any]]:
        """Return query metrics grouped into stable UTC buckets for dashboards.

        Aggregation is intentionally performed in Python because SQLite's date
        functions handle mixed ISO-8601 offsets inconsistently.  Only compact
        metric columns are loaded; retained prompt/query/document prose is never
        exposed through this endpoint.
        """
        if bucket not in {"hour", "day"}:
            raise ValueError("bucket must be 'hour' or 'day'")
        start_at = self._parse_timestamp(start)
        end_at = self._parse_timestamp(end)
        if end_at <= start_at:
            raise ValueError("end must be after start")
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT started_at,status,total_elapsed_ms,input_tokens,output_tokens,
                          citation_coverage,faithfulness,error_type
                   FROM traces WHERE trace_type='query' AND started_at>=? AND started_at<?
                   ORDER BY started_at""",
                (start, end),
            ).fetchall()
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            timestamp = self._parse_timestamp(row["started_at"])
            if bucket == "hour":
                timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
            else:
                timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            grouped.setdefault(timestamp.isoformat(), []).append(row)
        result: list[dict[str, Any]] = []
        for timestamp, values in sorted(grouped.items()):
            count = len(values)
            statuses: dict[str, int] = {}
            errors: dict[str, int] = {}
            for row in values:
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
                if row["error_type"]:
                    errors[row["error_type"]] = errors.get(row["error_type"], 0) + 1
            result.append({
                "bucket": timestamp,
                "request_count": count,
                "success_rate": 1.0 - (statuses.get("error", 0) / count),
                "failure_rate": statuses.get("error", 0) / count,
                "refusal_rate": statuses.get("refused", 0) / count,
                "degraded_rate": statuses.get("degraded", 0) / count,
                "latency_p50_ms": percentile((row["total_elapsed_ms"] for row in values), .5),
                "latency_p95_ms": percentile((row["total_elapsed_ms"] for row in values), .95),
                "input_tokens": sum(row["input_tokens"] for row in values),
                "output_tokens": sum(row["output_tokens"] for row in values),
                "citation_coverage": self._average(row["citation_coverage"] for row in values),
                "faithfulness": self._average(row["faithfulness"] for row in values),
                "error_counts": errors,
            })
        return result

    def compare_windows(self, start: str, end: str, baseline_start: str, baseline_end: str) -> dict[str, Any]:
        current = self.metrics(start, end)
        baseline = self.metrics(baseline_start, baseline_end)
        delta = {
            "failure_rate": current["failure_rate"] - baseline["failure_rate"],
            "refusal_rate": current["refusal_rate"] - baseline["refusal_rate"],
            "p95_latency_ms": current["latency_ms"]["p95"] - baseline["latency_ms"]["p95"],
            "citation_coverage": (current["citation_coverage"] or 0) - (baseline["citation_coverage"] or 0),
            "faithfulness": (current["faithfulness"] or 0) - (baseline["faithfulness"] or 0),
        }
        versions = {
            "current": self._version_breakdown(start, end),
            "baseline": self._version_breakdown(baseline_start, baseline_end),
        }
        deployment_events = self._deployment_events(baseline_start, end)
        return {
            "current": current,
            "baseline": baseline,
            "delta": delta,
            "versions": versions,
            "deployment_events": deployment_events,
            "root_cause_candidates": self._root_cause_candidates(
                current, baseline, delta, versions, deployment_events, start, end,
            ),
            "causality_notice": (
                "These are correlated signals for investigation, not proof of causation. "
                "Confirm candidates in individual traces and deployment records."
            ),
        }

    def _root_cause_candidates(
        self,
        current: dict[str, Any],
        baseline: dict[str, Any],
        delta: dict[str, float],
        versions: dict[str, dict[str, dict[str, int]]],
        deployment_events: list[dict[str, Any]],
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        signals = (
            ("failure_rate", "failure_rate_increase", "higher_is_worse"),
            ("refusal_rate", "refusal_rate_increase", "higher_may_be_worse"),
            ("p95_latency_ms", "p95_latency_increase", "higher_is_worse"),
            ("citation_coverage", "citation_coverage_decrease", "lower_is_worse"),
            ("faithfulness", "faithfulness_decrease", "lower_is_worse"),
        )
        for metric, signal, direction in signals:
            change = float(delta[metric])
            if metric in {"citation_coverage", "faithfulness"} and (
                current[metric] is None or baseline[metric] is None
            ):
                continue
            regressed = change > 0 if direction.startswith("higher") else change < 0
            if regressed:
                candidates.append({
                    "kind": "metric_regression", "signal": signal, "metric": metric,
                    "delta": change, "confidence": "observed", "direction": direction,
                })

        current_count = max(int(current["request_count"]), 1)
        baseline_count = max(int(baseline["request_count"]), 1)
        for error_type in sorted(set(current["error_counts"]) | set(baseline["error_counts"])):
            current_rate = current["error_counts"].get(error_type, 0) / current_count
            baseline_rate = baseline["error_counts"].get(error_type, 0) / baseline_count
            if current_rate > baseline_rate:
                candidates.append({
                    "kind": "error_increase", "signal": error_type,
                    "current_rate": current_rate, "baseline_rate": baseline_rate,
                    "delta": current_rate - baseline_rate, "confidence": "correlated",
                })

        for dimension, current_values in versions["current"].items():
            baseline_values = versions["baseline"].get(dimension, {})
            for version, count in current_values.items():
                if version != "unknown" and version not in baseline_values:
                    candidates.append({
                        "kind": "version_introduced", "signal": dimension,
                        "version": version, "request_count": count,
                        "confidence": "correlated",
                    })

        for event in deployment_events:
            event_at = self._parse_timestamp(event["created_at"])
            if self._parse_timestamp(start) <= event_at < self._parse_timestamp(end):
                candidates.append({
                    "kind": "deployment_in_window", "signal": event["event_type"],
                    "version": event["version"], "created_at": event["created_at"],
                    "deployment_event_id": event["id"], "confidence": "correlated",
                })
        priority = {"metric_regression": 0, "error_increase": 1, "deployment_in_window": 2,
                    "version_introduced": 3}
        return sorted(candidates, key=lambda item: (priority[item["kind"]], -abs(float(item.get("delta", 0)))))

    def _version_breakdown(self, start: str, end: str) -> dict[str, dict[str, int]]:
        columns = ("prompt_version", "model_version", "dataset_version", "config_version")
        result: dict[str, dict[str, int]] = {}
        with self.database.connect() as connection:
            for column in columns:
                rows = connection.execute(
                    f"""SELECT COALESCE({column}, 'unknown') AS version, COUNT(*) AS count
                        FROM traces WHERE trace_type='query' AND started_at>=? AND started_at<?
                        GROUP BY COALESCE({column}, 'unknown') ORDER BY count DESC""",
                    (start, end),
                ).fetchall()
                result[column] = {str(row["version"]): int(row["count"]) for row in rows}
        return result

    def _deployment_events(self, start: str, end: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT id,event_type,version,detail_json,created_by,created_at
                   FROM deployment_events WHERE created_at>=? AND created_at<? ORDER BY created_at""",
                (start, end),
            ).fetchall()
        return [
            {**{key: row[key] for key in row.keys() if key != "detail_json"},
             "detail": json.loads(row["detail_json"])}
            for row in rows
        ]

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def purge_content(self, now: datetime | None = None) -> int:
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=self.retention_days)).isoformat()
        with self.database.transaction(immediate=True) as connection:
            changed = connection.execute(
                "UPDATE traces SET query_text=NULL,response_text=NULL,prompt_text=NULL WHERE started_at<? AND (query_text IS NOT NULL OR response_text IS NOT NULL OR prompt_text IS NOT NULL)",
                (cutoff,),
            ).rowcount
            stage_rows = connection.execute(
                "SELECT ts.id,ts.data_json FROM trace_stages ts JOIN traces t ON t.id=ts.trace_id WHERE t.started_at<?",
                (cutoff,),
            ).fetchall()
            for row in stage_rows:
                data = json.loads(row["data_json"])
                scrubbed = self._purge_content_values(data)
                connection.execute(
                    "UPDATE trace_stages SET data_json=? WHERE id=?",
                    (json.dumps(scrubbed, ensure_ascii=False), row["id"]),
                )
        return changed

    @classmethod
    def _purge_content_values(cls, value: Any) -> Any:
        """Remove retained request/document prose at any nesting depth.

        Containers such as candidate or citation lists are retained so ranks, IDs,
        scores and degradation diagnostics remain useful after the content-retention
        window.  Keys carrying prose are removed recursively; identifiers such as
        ``chunk_id`` and hashes are intentionally unaffected.
        """
        if isinstance(value, list):
            return [cls._purge_content_values(item) for item in value]
        if not isinstance(value, dict):
            return value
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _CONTENT_KEYS or normalized.endswith(_CONTENT_SUFFIXES):
                continue
            result[key] = cls._purge_content_values(item)
        return result

    @classmethod
    def _redact(cls, value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        if isinstance(value, dict):
            return {key: ("[REDACTED]" if any(term in key.lower() for term in ("key", "password", "token", "authorization")) else cls._redact(item)) for key, item in value.items()}
        return value

    @staticmethod
    def _average(values: Iterable[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return sum(present) / len(present) if present else None
