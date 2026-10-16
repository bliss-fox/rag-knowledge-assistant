from __future__ import annotations

import json

from scripts.import_legacy_traces import main
from src.production.database import ProductionDatabase
from src.production.trace_store import SQLiteTraceStore

_STARTED_AT = "2026-08-14T08:00:00+00:00"


def _trace(trace_id: str, *, response: str = "answer") -> dict:
    return {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": _STARTED_AT,
        "finished_at": "2026-08-14T08:00:01+00:00",
        "total_elapsed_ms": 100,
        "metadata": {
            "query": "question",
            "response": response,
            "status": "answered",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        "stages": [],
    }


def test_jsonl_import_is_partial_idempotent_redacted_and_preserves_existing(tmp_path) -> None:
    database = ProductionDatabase(tmp_path / "production.db")
    store = SQLiteTraceStore(database)
    existing = _trace("production-trace", response="newer production response")
    existing["started_at"] = "2026-08-14T09:00:00+00:00"
    assert store.save(existing)

    legacy = _trace("legacy-old-layout", response="password=legacy-secret")
    legacy["metadata"]["query"] = "api_key=legacy-secret question"
    legacy["stages"] = [{
        "stage": "dense_retrieval",
        "timestamp": _STARTED_AT,
        "elapsed_ms": 12,
        "rank": 2,
        "text": "authorization=legacy-secret",
        "api_key": "legacy-secret",
    }]
    current = _trace("legacy-current-layout")
    current["stages"] = [{
        "stage": "rerank",
        "timestamp": _STARTED_AT,
        "elapsed_ms": 7,
        "rank": 99,
        "data": {"rank": 1, "prompt": "token=legacy-secret"},
    }]
    attempted_overwrite = _trace("production-trace", response="stale legacy response")
    sensitive_bad_line = '{"password":"must-not-appear"'
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text("\n".join([
        json.dumps(legacy),
        json.dumps(current),
        sensitive_bad_line,
        json.dumps({"started_at": _STARTED_AT, "query": "private"}),
        "",
        json.dumps(legacy),
        json.dumps(attempted_overwrite),
    ]) + "\n", encoding="utf-8")

    report = store.import_jsonl(input_path)

    assert {key: report[key] for key in (
        "total_lines", "imported", "skipped_existing", "blank_lines",
        "malformed_json", "invalid_schema", "storage_errors",
    )} == {
        "total_lines": 7,
        "imported": 2,
        "skipped_existing": 2,
        "blank_lines": 1,
        "malformed_json": 1,
        "invalid_schema": 1,
        "storage_errors": 0,
    }
    assert [item["line"] for item in report["errors"]] == [3, 4]
    assert "must-not-appear" not in json.dumps(report)
    assert "private" not in json.dumps(report)

    imported_old = store.get("legacy-old-layout", is_admin=True)
    assert imported_old is not None
    assert "legacy-secret" not in imported_old["query_text"]
    assert "legacy-secret" not in imported_old["response_text"]
    assert imported_old["stages"][0]["data"] == {
        "rank": 2,
        "text": "authorization=[REDACTED]",
        "api_key": "[REDACTED]",
    }
    imported_current = store.get("legacy-current-layout", is_admin=True)
    assert imported_current is not None
    assert imported_current["stages"][0]["data"] == {
        "rank": 1,
        "prompt": "token=[REDACTED]",
    }
    preserved = store.get("production-trace", is_admin=True)
    assert preserved is not None
    assert preserved["response_text"] == "newer production response"
    assert preserved["started_at"] == "2026-08-14T09:00:00+00:00"

    repeated = store.import_jsonl(input_path, max_error_details=1)
    assert repeated["imported"] == 0
    assert repeated["skipped_existing"] == 4
    assert len(repeated["errors"]) == 1
    assert repeated["error_details_truncated"] == 1


def test_jsonl_import_can_replace_only_when_explicitly_requested(tmp_path) -> None:
    store = SQLiteTraceStore(ProductionDatabase(tmp_path / "production.db"))
    assert store.save(_trace("same-id", response="production"))
    legacy = _trace("same-id", response="explicit replacement")
    path = tmp_path / "replace.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    report = store.import_jsonl(path, skip_existing=False)

    assert report["imported"] == 1
    assert report["skipped_existing"] == 0
    assert store.get("same-id", is_admin=True)["response_text"] == "explicit replacement"


def test_legacy_trace_import_cli_reports_results_and_fails_closed_on_bad_lines(
    tmp_path, capsys,
) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(json.dumps(_trace("cli-trace")) + "\n", encoding="utf-8")
    database = tmp_path / "cli.db"

    assert main(["--input", str(valid), "--database", str(database)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["imported"] == 1
    assert main(["--input", str(valid), "--database", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_existing"] == 1

    partial = tmp_path / "partial.jsonl"
    partial.write_text("not-json\n" + json.dumps(_trace("after-error")) + "\n", encoding="utf-8")
    assert main(["--input", str(partial), "--database", str(database)]) == 2
    partial_report = json.loads(capsys.readouterr().out)
    assert partial_report["malformed_json"] == 1
    assert partial_report["imported"] == 1
    assert main([
        "--input", str(partial), "--database", str(database), "--allow-partial",
    ]) == 0
