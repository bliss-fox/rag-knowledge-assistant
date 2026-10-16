#!/usr/bin/env python
"""Import legacy ``logs/traces.jsonl`` records into the production SQLite store."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings, resolve_path  # noqa: E402
from src.production.database import ProductionDatabase  # noqa: E402
from src.production.trace_store import SQLiteTraceStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import legacy JSONL traces into SQLite. Existing trace IDs are skipped "
            "atomically unless --replace-existing is explicitly supplied."
        ),
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--input",
        help="JSONL path; defaults to observability.trace_file from the configuration",
    )
    parser.add_argument(
        "--database",
        help="SQLite path; defaults to observability.sqlite_path from the configuration",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace matching SQLite trace IDs (unsafe for normal migrations)",
    )
    parser.add_argument("--max-error-details", type=int, default=100)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success even if malformed, invalid, or unstorable lines were reported",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        input_path = resolve_path(args.input or settings.observability.trace_file)
        database_path = resolve_path(args.database or settings.observability.sqlite_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"legacy trace file not found: {input_path}")
        store = SQLiteTraceStore(
            ProductionDatabase(database_path),
            retention_days=settings.observability.content_retention_days,
        )
        report = store.import_jsonl(
            input_path,
            skip_existing=not args.replace_existing,
            max_error_details=args.max_error_details,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Trace import failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = sum(
        int(report[key]) for key in ("malformed_json", "invalid_schema", "storage_errors")
    )
    return 0 if args.allow_partial or failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
