#!/usr/bin/env python
"""Verify local production dependencies and emit a sanitized JSON report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.production.local_preflight import LocalDependencyPreflight


def main(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[..., Any] = LocalDependencyPreflight,
) -> int:
    parser = argparse.ArgumentParser(
        description="Sanitized preflight for Ollama, Qdrant and Cross-Encoder",
    )
    parser.add_argument("--settings", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--model-timeout", type=float, default=180.0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    runner = runner_factory(
        settings,
        http_timeout=args.http_timeout,
        model_timeout=args.model_timeout,
    )
    report = runner.run()
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
