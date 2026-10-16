#!/usr/bin/env python
"""Fail fast before dispatching the model-backed public quality job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.quality_gate import THRESHOLDS
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible


def check_readiness(dataset_path: Path, baseline_path: Path) -> list[str]:
    """Return configuration failures without contacting models or external services."""
    failures: list[str] = []
    try:
        require_quality_gate_eligible(dataset_path)
        dataset = load_golden_dataset(dataset_path, split="final")
        if dataset.visibility != "public":
            failures.append("quality-gate dataset must be public")
        if not any(case.answerable for case in dataset.cases):
            failures.append("public final split has no answerable cases")
        if not any(not case.answerable for case in dataset.cases):
            failures.append("public final split has no unanswerable cases")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"dataset is not ready: {exc}")

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("status") == "pending_reproducible_freeze":
            failures.append("quality baseline has not been reproducibly frozen")
        metrics = baseline.get("metrics")
        if not isinstance(metrics, dict):
            failures.append("quality baseline metrics must be an object")
        else:
            for name in THRESHOLDS:
                value = metrics.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    failures.append(f"quality baseline is missing numeric metric: {name}")
                elif not 0.0 <= float(value) <= 1.0:
                    failures.append(f"quality baseline metric is outside [0, 1]: {name}")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"quality baseline is not readable: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/public_golden.json"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("evaluation/baseline.pending.json"),
    )
    args = parser.parse_args()
    failures = check_readiness(args.dataset, args.baseline)
    if failures:
        print("Public quality preflight failed:\n- " + "\n- ".join(failures))
        return 1
    print("Public quality preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
