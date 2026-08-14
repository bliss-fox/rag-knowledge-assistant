#!/usr/bin/env python
"""Fail CI when absolute or main-branch regression thresholds are violated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {"faithfulness": 0.85, "citation_coverage": 0.95, "recall_at_5": 0.80}
MAX_REGRESSION = 0.02


def check(current: dict[str, float], baseline: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, threshold in THRESHOLDS.items():
        if metric not in current:
            failures.append(f"missing current metric: {metric}")
            continue
        if current[metric] < threshold:
            failures.append(f"{metric}={current[metric]:.4f} < {threshold:.4f}")
        if metric not in baseline:
            failures.append(f"missing baseline metric: {metric}")
        elif baseline[metric] - current[metric] > MAX_REGRESSION:
            failures.append(
                f"{metric} regressed by {baseline[metric] - current[metric]:.4f} > {MAX_REGRESSION:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    current_payload = json.loads(Path(args.current).read_text(encoding="utf-8"))
    baseline_payload = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    if baseline_payload.get("status") == "pending_reproducible_freeze":
        print("Quality gate failed: baseline has not been reproducibly frozen")
        return 1
    current = current_payload["metrics"]
    baseline = baseline_payload["metrics"]
    failures = check(current, baseline)
    if failures:
        print("Quality gate failed:\n- " + "\n- ".join(failures))
        return 1
    print("Quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
