#!/usr/bin/env python
"""Evaluate only the repository public final split with a fixed DeepSeek judge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from src.core.settings import resolve_path
from src.production.evaluation import (
    load_golden_dataset, refusal_metrics, require_quality_gate_eligible,
)
from src.production.prompts import PromptRegistry


JUDGE_MODEL = "deepseek-chat"
JUDGE_TEMPERATURE = 0.0
JUDGE_PROMPT_ID = "grounded_answer_quality_judge"
SCORE_FIELDS = ("faithfulness", "answer_relevancy", "citation_coverage")


def _judge(api_key: str, rendered_prompt: str) -> dict[str, float]:
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": rendered_prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    value = json.loads(response.json()["choices"][0]["message"]["content"])
    scores = {name: float(value[name]) for name in SCORE_FIELDS}
    invalid = {name: score for name, score in scores.items() if not 0.0 <= score <= 1.0}
    if invalid:
        raise ValueError(f"Judge returned scores outside [0, 1]: {invalid}")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="JSON rows produced from public corpus only")
    parser.add_argument("--dataset", default="evaluation/public_golden.json")
    parser.add_argument("--output", default="artifacts/deepseek-quality.json")
    args = parser.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required; quality evaluation cannot be skipped")
    require_quality_gate_eligible(args.dataset)
    dataset = load_golden_dataset(args.dataset, split="final")
    if dataset.visibility != "public":
        raise SystemExit("Refusing to send a non-public dataset to DeepSeek")
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    by_id = {row["query_id"]: row for row in answers["answers"]}
    if len(by_id) != len(answers["answers"]):
        raise SystemExit("Duplicate query_id in answer rows")
    prompt = PromptRegistry(resolve_path("config/prompts")).get(JUDGE_PROMPT_ID)
    scores: list[dict[str, float]] = []
    refusal_rows: list[tuple[bool, bool]] = []
    case_results: list[dict[str, Any]] = []
    for case in dataset.cases:
        if case.query_id not in by_id:
            raise SystemExit(f"Missing answer row: {case.query_id}")
        row = by_id[case.query_id]
        refused = row.get("status") == "refused"
        refusal_rows.append((case.answerable, refused))
        item: dict[str, Any] = {
            "query_id": case.query_id,
            "answerable": case.answerable,
            "status": row.get("status"),
        }
        if case.answerable:
            if refused or row.get("status") == "error":
                score = {name: 0.0 for name in SCORE_FIELDS}
            else:
                rendered = prompt.render(
                    query=case.query,
                    key_points=json.dumps(case.answer_key_points, ensure_ascii=False),
                    answer=str(row.get("answer", "")),
                    citations=json.dumps(row.get("citations", []), ensure_ascii=False),
                    visibility="public",
                )
                score = _judge(api_key, rendered)
            scores.append(score)
            item["scores"] = score
        case_results.append(item)
    if not scores:
        raise SystemExit("Final split has no answerable cases to judge")
    metrics = {
        name: sum(row[name] for row in scores) / len(scores)
        for name in SCORE_FIELDS
    }
    metrics.update(refusal_metrics(refusal_rows))
    metrics["recall_at_5"] = float(answers["metrics"]["recall_at_5"])
    output = {
        "metrics": metrics,
        "judge": {"model": JUDGE_MODEL, "temperature": JUDGE_TEMPERATURE,
                  "prompt_id": prompt.prompt_id, "prompt_version": prompt.version,
                  "prompt_sha256": prompt.sha256},
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "parameters": answers.get("parameters", {}),
        "cases": case_results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
