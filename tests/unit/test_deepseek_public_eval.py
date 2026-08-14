from __future__ import annotations

import json

import pytest

from scripts import deepseek_public_eval as quality


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        content = json.dumps({
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "citation_coverage": 1.0,
        })
        return {"choices": [{"message": {"content": content}}]}


def test_judge_uses_rendered_versioned_prompt(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr(quality.httpx, "post", post)
    assert quality._judge("secret", "rendered prompt") == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "citation_coverage": 1.0,
    }
    assert captured["json"]["messages"][0]["content"] == "rendered prompt"


def test_judge_rejects_out_of_range_score(monkeypatch):
    class BadResponse(_Response):
        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "faithfulness": 1.1,
                "answer_relevancy": 0.8,
                "citation_coverage": 1.0,
            })}}]}

    monkeypatch.setattr(quality.httpx, "post", lambda *args, **kwargs: BadResponse())
    with pytest.raises(ValueError, match="outside"):
        quality._judge("secret", "rendered prompt")
