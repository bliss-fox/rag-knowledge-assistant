from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.production.evidence import EvidenceGate
from src.production.prompts import PromptRegistry


def _result(score=0.8):
    return RetrievalResult("chunk-1", score, "支持该事实的文本", {"source_path": "doc.md"})


def test_evidence_gate_precheck_and_postcheck():
    gate = EvidenceGate(min_top_score=.4, min_evidence_count=1, min_citation_coverage=1.0)
    assert gate.precheck([]).reason == "no_evidence"
    assert gate.precheck([_result(.2)]).reason == "low_relevance"
    assert gate.precheck([_result()]).accepted
    assert gate.postcheck("事实成立。[1]", [_result()]).accepted
    assert gate.postcheck("事实成立。", [_result()]).reason == "insufficient_citation_coverage"
    assert gate.postcheck("事实成立。[2]", [_result()]).reason == "invalid_citation"
    assert gate.postcheck("火星氧气产量是 900 吨。[1]", [_result()]).reason == "unsupported_citation"


def test_evidence_gate_requires_numeric_and_lexical_support():
    gate = EvidenceGate(min_top_score=.4, min_evidence_count=1, min_citation_coverage=1.0)
    evidence = RetrievalResult(
        "chunk-1", .9, "RRF 的默认平滑参数 k 为 60。", {"source_path": "doc.md"}
    )
    assert gate.postcheck("RRF 默认 k 是 60。[1]", [evidence]).accepted
    assert gate.postcheck("RRF 默认 k 是 90。[1]", [evidence]).reason == "unsupported_citation"


def test_versioned_prompt_registry_and_hash(tmp_path):
    (tmp_path / "prompt.yaml").write_text(
        "id: p\nversion: 1.0.0\ndescription: test\nrequired_variables: [name]\ntemplate: 'Hi {name}'\n",
        encoding="utf-8",
    )
    prompt = PromptRegistry(tmp_path).get("p")
    assert prompt.render(name="Codex") == "Hi Codex"
    assert len(prompt.sha256) == 64
    with pytest.raises(ValueError, match="Missing"):
        prompt.render()


def test_prompt_registry_rejects_variable_mismatch(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "id: p\nversion: 1\ndescription: test\nrequired_variables: []\ntemplate: 'Hi {name}'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="variables"):
        PromptRegistry(tmp_path)
