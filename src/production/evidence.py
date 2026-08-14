"""Deterministic evidence sufficiency and citation coverage gate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.core.types import RetrievalResult


_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
_CJK = re.compile(r"[\u3400-\u9fff]+")
_STOP_WORDS = {
    "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
    "has", "have", "uses", "using", "into", "its", "not", "can",
}


@dataclass(frozen=True)
class EvidenceDecision:
    accepted: bool
    score: float
    reason: str | None
    evidence_count: int
    citation_coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted, "score": round(self.score, 4),
            "reason": self.reason, "evidence_count": self.evidence_count,
            "citation_coverage": self.citation_coverage,
        }


class EvidenceGate:
    def __init__(self, min_top_score: float = 0.35, min_evidence_count: int = 1, min_citation_coverage: float = 0.95) -> None:
        self.min_top_score = min_top_score
        self.min_evidence_count = min_evidence_count
        self.min_citation_coverage = min_citation_coverage

    def precheck(self, results: Sequence[RetrievalResult]) -> EvidenceDecision:
        if not results:
            return EvidenceDecision(False, 0.0, "no_evidence", 0)
        top_score = max(float(item.score) for item in results)
        if len(results) < self.min_evidence_count:
            return EvidenceDecision(False, top_score, "insufficient_evidence_count", len(results))
        if top_score < self.min_top_score:
            return EvidenceDecision(False, top_score, "low_relevance", len(results))
        return EvidenceDecision(True, top_score, None, len(results))

    def postcheck(self, answer: str, results: Sequence[RetrievalResult]) -> EvidenceDecision:
        sentences = [item.strip() for item in _SENTENCE.split(answer) if item.strip()]
        factual = [item for item in sentences if self._is_factual(item)]
        cited = 0
        invalid = False
        unsupported = False
        for sentence in factual:
            markers = [int(value) for value in _CITATION.findall(sentence)]
            if markers:
                cited += 1
            if any(value < 1 or value > len(results) for value in markers):
                invalid = True
            elif markers and not any(
                self._supports(sentence, results[value - 1].text) for value in markers
            ):
                unsupported = True
        coverage = cited / len(factual) if factual else 1.0
        accepted = (
            bool(answer.strip()) and not invalid and not unsupported
            and coverage >= self.min_citation_coverage
        )
        if accepted:
            reason = None
        elif invalid:
            reason = "invalid_citation"
        elif unsupported:
            reason = "unsupported_citation"
        else:
            reason = "insufficient_citation_coverage"
        score = max((float(item.score) for item in results), default=0.0)
        return EvidenceDecision(accepted, score, reason, len(results), round(coverage, 4))

    @staticmethod
    def _is_factual(sentence: str) -> bool:
        stripped = _CITATION.sub("", sentence).strip()
        if not stripped or stripped.startswith(("现有知识库证据不足", "无法可靠回答")):
            return False
        return len(stripped) >= 2

    @classmethod
    def _supports(cls, claim: str, evidence: str) -> bool:
        """Conservative local check that a cited passage can support a claim."""
        clean_claim = _CITATION.sub("", claim).strip().lower()
        clean_evidence = evidence.lower()
        numbers = set(_NUMBER.findall(clean_claim))
        if numbers and not numbers <= set(_NUMBER.findall(clean_evidence)):
            return False
        claim_tokens = cls._tokens(clean_claim)
        if not claim_tokens:
            return False
        evidence_tokens = cls._tokens(clean_evidence)
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        return bool(claim_tokens & evidence_tokens) and overlap >= 0.15

    @staticmethod
    def _tokens(value: str) -> set[str]:
        words = {item.lower() for item in _WORD.findall(value)} - _STOP_WORDS
        cjk: set[str] = set()
        for sequence in _CJK.findall(value):
            if len(sequence) == 1:
                cjk.add(sequence)
            else:
                cjk.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
        return words | cjk
