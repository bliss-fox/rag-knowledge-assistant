"""SelfChecker — LLM-based grounding check and hallucination detection."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import RetrievalResult
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_CONFIDENCE_RE = re.compile(r'"?confidence"?\s*:\s*([0-9.]+)', re.IGNORECASE)
_GROUNDED_RE = re.compile(r'"?is_grounded"?\s*:\s*(true|false)', re.IGNORECASE)
_MISSING_RE = re.compile(r'"?missing_aspects"?\s*:\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL)
_RETRY_RE = re.compile(r'"?should_retry"?\s*:\s*(true|false)', re.IGNORECASE)


@dataclass
class CheckResult:
    """Output of SelfChecker.check()."""
    confidence: float          # 0.0 – 1.0
    is_grounded: bool          # Every claim has context support
    missing_aspects: List[str] # Information gaps identified
    should_retry: bool         # Whether to trigger another retrieval round
    raw_response: str = ""     # Raw LLM output for debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "is_grounded": self.is_grounded,
            "missing_aspects": self.missing_aspects,
            "should_retry": self.should_retry,
        }


_SELF_CHECK_PROMPT = """\
You are a fact-checking assistant. Given a question, a candidate answer, \
and the retrieved context passages, evaluate the answer quality.

Question: {question}

Retrieved context:
{context}

Candidate answer:
{answer}

Evaluate the answer on these dimensions and respond ONLY with valid JSON:
{{
  "confidence": <float 0.0-1.0 — how confident you are the answer is correct and complete>,
  "is_grounded": <true/false — every claim in the answer is supported by the context>,
  "missing_aspects": [<string>, ...],  // aspects of the question not addressed
  "should_retry": <true/false — would another retrieval round meaningfully improve the answer>
}}
"""


class SelfChecker:
    """LLM-based grounding checker for ReAct agent answers.

    Evaluates whether the agent's answer is:
    1. Grounded in the retrieved context (no hallucination)
    2. Sufficiently complete (no major information gaps)
    3. Worth retrying with additional retrieval

    Args:
        llm: LLM instance for evaluation.
        confidence_threshold: Below this value, should_retry is forced True.
    """

    def __init__(
        self,
        llm: BaseLLM,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def check(
        self,
        question: str,
        answer: str,
        context: List[RetrievalResult],
    ) -> CheckResult:
        """Check whether the answer is grounded and complete.

        Args:
            question: The original user question.
            answer: The agent's candidate answer.
            context: Retrieved chunks used to produce the answer.

        Returns:
            CheckResult with confidence score and grounding assessment.
        """
        if not answer or not answer.strip():
            return CheckResult(
                confidence=0.0,
                is_grounded=False,
                missing_aspects=["No answer was produced."],
                should_retry=True,
                raw_response="",
            )

        context_text = self._format_context(context)
        prompt = _SELF_CHECK_PROMPT.format(
            question=question,
            context=context_text or "(no context retrieved)",
            answer=answer,
        )

        from src.libs.llm.base_llm import Message
        try:
            response = self.llm.chat([Message(role="user", content=prompt)])
            raw = response.content.strip()
        except Exception as exc:
            logger.warning("SelfChecker LLM call failed: %s", exc)
            return self._fallback_result()

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, context: List[RetrievalResult]) -> str:
        if not context:
            return ""
        parts: List[str] = []
        for i, r in enumerate(context[:5], 1):  # limit to top-5 for prompt length
            src = r.metadata.get("source_path", "unknown")
            snippet = (r.text or "")[:400].replace("\n", " ")
            parts.append(f"[{i}] {src}: {snippet}")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> CheckResult:
        """Parse JSON from LLM output with regex fallbacks."""
        # Try clean JSON parse first
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
            data = json.loads(cleaned)
            confidence = float(data.get("confidence", 0.5))
            is_grounded = bool(data.get("is_grounded", False))
            missing = [str(x) for x in data.get("missing_aspects", [])]
            should_retry = bool(data.get("should_retry", confidence < self.confidence_threshold))
            return CheckResult(
                confidence=round(min(max(confidence, 0.0), 1.0), 3),
                is_grounded=is_grounded,
                missing_aspects=missing,
                should_retry=should_retry,
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback
        conf_match = _CONFIDENCE_RE.search(raw)
        ground_match = _GROUNDED_RE.search(raw)
        missing_match = _MISSING_RE.search(raw)
        retry_match = _RETRY_RE.search(raw)

        confidence = float(conf_match.group(1)) if conf_match else 0.5
        is_grounded = ground_match.group(1).lower() == "true" if ground_match else False
        missing: List[str] = []
        if missing_match:
            for item in missing_match.group(1).split(","):
                cleaned_item = item.strip().strip('"\'')
                if cleaned_item:
                    missing.append(cleaned_item)
        should_retry = (
            retry_match.group(1).lower() == "true"
            if retry_match
            else confidence < self.confidence_threshold
        )

        return CheckResult(
            confidence=round(min(max(confidence, 0.0), 1.0), 3),
            is_grounded=is_grounded,
            missing_aspects=missing,
            should_retry=should_retry,
            raw_response=raw,
        )

    def _fallback_result(self) -> CheckResult:
        return CheckResult(
            confidence=0.5,
            is_grounded=False,
            missing_aspects=[],
            should_retry=False,
            raw_response="",
        )
