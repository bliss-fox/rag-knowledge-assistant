"""ConversationMemory — multi-turn context tracking for the ReAct agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class _MemoryTurn:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    """Stores multi-turn dialogue history and provides context for query rewriting.

    Usage in a multi-round session:
    - Call add_turn() after each user/assistant exchange.
    - Call get_context_messages() to prepend history to LLM prompts.
    - Call rewrite_query() to resolve pronouns/references using history.
    - Call summarize_if_long() to compress old turns when context grows.

    Args:
        max_turns: Maximum number of turns kept verbatim before summarising.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._turns: List[_MemoryTurn] = []
        self._summary: Optional[str] = None
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: "user" or "assistant".
            content: The message text.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")
        self._turns.append(_MemoryTurn(role=role, content=content))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return conversation history in OpenAI-style messages format.

        Returns:
            List of {"role": ..., "content": ...} dicts, optionally
            prefixed with a summary message if summarisation has run.
        """
        messages: List[Dict[str, str]] = []
        if self._summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary so far]\n{self._summary}",
            })
        for t in self._turns:
            messages.append({"role": t.role, "content": t.content})
        return messages

    def get_recent_queries(self, n: int = 3) -> List[str]:
        """Return the last n user queries from history.

        Args:
            n: How many recent user messages to return.

        Returns:
            List of user query strings (most recent last).
        """
        user_msgs = [t.content for t in self._turns if t.role == "user"]
        return user_msgs[-n:]

    def format_as_text(self) -> str:
        """Render history as a plain text block for inclusion in prompts."""
        if not self._turns and not self._summary:
            return ""
        lines: List[str] = []
        if self._summary:
            lines.append(f"[Previous conversation summary]\n{self._summary}")
        for t in self._turns:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, llm: BaseLLM) -> str:
        """Rewrite a query to resolve pronouns/references using conversation history.

        Falls back to the original query if no history exists or LLM call fails.

        Args:
            query: The latest user query (may contain "it", "this", etc.).
            llm: LLM instance used for rewriting.

        Returns:
            A standalone, self-contained version of the query.
        """
        recent = self.get_recent_queries(n=3)
        if not recent:
            return query  # nothing to resolve

        context_str = "\n".join(f"- {q}" for q in recent)
        prompt = (
            "Given the following recent questions from a conversation:\n"
            f"{context_str}\n\n"
            "Rewrite this follow-up question as a fully self-contained query that "
            "does not rely on pronouns or implicit references:\n"
            f'"{query}"\n\n'
            "Output ONLY the rewritten question, nothing else."
        )
        from src.libs.llm.base_llm import Message
        try:
            response = llm.chat([Message(role="user", content=prompt)])
            rewritten = response.content.strip().strip('"')
            logger.debug("Query rewritten: '%s' → '%s'", query, rewritten)
            return rewritten
        except Exception as exc:
            logger.warning("Query rewriting failed: %s. Using original.", exc)
            return query

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize_if_long(self, llm: BaseLLM, max_turns: Optional[int] = None) -> None:
        """Compress old turns into a summary when history exceeds max_turns.

        After summarisation, older turns are replaced by the summary and only
        the most recent turns are kept verbatim.

        Args:
            llm: LLM instance used to generate the summary.
            max_turns: Override for the max_turns threshold.
        """
        threshold = max_turns or self.max_turns
        if len(self._turns) <= threshold:
            return

        turns_to_summarise = self._turns[:-threshold]
        keep = self._turns[-threshold:]

        dialogue = "\n".join(
            f"{t.role.capitalize()}: {t.content}" for t in turns_to_summarise
        )
        prompt = (
            "Summarise the following conversation in 3-5 sentences, "
            "preserving the key facts and decisions:\n\n"
            f"{dialogue}"
        )
        from src.libs.llm.base_llm import Message
        try:
            response = llm.chat([Message(role="user", content=prompt)])
            self._summary = response.content.strip()
            self._turns = list(keep)
            logger.debug("Conversation summarised; %d turns retained.", len(keep))
        except Exception as exc:
            logger.warning("Summarisation failed: %s. History unchanged.", exc)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._turns.clear()
        self._summary = None

    def __len__(self) -> int:
        return len(self._turns)
