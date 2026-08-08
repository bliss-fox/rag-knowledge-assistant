"""Shared lexical tokenization for BM25 indexing and querying."""

from __future__ import annotations

import re

import jieba

_LEXICAL_SEGMENT = re.compile(
    r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+"
)
_CHINESE_SEGMENT = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_search_text(text: str) -> list[str]:
    """Tokenize Chinese text while preserving ASCII technical identifiers.

    Jieba separates punctuation such as the hyphen in ``GPT-4``.  Keeping
    dotted, hyphenated, and underscored identifiers intact is important for
    exact BM25 matches, so ASCII compounds are extracted first and only
    contiguous Chinese spans are passed through jieba.
    """
    tokens: list[str] = []
    for match in _LEXICAL_SEGMENT.finditer(text):
        segment = match.group(0)
        if _CHINESE_SEGMENT.fullmatch(segment):
            tokens.extend(token.strip() for token in jieba.lcut(segment) if token.strip())
        else:
            tokens.append(segment)
    return tokens
