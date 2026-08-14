"""Safe loaders for UTF-8/Unicode Markdown and plain-text documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Load Markdown or text while preserving stable content identities."""

    SUPPORTED = {".md", ".markdown", ".txt"}

    def __init__(self, max_bytes: int = 100 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported text document: {suffix}")
        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"Document exceeds {self.max_bytes} byte limit")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        if not text.strip():
            raise ValueError("Document is empty")
        digest = hashlib.sha256(raw).hexdigest()
        title = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.strip()),
            path.stem,
        )
        return Document(
            id=f"doc_{digest[:16]}",
            text=text,
            metadata={
                "source_path": str(path),
                "source_uri": path.as_uri(),
                "doc_type": "markdown" if suffix in {".md", ".markdown"} else "text",
                "doc_hash": digest,
                "title": title[:300],
                "size_bytes": size,
            },
        )
