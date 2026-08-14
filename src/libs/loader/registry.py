"""Central loader registry used by CLI, API and workers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader


class LoaderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., BaseLoader]] = {}

    def register(self, extensions: tuple[str, ...], factory: Callable[..., BaseLoader]) -> None:
        for extension in extensions:
            self._factories[extension.lower()] = factory

    def create(self, source: str | Path, **kwargs) -> BaseLoader:
        extension = Path(source).suffix.lower()
        factory = self._factories.get(extension)
        if factory is None:
            raise ValueError(f"Unsupported document type: {extension or '(none)'}")
        return factory(**kwargs)

    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


default_loader_registry = LoaderRegistry()
default_loader_registry.register((".pdf",), PdfLoader)
default_loader_registry.register((".md", ".markdown", ".txt"), TextLoader)
