"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.registry import LoaderRegistry, default_loader_registry
from src.libs.loader.text_loader import TextLoader
from src.libs.loader.web_loader import UnsafeURLError, WebLoader
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "TextLoader",
    "WebLoader",
    "UnsafeURLError",
    "LoaderRegistry",
    "default_loader_registry",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
