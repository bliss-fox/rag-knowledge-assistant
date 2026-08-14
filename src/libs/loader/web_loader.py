"""Restricted same-origin web loader with SSRF protection."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import httpx

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class UnsafeURLError(ValueError):
    """Raised when a URL can reach a non-public network."""


@dataclass(frozen=True)
class WebCrawlResult:
    """Documents plus enough crawl state to make deletion decisions safely."""

    documents: tuple[Document, ...]
    fetched_pages: int
    complete: bool
    truncation_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "discovered": len(self.documents),
            "fetched_pages": self.fetched_pages,
            "complete": self.complete,
            "truncation_reason": self.truncation_reason,
        }


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.links: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.text.append(data.strip())


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURLError("Only absolute HTTP(S) URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs containing credentials are not allowed")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        raise UnsafeURLError("Local hosts are not allowed")
    default_port = 80 if parsed.scheme == "http" else 443
    try:
        addresses = {
            item[4][0] for item in socket.getaddrinfo(host, parsed.port or default_port)
        }
    except socket.gaierror as exc:
        raise UnsafeURLError("URL host cannot be resolved") from exc
    for value in addresses:
        ip = ipaddress.ip_address(value)
        if not ip.is_global:
            raise UnsafeURLError("Private, loopback, link-local and reserved IPs are blocked")
    port = parsed.port
    rendered_host = f"[{host}]" if ":" in host else host
    netloc = rendered_host if port in {None, default_port} else f"{rendered_host}:{port}"
    canonical = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=parsed.path or "/",
        fragment="",
    )
    return urlunparse(canonical)


class WebLoader(BaseLoader):
    """Fetch one page or a bounded same-origin crawl."""

    def __init__(
        self,
        max_depth: int = 1,
        max_pages: int = 20,
        max_response_bytes: int = 5 * 1024 * 1024,
        timeout: float = 15.0,
        allowed_path_prefixes: Iterable[str] = (),
        client: httpx.Client | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_response_bytes = max_response_bytes
        self.timeout = timeout
        self.allowed_path_prefixes = tuple(allowed_path_prefixes)
        self._client = client

    def load(self, file_path: str) -> Document:
        return self.load_url(file_path)

    def load_url(self, url: str) -> Document:
        docs = self.crawl(url, max_pages=1)
        if not docs:
            raise RuntimeError("Web page returned no readable content")
        return docs[0]

    def crawl(self, url: str, max_pages: int | None = None) -> list[Document]:
        return list(self.crawl_with_report(url, max_pages=max_pages).documents)

    def crawl_with_report(
        self, url: str, max_pages: int | None = None,
    ) -> WebCrawlResult:
        """Crawl while reporting whether the configured frontier was exhausted.

        A caller must not infer that previously indexed pages were deleted when
        ``complete`` is false. This happens most commonly when ``max_pages`` is
        reached while unseen same-origin URLs remain in the frontier.
        """
        root = _validate_public_url(url)
        origin = urlparse(root)
        requested_limit = self.max_pages if max_pages is None else max_pages
        limit = min(requested_limit, self.max_pages)
        if limit <= 0:
            raise ValueError("max_pages must be greater than zero")
        pending = deque([(root, 0)])
        visited: set[str] = set()
        documents: list[Document] = []
        fetched_pages = 0
        client = self._client or httpx.Client(timeout=self.timeout, follow_redirects=False)
        owns_client = self._client is None
        try:
            while pending and fetched_pages < limit:
                current, depth = pending.popleft()
                current = _validate_public_url(current)
                if current in visited or not self._path_allowed(current):
                    continue
                visited.add(current)
                fetched_pages += 1
                response = client.get(current, headers={"User-Agent": "ModularRAG/1.0"})
                if response.is_redirect:
                    target = urljoin(current, response.headers.get("location", ""))
                    self._assert_same_origin(origin, target)
                    pending.appendleft((_validate_public_url(target), depth))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    continue
                raw = response.content
                if len(raw) > self.max_response_bytes:
                    raise ValueError("Web response exceeds configured size limit")
                parser = _HTMLTextParser()
                parser.feed(response.text)
                text = re.sub(r"\s+", " ", "\n".join(parser.text)).strip()
                if text:
                    digest = hashlib.sha256((current + "\n" + text).encode()).hexdigest()
                    documents.append(Document(
                        id=f"web_{digest[:16]}",
                        text=text,
                        metadata={
                            "source_path": current,
                            "source_uri": current,
                            "doc_type": "web",
                            "doc_hash": digest,
                            "title": text[:120],
                        },
                    ))
                if depth < self.max_depth:
                    for href in parser.links:
                        target = urldefrag(urljoin(current, href)).url
                        try:
                            self._assert_same_origin(origin, target)
                            target = _validate_public_url(target)
                        except (UnsafeURLError, ValueError):
                            continue
                        if self._path_allowed(target):
                            pending.append((target, depth + 1))
        finally:
            if owns_client:
                client.close()
        unseen_frontier = any(candidate not in visited for candidate, _ in pending)
        complete = not unseen_frontier
        return WebCrawlResult(
            documents=tuple(documents),
            fetched_pages=fetched_pages,
            complete=complete,
            truncation_reason="page_limit_reached" if not complete else None,
        )

    def _assert_same_origin(self, origin, target: str) -> None:
        parsed = urlparse(_validate_public_url(target))
        origin_port = origin.port or (80 if origin.scheme == "http" else 443)
        target_port = parsed.port or (80 if parsed.scheme == "http" else 443)
        if (
            parsed.scheme.lower() != origin.scheme.lower()
            or parsed.hostname != origin.hostname
            or target_port != origin_port
        ):
            raise UnsafeURLError("Cross-origin redirects and links are blocked")

    def _path_allowed(self, url: str) -> bool:
        if not self.allowed_path_prefixes:
            return True
        path = urlparse(url).path or "/"
        return any(path.startswith(prefix) for prefix in self.allowed_path_prefixes)
