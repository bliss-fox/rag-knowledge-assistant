from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest

from src.libs.loader.registry import default_loader_registry
from src.libs.loader.text_loader import TextLoader
from src.libs.loader.web_loader import UnsafeURLError, WebLoader, _validate_public_url


def test_text_loader_markdown_stable_identity(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# 指南\n\n正文", encoding="utf-8")
    first = TextLoader().load(path)
    second = default_loader_registry.create(path).load(path)
    assert first.id == second.id
    assert first.metadata["doc_type"] == "markdown"
    assert first.metadata["title"] == "指南"


def test_text_loader_size_and_empty_limits(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        TextLoader().load(empty)
    full = tmp_path / "full.txt"
    full.write_text("abcd", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        TextLoader(max_bytes=2).load(full)


def test_ssrf_blocks_local_and_non_http(monkeypatch):
    with pytest.raises(UnsafeURLError):
        _validate_public_url("file:///etc/passwd")
    with pytest.raises(UnsafeURLError):
        _validate_public_url("http://localhost/x")
    monkeypatch.setattr("socket.getaddrinfo", lambda *args: [(None, None, None, None, ("127.0.0.1", 80))])
    with pytest.raises(UnsafeURLError):
        _validate_public_url("http://example.test/x")


def test_public_url_canonicalizes_origin_and_uses_scheme_port(monkeypatch):
    calls = []

    def fake_getaddrinfo(host, port):
        calls.append((host, port))
        return [(None, None, None, None, ("93.184.216.34", port))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)

    assert _validate_public_url("http://EXAMPLE.test:80#part") == "http://example.test/"
    assert calls == [("example.test", 80)]
    with pytest.raises(UnsafeURLError, match="credentials"):
        _validate_public_url("https://user:pass@example.test/")


def test_web_loader_single_page_with_injected_client(monkeypatch):
    monkeypatch.setattr("src.libs.loader.web_loader._validate_public_url", lambda value: value)
    response = httpx.Response(
        200, headers={"content-type": "text/html"},
        text="<html><title>Doc</title><script>secret()</script><body>Hello <a href='/next'>World</a></body></html>",
        request=httpx.Request("GET", "https://example.com"),
    )
    client = Mock()
    client.get.return_value = response
    document = WebLoader(client=client).load_url("https://example.com")
    assert "Hello" in document.text and "secret" not in document.text
    assert document.metadata["source_uri"] == "https://example.com"


def test_web_loader_reports_page_limit_without_claiming_complete(monkeypatch):
    monkeypatch.setattr("src.libs.loader.web_loader._validate_public_url", lambda value: value)
    response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text="<body>Root <a href='/one'>One</a><a href='/two'>Two</a></body>",
        request=httpx.Request("GET", "https://example.com"),
    )
    client = Mock()
    client.get.return_value = response

    report = WebLoader(client=client, max_depth=1, max_pages=1).crawl_with_report(
        "https://example.com",
    )

    assert len(report.documents) == 1
    assert report.fetched_pages == 1
    assert report.complete is False
    assert report.truncation_reason == "page_limit_reached"


def test_web_loader_reports_complete_when_frontier_is_exhausted(monkeypatch):
    monkeypatch.setattr("src.libs.loader.web_loader._validate_public_url", lambda value: value)
    response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        text="<body>Only page</body>",
        request=httpx.Request("GET", "https://example.com"),
    )
    client = Mock()
    client.get.return_value = response

    report = WebLoader(client=client, max_depth=1, max_pages=5).crawl_with_report(
        "https://example.com",
    )

    assert report.complete is True
    assert report.truncation_reason is None
