#!/usr/bin/env python
"""Authenticated production RAG CLI.

The historical CLI constructed retrievers and an LLM in-process.  That made it
possible to generate an answer without authentication, EvidenceGate, or the
SQLite trace store.  This entry point is intentionally a thin FastAPI client:
all answers now pass through the same ``RAGApplicationService.answer`` path as
the production Dashboard.

Usage:
    python scripts/agent.py --query "What is BM25?"
    python scripts/agent.py --collection default --top-k 5 --verbose

Credentials are requested interactively.  Automation may set
``RAG_CLI_USERNAME`` and ``RAG_CLI_PASSWORD``; passwords are deliberately not
accepted as command-line arguments because process arguments are observable.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.observability.dashboard.api_client import (  # noqa: E402
    DashboardAPIClient,
    DashboardAPIError,
)


def _configure_windows_console() -> None:
    """Use UTF-8 only for a real script process, never while imported by pytest."""
    if sys.platform != "win32":
        return
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class AuthenticatedRAGClient:
    """Short-lived authenticated client with refresh and explicit logout."""

    def __init__(
        self,
        base_url: str | None = None,
        client_factory: Callable[..., DashboardAPIClient] = DashboardAPIClient,
    ) -> None:
        self.base_url = base_url
        self._client_factory = client_factory
        self._anonymous = client_factory(base_url=base_url)
        self._authenticated: DashboardAPIClient | None = None
        self._refresh_token: str | None = None

    def login(self, username: str, password: str) -> dict[str, Any]:
        tokens = self._anonymous.post(
            "/auth/login", json={"username": username, "password": password},
        )
        self._set_tokens(tokens)
        return dict(tokens.get("user") or {})

    def answer(self, query: str, collection: str, top_k: int) -> dict[str, Any]:
        if self._authenticated is None:
            raise RuntimeError("Client is not authenticated")
        payload = {"query": query, "collection": collection, "top_k": top_k}
        try:
            return self._authenticated.post("/answer", json=payload)
        except DashboardAPIError as exc:
            if exc.status_code != 401 or not self._refresh_token:
                raise
        tokens = self._anonymous.post(
            "/auth/refresh", json={"refresh_token": self._refresh_token},
        )
        self._set_tokens(tokens)
        assert self._authenticated is not None
        return self._authenticated.post("/answer", json=payload)

    def logout(self) -> None:
        if self._authenticated is None or not self._refresh_token:
            return
        try:
            self._authenticated.post(
                "/auth/logout", json={"refresh_token": self._refresh_token},
            )
        except DashboardAPIError:
            # Logout is best effort during CLI shutdown.  The server-side
            # refresh session also expires and can be revoked by an admin.
            pass
        finally:
            self._authenticated = None
            self._refresh_token = None

    def _set_tokens(self, tokens: dict[str, Any]) -> None:
        access_token = str(tokens.get("access_token") or "")
        refresh_token = str(tokens.get("refresh_token") or "")
        if not access_token or not refresh_token:
            raise DashboardAPIError("Authentication response did not contain both tokens")
        self._refresh_token = refresh_token
        self._authenticated = self._client_factory(
            base_url=self.base_url, access_token=access_token,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticated CLI for evidence-constrained production RAG.",
    )
    parser.add_argument("--query", "-q", help="Single query; omit for interactive mode")
    parser.add_argument("--collection", "-c", default="default", help="Collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Final retrieval count")
    parser.add_argument("--api-url", default=None, help="FastAPI base URL")
    parser.add_argument(
        "--username", default=None,
        help="Local account name; defaults to RAG_CLI_USERNAME or an interactive prompt",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show status, evidence decision, versions and trace ID",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > 100:
        parser.error("--top-k must be between 1 and 100")
    return args


def _credentials(
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
    password_fn: Callable[[str], str],
) -> tuple[str, str]:
    username = (args.username or os.environ.get("RAG_CLI_USERNAME") or "").strip()
    if not username:
        username = input_fn("Username: ").strip()
    password = os.environ.get("RAG_CLI_PASSWORD") or password_fn("Password: ")
    if not username or not password:
        raise ValueError("Username and password are required")
    return username, password


def _print_answer(result: dict[str, Any], verbose: bool) -> None:
    status = str(result.get("status") or "error")
    prefix = {
        "refused": "[REFUSED] ",
        "degraded": "[DEGRADED] ",
        "error": "[ERROR] ",
    }.get(status, "")
    print(f"\n{prefix}{result.get('answer') or 'No answer returned.'}")
    citations = result.get("citations") or []
    if citations:
        print("\nSources:")
        for item in citations:
            if isinstance(item, dict):
                source = item.get("source") or "unknown"
                page = item.get("page")
                chunk_id = item.get("chunk_id") or "unknown"
                location = f"{source} page={page}" if page is not None else str(source)
                print(f"  [{item.get('index', '?')}] {location} chunk={chunk_id}")
            else:
                print(f"  - {item}")
    if verbose:
        print("\nMetadata:")
        metadata = {
            "status": status,
            "trace_id": result.get("trace_id"),
            "prompt_version": result.get("prompt_version"),
            "model_version": result.get("model_version"),
            "evidence": result.get("evidence") or {},
        }
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


def _run_single(
    client: AuthenticatedRAGClient,
    query: str,
    collection: str,
    top_k: int,
    verbose: bool,
) -> int:
    result = client.answer(query, collection, top_k)
    _print_answer(result, verbose)
    return 1 if result.get("status") == "error" else 0


def _run_interactive(
    client: AuthenticatedRAGClient,
    collection: str,
    top_k: int,
    verbose: bool,
    input_fn: Callable[[str], str],
) -> int:
    print("\nProduction RAG CLI. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit", "bye"}:
            print("Goodbye.")
            return 0
        try:
            _run_single(client, query, collection, top_k, verbose)
        except DashboardAPIError as exc:
            print(f"[FAIL] {exc}")


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    password_fn: Callable[[str], str] = getpass.getpass,
    client_factory: Callable[..., DashboardAPIClient] = DashboardAPIClient,
) -> int:
    args = parse_args(argv)
    try:
        username, password = _credentials(args, input_fn, password_fn)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    client = AuthenticatedRAGClient(args.api_url, client_factory=client_factory)
    try:
        user = client.login(username, password)
        print(f"[OK] Logged in as {user.get('username', username)} ({user.get('role', 'user')})")
        if args.query:
            return _run_single(client, args.query, args.collection, args.top_k, args.verbose)
        return _run_interactive(
            client, args.collection, args.top_k, args.verbose, input_fn,
        )
    except DashboardAPIError as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        client.logout()


if __name__ == "__main__":
    _configure_windows_console()
    sys.exit(main())
