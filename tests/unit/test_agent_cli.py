from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

from scripts.agent import AuthenticatedRAGClient, main
from src.observability.dashboard.api_client import DashboardAPIError


def _tokens(access: str = "access-1", refresh: str = "refresh-1") -> dict:
    return {
        "access_token": access,
        "refresh_token": refresh,
        "user": {"id": "user-1", "username": "alice", "role": "user"},
    }


def test_agent_cli_calls_authenticated_answer_api_and_logs_out(capsys) -> None:
    anonymous = MagicMock()
    authenticated = MagicMock()
    anonymous.post.return_value = _tokens()
    authenticated.post.side_effect = [
        {
            "status": "answered",
            "answer": "BM25 is a lexical ranking function. [1]",
            "citations": [{
                "index": 1, "source": "docs/search.md", "page": 2,
                "chunk_id": "chunk-1",
            }],
            "evidence": {"accepted": True, "citation_coverage": 1.0},
            "trace_id": "trace-1",
            "prompt_version": "1.0.0",
            "model_version": "qwen3:8b",
        },
        None,
    ]
    factory = MagicMock(side_effect=[anonymous, authenticated])

    exit_code = main(
        ["--query", "What is BM25?", "--username", "alice", "--verbose"],
        password_fn=lambda _: "correct horse battery staple",
        client_factory=factory,
    )

    assert exit_code == 0
    anonymous.post.assert_called_once_with(
        "/auth/login",
        json={"username": "alice", "password": "correct horse battery staple"},
    )
    assert authenticated.post.call_args_list == [
        call("/answer", json={
            "query": "What is BM25?", "collection": "default", "top_k": 5,
        }),
        call("/auth/logout", json={"refresh_token": "refresh-1"}),
    ]
    output = capsys.readouterr().out
    assert "trace-1" in output
    assert "docs/search.md" in output


def test_agent_cli_treats_evidence_refusal_as_safe_success(capsys) -> None:
    anonymous = MagicMock()
    authenticated = MagicMock()
    anonymous.post.return_value = _tokens()
    authenticated.post.side_effect = [
        {
            "status": "refused",
            "answer": "现有知识库证据不足，无法可靠回答。",
            "citations": [],
            "evidence": {"accepted": False, "reason": "insufficient_evidence"},
            "trace_id": "trace-refused",
        },
        None,
    ]
    factory = MagicMock(side_effect=[anonymous, authenticated])

    exit_code = main(
        ["--query", "Unsupported claim", "--username", "alice"],
        password_fn=lambda _: "secret",
        client_factory=factory,
    )

    assert exit_code == 0
    assert "[REFUSED]" in capsys.readouterr().out


def test_authenticated_client_refreshes_once_after_expired_access_token() -> None:
    anonymous = MagicMock()
    expired = MagicMock()
    refreshed = MagicMock()
    anonymous.post.side_effect = [
        _tokens(),
        _tokens(access="access-2", refresh="refresh-2"),
    ]
    expired.post.side_effect = DashboardAPIError(
        "401: Access token expired", status_code=401, detail="Access token expired",
    )
    refreshed.post.return_value = {"status": "answered", "answer": "ok"}
    factory = MagicMock(side_effect=[anonymous, expired, refreshed])
    client = AuthenticatedRAGClient(client_factory=factory)
    client.login("alice", "secret")

    result = client.answer("query", "default", 5)

    assert result["status"] == "answered"
    assert anonymous.post.call_args_list[1] == call(
        "/auth/refresh", json={"refresh_token": "refresh-1"},
    )
    refreshed.post.assert_called_once_with(
        "/answer", json={"query": "query", "collection": "default", "top_k": 5},
    )


def test_agent_cli_contains_no_direct_rag_or_llm_construction() -> None:
    source = Path("scripts/agent.py").read_text(encoding="utf-8")
    assert '"/answer"' in source
    for forbidden in (
        "VectorStoreFactory",
        "EmbeddingFactory",
        "LLMFactory",
        "create_hybrid_search",
        "ReActAgent(",
    ):
        assert forbidden not in source
