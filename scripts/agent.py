#!/usr/bin/env python
"""ReAct Agent CLI for the Modular RAG MCP Server.

Launches an interactive multi-turn agent session or runs a single query.

Usage:
    # Interactive multi-turn session
    python scripts/agent.py

    # Single query
    python scripts/agent.py --query "What is BM25?"

    # Specify collection and top-k
    python scripts/agent.py --query "Explain RRF" --collection docs --top-k 5

    # Verbose (show ReAct turns)
    python scripts/agent.py --query "Explain RRF" --verbose

Exit codes:
    0 - Success
    1 - Query failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.core.settings import load_settings
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.agent.react_agent import ReActAgent, AgentResponse
from src.agent.tool_registry import ToolRegistry
from src.agent.tools.hybrid_search_tool import HybridSearchTool
from src.agent.tools.semantic_search_tool import SemanticSearchTool
from src.agent.tools.keyword_search_tool import KeywordSearchTool
from src.agent.tools.document_summary_tool import DocumentSummaryTool
from src.agent.tools.list_documents_tool import ListDocumentsTool
from src.agent.memory.conversation_memory import ConversationMemory
from src.observability.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReAct Agent CLI for the Modular RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", help="Single query (omit for interactive mode)")
    parser.add_argument("--collection", "-c", default=None, help="Collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval (default: 5)")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument("--verbose", action="store_true", help="Show ReAct turns")
    parser.add_argument("--no-memory", action="store_true", help="Disable conversation memory")
    return parser.parse_args()


def _build_agent(settings, collection: Optional[str], top_k: int) -> tuple[ReActAgent, ConversationMemory]:
    coll = collection or getattr(
        getattr(settings, "agent", None), "default_collection", "default"
    )

    vector_store = VectorStoreFactory.create(settings, collection_name=coll)
    embedding_client = EmbeddingFactory.create(settings)
    llm = LLMFactory.create(settings)

    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{coll}")
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = coll
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    # Shared context sink so citations flow back to the agent
    context_sink = []

    registry = ToolRegistry()
    registry.register(
        name=HybridSearchTool.name,
        fn=HybridSearchTool(hybrid_search, default_top_k=top_k, default_collection=coll, context_sink=context_sink).run,
        description=HybridSearchTool.description,
        params_schema=HybridSearchTool.parameters_schema,
    )
    registry.register(
        name=SemanticSearchTool.name,
        fn=SemanticSearchTool(dense_retriever, default_top_k=top_k, default_collection=coll).run,
        description=SemanticSearchTool.description,
        params_schema=SemanticSearchTool.parameters_schema,
    )
    registry.register(
        name=KeywordSearchTool.name,
        fn=KeywordSearchTool(sparse_retriever, default_top_k=top_k, default_collection=coll).run,
        description=KeywordSearchTool.description,
        params_schema=KeywordSearchTool.parameters_schema,
    )
    registry.register(
        name=DocumentSummaryTool.name,
        fn=DocumentSummaryTool(vector_store, default_collection=coll).run,
        description=DocumentSummaryTool.description,
        params_schema=DocumentSummaryTool.parameters_schema,
    )
    registry.register(
        name=ListDocumentsTool.name,
        fn=ListDocumentsTool(vector_store, default_collection=coll).run,
        description=ListDocumentsTool.description,
        params_schema=ListDocumentsTool.parameters_schema,
    )

    agent = ReActAgent(settings=settings, llm=llm, tool_registry=registry)
    memory = ConversationMemory()
    return agent, memory, context_sink


def _print_response(response: AgentResponse, verbose: bool) -> None:
    if verbose and response.turns:
        print("\n" + "─" * 60)
        print("ReAct Trace:")
        for i, turn in enumerate(response.turns, 1):
            print(f"\n  Turn {i}:")
            print(f"    Thought: {turn.thought}")
            if turn.action:
                print(f"    Action: {turn.action.tool_name}")
                print(f"    Input:  {turn.action.tool_input}")
            if turn.observation:
                obs_preview = (turn.observation or "")[:200].replace("\n", " ")
                print(f"    Obs:    {obs_preview}...")
        print("─" * 60)

    print(f"\nAnswer:\n{response.answer}")
    if response.citations:
        print("\nSources:")
        for c in response.citations:
            print(f"  - {c}")
    print(
        f"\n[confidence={response.confidence:.2f}  "
        f"turns={len(response.turns)}  "
        f"tools={response.used_tools}  "
        f"elapsed={response.elapsed_ms:.0f}ms]"
    )


def _run_single(agent: ReActAgent, query: str, verbose: bool) -> int:
    print(f"\nQuestion: {query}")
    try:
        response = agent.run(query)
        _print_response(response, verbose)
        return 0
    except Exception as exc:
        print(f"[FAIL] Agent error: {exc}")
        logger.exception("Agent run failed")
        return 1


def _run_interactive(
    agent: ReActAgent,
    memory: ConversationMemory,
    verbose: bool,
    use_memory: bool,
) -> int:
    print("\nReAct Agent — interactive session. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            print("Goodbye.")
            return 0

        try:
            response = agent.run(query)
            _print_response(response, verbose)
            if use_memory:
                memory.add_turn("user", query)
                memory.add_turn("assistant", response.answer)
        except Exception as exc:
            print(f"[FAIL] {exc}")
            logger.exception("Agent interactive run failed")


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        return 2

    try:
        settings = load_settings(str(config_path))
        print(f"[OK] Config loaded: {config_path}")
    except Exception as exc:
        print(f"[FAIL] Config error: {exc}")
        return 2

    try:
        agent, memory, _ctx = _build_agent(
            settings,
            collection=args.collection,
            top_k=args.top_k,
        )
        print("[OK] Agent initialized")
    except Exception as exc:
        print(f"[FAIL] Agent init failed: {exc}")
        logger.exception("Agent init failed")
        return 2

    print("\n[*] Modular RAG — ReAct Agent")
    print("=" * 60)

    if args.query:
        return _run_single(agent, args.query, args.verbose)
    return _run_interactive(agent, memory, args.verbose, not args.no_memory)


if __name__ == "__main__":
    sys.exit(main())
