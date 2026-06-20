# Modular RAG MCP Server

> 生产级 Agentic RAG 系统 — ReAct Agent · 混合检索 · MCP 协议 · 全链路可观测性

A production-grade **Agentic RAG** framework built from scratch. Features a ReAct Agent with self-checking, Hybrid Search (Dense + BM25 + RRF), Model Context Protocol (MCP) server compatible with Claude Desktop, and full observability via Streamlit Dashboard.

---

## Benchmark Results

21-query bilingual test set (Chinese + English technical docs, 70 chunks):

| Retrieval Mode | Hit@1 | Hit@5 | MRR@10 | Avg Latency |
|---|---|---|---|---|
| Dense Only (BGE-m3) | 66.7% | **100%** | 0.794 | 315 ms |
| Sparse Only (BM25) | **90.5%** | **100%** | **0.952** | **14 ms** |
| Hybrid / RRF Fusion | 76.2% | **100%** | 0.881 | 259 ms |

> All modes achieve Hit@5 = 100%. Full methodology in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               User / Claude Desktop / CLI                    │
└───────────────┬──────────────────────────┬───────────────────┘
                │ MCP JSON-RPC             │ Streamlit
                ▼                          ▼
    ┌───────────────────┐     ┌────────────────────────────┐
    │    MCP Server     │     │    Observability Dashboard │
    │ (stdio transport) │     │  Overview · Agent Chat ·   │
    │  query_knowledge  │     │  Ingestion · Traces · Eval │
    └────────┬──────────┘     └────────────────────────────┘
             │
             ▼
    ┌───────────────────────────────────────────────────────┐
    │                    ReAct Agent                        │
    │  ┌──────────────┐  ┌───────────┐  ┌───────────────┐  │
    │  │ Tool Registry│  │SelfChecker│  │  Conversation │  │
    │  │ 5 RAG tools  │  │(LLM judge)│  │    Memory     │  │
    │  └──────────────┘  └───────────┘  └───────────────┘  │
    └────────┬──────────────────────────────────────────────┘
             │
             ▼
    ┌───────────────────────────────────────────────────────┐
    │                    RAG Core                           │
    │  Dense Search    BM25 Search      Reranker            │
    │  (ChromaDB)  +  (jieba+rank_bm25) (Cross-Encoder)    │
    │                      │                                │
    │              RRF Fusion (k=60)                        │
    └───────────────────────────────────────────────────────┘
             │
             ▼
    ┌───────────────────────────────────────────────────────┐
    │           Pluggable Provider Layer                    │
    │  LLM: OpenAI · Azure · DeepSeek · Ollama             │
    │  Embedding: OpenAI · SiliconFlow · Ollama            │
    │  VectorStore: ChromaDB (Qdrant / Milvus planned)     │
    └───────────────────────────────────────────────────────┘
```

---

## Key Features

**Agentic RAG**
- ReAct main loop with multi-step reasoning and tool use
- 5 built-in tools: `query_knowledge`, `search_by_keyword`, `get_document_list`, `calculate`, `get_system_status`
- SelfChecker: LLM-based hallucination detection and answer validation
- ConversationMemory: sliding-window context for multi-turn dialogue

**Hybrid Search**
- Dense retrieval (BGE-m3 via SiliconFlow or any OpenAI-compatible embedding)
- Sparse retrieval (BM25 with jieba Chinese tokenization)
- RRF (Reciprocal Rank Fusion) score merging — no hyperparameter tuning needed
- Optional Cross-Encoder reranker for precision-critical scenarios

**MCP Protocol**
- Full JSON-RPC 2.0 over stdio transport
- Plug into Claude Desktop with a one-line config addition
- Exposes `query_knowledge`, `ingest_document`, `list_documents` as MCP tools

**Full-Stack Observability**
- `TraceContext` captures per-stage latency and intermediate results for every query
- Streamlit Dashboard: Overview metrics, Agent Chat, Ingestion Manager, Query Traces, Evaluation Panel
- Structured logging throughout

**Evaluation Pipeline**
- Ragas integration + custom Hit@K / MRR@K metrics
- Golden test set with 21 hand-labeled bilingual QA pairs
- Reproducible benchmark scripts; one-click run from Dashboard

**Pluggable Architecture**
- 6 swappable layers: LLM · Embedding · VectorStore · Reranker · Splitter · Evaluator
- Switch providers by editing `config/settings.yaml` — zero code changes required
- Abstract factory pattern with dependency injection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent | Custom ReAct loop, SelfChecker, ConversationMemory |
| Retrieval | ChromaDB, rank-bm25, jieba, RRF |
| Reranker | sentence-transformers (Cross-Encoder) |
| LLM / Embedding | OpenAI / Azure / DeepSeek / Ollama / SiliconFlow |
| MCP | `mcp` SDK, JSON-RPC 2.0, stdio transport |
| Dashboard | Streamlit |
| Evaluation | Ragas, custom metrics |
| Runtime | Python 3.10+, uv |
| Testing | pytest (unit · integration · e2e) |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd modular-rag-mcp-server
pip install uv && uv sync

# 2. Configure API keys
cp config/settings.yaml  # edit llm.api_key and embedding.api_key

# 3. Ingest documents
python scripts/ingest.py --source path/to/your/docs

# 4. Launch Dashboard
streamlit run src/observability/dashboard/app.py

# 5. Query via CLI
python scripts/query.py "What is the RRF algorithm?"

# 6. Use as MCP Server (add to Claude Desktop config)
# {"mcpServers": {"rag": {"command": "python", "args": ["-m", "main"]}}}
python -m main
```

Supported LLM providers: `openai` · `azure` · `deepseek` · `ollama`  
Supported Embedding providers: `openai` · `azure` · `siliconflow` · `ollama`

---

## Project Structure

```
src/
├── agent/              # ReAct Agent, tool registry, memory, self-checker
│   ├── react_agent.py
│   ├── tool_registry.py
│   ├── tools/          # query, search, list, calculate, status
│   ├── memory/         # ConversationMemory
│   └── reflection/     # SelfChecker (LLM hallucination judge)
├── core/               # Config, settings, DI container
├── ingestion/          # Document parsing (PDF→MD), chunking, embedding pipeline
├── libs/               # Abstract LLM / Embedding / Reranker / Splitter
├── mcp_server/         # MCP server + tool handlers
└── observability/      # Logger, TraceContext, Streamlit Dashboard
scripts/
├── ingest.py           # Ingest documents from CLI
├── query.py            # Single-turn query from CLI
├── agent.py            # Multi-turn agent session from CLI
├── run_benchmark.py    # 4-mode retrieval benchmark
└── evaluate.py         # Ragas evaluation runner
config/
└── settings.yaml       # All configuration in one file
tests/
├── unit/               # Per-module unit tests (no external deps)
├── integration/        # Cross-module integration tests
└── e2e/                # Full pipeline end-to-end tests
```

---

## Documents

| Document | Description |
|---|---|
| [TECHNICAL_DOC.md](TECHNICAL_DOC.md) | Architecture deep-dive, algorithm design, key tradeoffs, interview Q&A |
| [EVALUATION_REPORT.md](EVALUATION_REPORT.md) | Benchmark methodology, results analysis, reproducible scripts |

---

## Design Highlights

**Why RRF over weighted sum for score fusion?**  
RRF is rank-based, so it's immune to score distribution differences between Dense and BM25 retrievers — no calibration needed.

**Why two-stage retrieval (coarse → fine)?**  
Dense/BM25 recall cheap candidates at low cost; Cross-Encoder reranker scores the top-K precisely. This keeps latency manageable without sacrificing final precision.

**Why ReAct over single-pass RAG?**  
Multi-step queries (comparison, multi-hop) can't be answered in one retrieval pass. ReAct lets the agent decompose the question, retrieve incrementally, and validate its own answer via SelfChecker.

---

## License

MIT
