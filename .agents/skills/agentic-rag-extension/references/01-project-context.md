# Project Context: MODULAR-RAG-MCP-SERVER

## 项目背景

这是一个求职用主项目，作者：2本计算机科学与技术应届生，无实习，目标岗位为 Agent 方向。

项目当前状态（A~I 阶段全部完成，共68个任务）：
- 完整 RAG 链路：PDF摄取 → 分块 → BM25+向量 Hybrid Search → Rerank → MCP Tool 响应
- MCP Server（stdio）：`query_knowledge_hub` / `list_collections` / `get_document_summary`
- 可观测性 Dashboard（Streamlit 6页面）
- 评估体系（RagasEvaluator + CompositeEvaluator + EvalRunner，代码已有但未跑真实数据）

## 目录结构关键路径

```
src/
├── core/
│   ├── query_engine/         # HybridSearch, DenseRetriever, SparseRetriever, RRFFusion, Reranker
│   ├── response/             # ResponseBuilder, CitationGenerator, MultimodalAssembler
│   ├── trace/                # TraceContext, TraceCollector
│   └── types.py              # Document, Chunk, ChunkRecord, RetrievalResult, ProcessedQuery
├── ingestion/                # Pipeline, DocumentChunker, BatchProcessor, BM25Indexer, VectorUpserter
├── libs/
│   ├── llm/                  # BaseLLM + OpenAI/Azure/DeepSeek/Ollama 实现
│   ├── embedding/            # BaseEmbedding + 各实现
│   ├── reranker/             # BaseReranker + LLMReranker + CrossEncoderReranker
│   ├── evaluator/            # BaseEvaluator + RagasEvaluator + CompositeEvaluator
│   └── vector_store/         # ChromaStore
├── mcp_server/               # MCP protocol + tools
└── observability/
    ├── dashboard/pages/       # overview, data_browser, ingestion_mgmt, ingestion_traces, query_traces, evaluation_panel
    └── logger.py              # JSON Lines trace logger
scripts/
├── ingest.py                 # CLI: python scripts/ingest.py --path <dir> --collection <name>
├── query.py                  # CLI: python scripts/query.py "<question>"
└── start_dashboard.py        # CLI: python scripts/start_dashboard.py
config/settings.yaml          # 全局配置：llm/embedding/vector_store/retrieval/rerank/evaluation
tests/
├── unit/ integration/ e2e/
└── fixtures/                 # golden_test_set.json (待创建)
```

## 核心接口速查

```python
# HybridSearch - 核心检索入口
results: List[RetrievalResult] = hybrid_search.search(query, top_k, filters, trace)

# RetrievalResult 结构
@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict  # source_path, collection, chunk_index, ...

# BaseLLM - LLM调用
response = llm.chat(messages=[{"role":"user","content":"..."}])

# Settings - 配置访问
settings = load_settings("config/settings.yaml")
settings.llm.provider / settings.retrieval.top_k / ...
```

## 技术栈

| 层次 | 技术 |
|------|------|
| LLM | OpenAI / Azure OpenAI / DeepSeek / Ollama（可配置切换） |
| Embedding | OpenAI / Azure / Ollama |
| Vector Store | ChromaDB（本地持久化） |
| BM25 | 自研倒排索引（data/db/bm25/） |
| Reranker | LLM Reranker / Cross-Encoder / None（可配置） |
| Evaluator | Ragas + 自定义指标（Hit@K, MRR） |
| MCP | anthropic/mcp SDK，stdio 协议 |
| Dashboard | Streamlit multi-page |
| Trace | JSON Lines（data/traces/） |
