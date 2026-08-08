<div align="center">

# 智能知识检索与问答系统

**Agentic RAG · Hybrid Search · MCP Protocol · Full Observability**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-1200%2B-brightgreen)
![Code](https://img.shields.io/badge/Core_Code-2600%2B_lines-blue)
![Stages](https://img.shields.io/badge/Dev_Stages-9_phases_·_68_tasks-orange)

**从零构建的生产级 Agentic RAG 系统**

ReAct Agent 自主多步推理 · Dense + BM25 混合检索 · MCP 协议服务端 · Streamlit 全链路可观测平台
6 大核心组件全部可通过配置文件零代码切换

</div>

---

## 项目亮点

| 维度 | 指标 |
|------|------|
| **代码规模** | ~2,600 行核心代码，116 个 Python 文件 |
| **检索质量** | Hit@5 = **100%**（21 题双语黄金集，四种模式全部达成）|
| **关键词检索** | BM25 Hit@1 = **90.5%**，端到端延迟仅 **14 ms** |
| **测试覆盖** | **1,200+** 自动化测试（Unit · Integration · E2E 三层金字塔）|
| **可插拔架构** | **6** 大组件：LLM · Embedding · VectorStore · Reranker · Splitter · Evaluator |
| **开发完整度** | **9** 阶段 · **68** 子任务 · 全部闭环完成 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                  用户 / Claude Desktop / CLI                      │
└───────────────────┬──────────────────────┬───────────────────────┘
                    │ MCP JSON-RPC 2.0     │ Streamlit Dashboard
                    ▼                      ▼
        ┌─────────────────┐    ┌────────────────────────────────┐
        │   MCP Server    │    │        可观测性管理平台          │
        │ (stdio 传输)    │    │  系统总览 · Agent 对话 · 知识库  │
        │ query_knowledge │    │  摄取管理 · 导入追踪 · 评估面板  │
        │ list_collections│    └────────────────────────────────┘
        │ get_doc_summary │
        └────────┬────────┘
                 │
                 ▼
        ┌────────────────────────────────────────────────────┐
        │                    ReAct Agent                     │
        │                                                    │
        │  ┌────────────┐  Thought/Action/Observation        │
        │  │  LLM 推理  │◄──────────────────────────────┐   │
        │  └────────────┘                               │   │
        │  ┌────────────┐                    ┌──────────▼─┐ │
        │  │SelfChecker │  幻觉检测 · 置信度   │ToolRegistry│ │
        │  └────────────┘  打分（0.0–1.0）    │ 5 个工具   │ │
        │  ┌────────────┐                    └────────────┘ │
        │  │ 多轮对话记忆│  滑动窗口历史管理                   │
        │  └────────────┘                                   │
        └────────┬───────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────────────────────┐
        │                  混合检索引擎                        │
        │   Query Processor（jieba 分词 + 语义过滤）            │
        │          │                     │                   │
        │   ┌──────▼──────┐     ┌────────▼──────┐           │
        │   │ Dense 检索   │     │  Sparse 检索   │           │
        │   │ BGE-m3 向量  │并行  │   BM25 倒排    │           │
        │   │ + ChromaDB  │     │  + jieba 分词  │           │
        │   └──────┬──────┘     └────────┬──────┘           │
        │          └──────────┬───────────┘                  │
        │                     ▼                              │
        │              RRF 融合（k=60）                       │
        │                     ▼                              │
        │         Cross-Encoder 精排（可插拔，可选）            │
        └────────────────────────────────────────────────────┘
                 ▲
                 │ 数据摄取（6 阶段流水线）
        ┌────────────────────────────────────────────────────┐
        │ ①完整性校验 → ②PDF解析 → ③智能分块                  │
        │ → ④LLM增强Transform → ⑤双路编码 → ⑥存储            │
        └────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────────────────────┐
        │               可插拔 Provider 层                    │
        │  LLM:       OpenAI · Azure · DeepSeek · Ollama     │
        │  Embedding: OpenAI · SiliconFlow · Azure · Ollama   │
        │  VectorStore: ChromaDB（Qdrant / Milvus 规划中）    │
        └────────────────────────────────────────────────────┘
```

---

## 核心模块

### Agentic RAG — 自主多步推理

从零实现 ReAct（Reasoning + Acting）主循环，不依赖 LangChain Agent 框架：

- **ToolRegistry**：Registry 模式，工具注册与 LLM 调度解耦，新增工具只需一行 `register()` 调用
- **5 个 RAG 工具**：`hybrid_search` · `semantic_search` · `keyword_search` · `document_summary` · `list_documents`
- **SelfChecker**：LLM-as-judge，在 Agent 给出最终答案后判断答案是否有文档支撑，输出置信度（0.0–1.0）
- **ConversationMemory**：滑动窗口多轮历史管理，支持 `format_for_prompt()` 直接注入 Prompt
- **流式生成器**：`run_stream()` 每完成一个推理步骤立即 `yield AgentStreamEvent`，Dashboard 实时展示推理轨迹
- **防死循环**：`max_turns` 硬性截断 + `Final Answer:` 正则检测双重保险，超时 fallback 不返回空响应

```
Thought → Action → Observation → Thought → ... → Final Answer
              ↑每步 yield，Dashboard 实时刷新↑
```

---

### Hybrid Search — 混合检索引擎

| 路径 | 技术 | 优势 |
|------|------|------|
| Dense 检索 | BGE-m3（1024-dim）+ ChromaDB HNSW | 语义理解，跨语言，同义词命中 |
| Sparse 检索 | BM25 + jieba 中文分词 | 精确匹配，专有名词，低延迟（14ms）|
| 融合策略 | RRF（Reciprocal Rank Fusion，k=60） | 无需归一化，rank-agnostic，稳定可靠 |
| 精排（可选）| Cross-Encoder / LLM Rerank | 高精度场景最终排序优化 |

**为什么选 RRF 而非加权求和**：Dense 输出 cosine 相似度，BM25 输出 TF-IDF 分数，量纲完全不同。RRF 只看排名不看分值，天然规避了归一化问题，实测无需调参。

并行检索实现：`ThreadPoolExecutor(max_workers=2)` 并发执行双路，任意一路失败自动 Graceful Degradation，不中断整个查询。

---

### 数据摄取流水线 — 6 阶段智能处理

```
①完整性校验  →  ②PDF解析  →  ③递归分块  →  ④Transform  →  ⑤双路编码  →  ⑥存储
SHA256幂等        MarkItDown      chunk_size      ChunkRefiner     Dense+Sparse    ChromaDB
hash guard       → Markdown       =1000          MetadataEnricher  并行批处理      + BM25索引
                 图像提取占位符    overlap=200    ImageCaptioner               + ImageStore
```

- **幂等摄取**：SHA256 hash 存入 SQLite，重复文件直接跳过，任何时候重跑都安全
- **LLM 增强 Transform**：ChunkRefiner（去噪重组）· MetadataEnricher（自动补 Title/Tags/Summary）· ImageCaptioner（Vision LLM 生成图像描述）
- **Image-to-Text**：PDF 图像 → Vision LLM Caption → 缝入 Chunk 文本 → 走统一检索链路，无需引入 CLIP 等多模态向量库
- **LLM 失败 Fallback**：Transform 三步骤均支持 LLM 方案和规则方案，LLM 不可用时自动切换

---

### MCP Server — 接入 Claude Desktop

实现标准 **JSON-RPC 2.0 + stdio transport** 协议：

```json
// claude_desktop_config.json 添加一行即可接入
{ "mcpServers": { "rag": { "command": "python", "args": ["-m", "main"] } } }
```

暴露 3 个 MCP Tools：`query_knowledge_hub` · `list_collections` · `get_document_summary`

stdout 纯净输出 JSON-RPC 响应，stderr 输出结构化日志，符合 MCP 协议规范。

---

### 可观测性平台 — Streamlit Dashboard

| 页面 | 功能 |
|------|------|
| 系统总览 | 配置摘要、各组件健康状态、知识库统计 |
| 知识库浏览 | Chunk 列表、元数据查看、图像预览 |
| 文档摄取 | 文件上传、实时进度条、`on_progress` 回调 |
| 摄取追踪 | 6 阶段耗时、成功/失败状态 |
| 查询追踪 | Dense vs Sparse 结果对比、Rerank 前后变化 |
| 评估面板 | 运行 Ragas / Custom 评估，查看 Faithfulness/MRR 等指标 |
| **Agent 对话** | 多会话 ChatGPT 风格，ReAct 推理轨迹**实时流式可视化** |

**Agent 流式推理轨迹**（使用 `st.empty()` 原位刷新）：

```
💭 THOUGHT: 需要先搜索 BM25 的相关内容...       ← Turn 1 完成
⚡ ACTION:  hybrid_search  ● 工具执行中…
────────────────────────────────────────
💭 THOUGHT: 找到了相关定义，再补充对比信息...    ← Turn 2 推理中...
⚡ ACTION:  semantic_search  ● 工具执行中…
         ↓  done 事件到达
[最终答案] · [置信度 0.87] · [2 轮] · [1.3s]
```

---

### 可插拔架构 — Factory + 配置驱动

6 大组件均遵循同一模式：`Base 抽象类 → Factory 注册 → YAML 配置切换`

```yaml
# 切换 LLM Provider：只改 settings.yaml，零代码修改
llm:
  provider: "DeepSeek"     # openai / azure / deepseek / ollama
  model: "deepseek-v4-flash"
  api_key: "sk-..."
```

```python
# Factory Pattern：新增 Provider 3 步完成
class MoonshotLLM(BaseLLM):
    def chat(self, messages): ...           # Step 1: 实现接口

LLMFactory.register_provider("moonshot", MoonshotLLM)  # Step 2: 注册

# settings.yaml: provider: "moonshot"      # Step 3: 配置
```

同一 Factory 模式应用于：`LLMFactory` · `EmbeddingFactory` · `VectorStoreFactory` · `RerankerFactory` · `SplitterFactory` · `EvaluatorFactory`

---

## 评测结果

**测试集**：21 条手工标注双语问答对（中文 + 英文技术文档，70 chunks，覆盖事实型、对比型、概念型、多步推理型）

| 检索模式 | Hit@1 | Hit@5 | MRR@10 | 平均延迟 |
|---------|-------|-------|--------|---------|
| Dense Only（BGE-m3） | 66.7% | **100%** | 0.794 | 315 ms |
| **Sparse Only（BM25）** | **90.5%** | **100%** | **0.952** | **14 ms** |
| Hybrid / RRF 融合 | 76.2% | **100%** | 0.881 | 259 ms |

> 四种模式 Hit@5 均达到 100%，验证知识库构建质量和分块策略合理性。完整方法论与逐题分析见 [EVALUATION_REPORT.md](EVALUATION_REPORT.md)。

**Hybrid vs Dense 相对提升**：Hit@1 +9.5pp，MRR@10 +11.0%

**BM25 在中文技术领域的优势**：领域专有名词（BM25、RRF、ChromaDB、LoRA 等）精确匹配能力强，且延迟为 Dense 的 **1/22**（14ms vs 315ms）。

---

## 工程实践

### 三层测试金字塔

```
tests/
├── unit/               # 快速，纯 Python，无外部依赖
│   ├── test_rrf_fusion.py         # RRF 算法正确性验证
│   ├── test_react_agent.py        # Agent 循环（Mock LLM + Mock 工具）
│   ├── test_conversation_memory.py
│   └── test_self_checker.py
│
├── integration/        # 需要 ChromaDB / LLM API
│   ├── test_ingestion_pipeline.py
│   ├── test_hybrid_search.py
│   └── test_mcp_server.py
│
└── e2e/                # 完整业务流程验证
    ├── test_data_ingestion.py     # 上传 → 检索 → 验证
    ├── test_mcp_client.py         # 真实 MCP 协议调用
    └── test_recall.py             # Golden Set 召回率回归
```

```bash
pytest tests/unit -v                         # 仅单元测试（秒级）
pytest tests/ -m "not llm" -v               # 跳过需要 API 的测试
pytest tests/ --cov=src --cov-report=html   # 生成覆盖率报告
```

### 关键设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| 检索融合 | RRF（rank-based）而非加权求和 | 无需归一化，免调参，鲁棒性强 |
| 图像检索 | Image-to-Text（Vision LLM Caption）| 复用纯文本链路，无需 CLIP，部署简单 |
| 流式 Agent | Turn 级 `yield AgentStreamEvent` | 比 Token 级流式延迟更低，UI 实现更简洁 |
| 摄取幂等 | SHA256 hash + SQLite 记录 | 任何时候重跑脚本都安全，无重复数据 |
| LLM Fallback | Transform 阶段 LLM → 规则 | LLM 不可用时系统降级继续运行 |
| 数据契约 | `src/core/types.py` dataclass | 模块间解耦，只通过 `Chunk` / `RetrievalResult` 等类型通信 |

---

## 技术栈

| 层次 | 技术选型 |
|------|---------|
| **Agent** | 自研 ReAct 主循环 · SelfChecker · ConversationMemory |
| **检索** | ChromaDB HNSW · rank-bm25 · jieba · RRF Fusion |
| **精排** | sentence-transformers（Cross-Encoder）· LLM Rerank |
| **LLM / Embedding** | OpenAI · Azure · DeepSeek · Ollama · SiliconFlow |
| **MCP 协议** | `mcp` SDK · JSON-RPC 2.0 · stdio transport |
| **可观测性** | Streamlit · TraceContext · JSONL 结构化日志 |
| **评估** | Ragas · 自定义 Hit@K / MRR@K 指标 |
| **数据解析** | MarkItDown（PDF → Markdown）· langchain-text-splitters |
| **运行时** | Python 3.10+ · uv 包管理 |
| **测试** | pytest · Unit / Integration / E2E 三层 |

---

## 快速开始

```bash
# 1. 克隆并安装
git clone <repo-url>
cd <project-dir>
pip install uv && uv sync

# 2. 配置 API Key
cp config/settings.yaml config/settings.local.yaml
# 编辑 llm.api_key 和 embedding.api_key

# 3. 摄取文档
python scripts/ingest.py --source path/to/your/docs

# 4. 启动 Streamlit 管理平台
python scripts/start_dashboard.py

# 5. 命令行单次查询
python scripts/query.py "RRF 算法的原理是什么？"

# 6. 命令行多轮 Agent 对话
python scripts/agent.py

# 7. 运行检索评测基准
python scripts/run_benchmark.py

# 8. 接入 Claude Desktop（MCP 模式）
# claude_desktop_config.json 添加：
# {"mcpServers": {"rag": {"command": "python", "args": ["-m", "main"]}}}
python -m main
```

**支持的 LLM Provider**：`openai` · `azure` · `deepseek` · `ollama`  
**支持的 Embedding Provider**：`openai` · `azure` · `siliconflow` · `ollama`

---

## 项目结构

```
src/
├── agent/              # ReAct Agent 核心
│   ├── react_agent.py  # 主循环（run / run_stream）
│   ├── tool_registry.py
│   ├── tools/          # 5 个 RAG 工具实现
│   ├── memory/         # ConversationMemory
│   └── reflection/     # SelfChecker（LLM 幻觉检测）
│
├── core/
│   ├── query_engine/   # HybridSearch · DenseRetriever · SparseRetriever · RRF
│   ├── response/       # 答案组装 · Citation 生成
│   ├── trace/          # TraceContext · TraceCollector
│   ├── types.py        # 全项目数据契约（Document / Chunk / RetrievalResult）
│   └── settings.py     # YAML 配置加载与验证
│
├── ingestion/          # 6 阶段摄取流水线
│   ├── pipeline.py     # IngestionPipeline.run()
│   ├── chunking/       # RecursiveCharacterTextSplitter 适配
│   ├── embedding/      # Dense + Sparse 双路批处理
│   ├── storage/        # ChromaDB · BM25 Index · Image Store
│   └── transform/      # ChunkRefiner · MetadataEnricher · ImageCaptioner
│
├── libs/               # 可插拔抽象层
│   ├── llm/            # BaseLLM + OpenAI/Azure/Ollama/DeepSeek 实现
│   ├── embedding/      # BaseEmbedding + 各 Provider 实现
│   ├── reranker/       # BaseReranker + Cross-Encoder / LLM Rerank
│   ├── splitter/       # BaseSplitter + RecursiveSplitter
│   ├── vector_store/   # BaseVectorStore + ChromaDB 实现
│   ├── evaluator/      # BaseEvaluator + Ragas / Custom / Composite
│   └── loader/         # PDF 解析 · 文件完整性校验
│
├── mcp_server/         # MCP Server 实现
│   ├── server.py       # stdio 传输入口
│   ├── protocol_handler.py
│   └── tools/          # query_knowledge_hub / list_collections / get_doc_summary
│
└── observability/      # 可观测性层
    ├── logger.py        # 结构化日志
    ├── evaluation/      # Ragas + Custom 评估运行器
    └── dashboard/       # Streamlit 7 页面管理平台

scripts/
├── ingest.py            # 文档摄取 CLI
├── query.py             # 单次查询 CLI
├── agent.py             # 多轮 Agent 对话 CLI
├── run_benchmark.py     # 4 模式检索基准测试
└── evaluate.py          # Ragas 评估运行器

config/
├── settings.yaml        # 全局配置（LLM · Embedding · 检索 · Agent）
└── prompts/
    └── react_agent.txt  # ReAct Prompt 模板（外置，支持热更新）

tests/
├── unit/                # 单元测试（无外部依赖）
├── integration/         # 集成测试（需 ChromaDB / LLM API）
└── e2e/                 # 端到端测试（完整业务流程）
```

---

## 深入了解

| 文档 | 内容 |
|------|------|
| [TECHNICAL_DOC.md](TECHNICAL_DOC.md) | 架构设计详解、关键算法、设计模式、面试高频问答 |
| [EVALUATION_REPORT.md](EVALUATION_REPORT.md) | 评测方法论、逐题分析、配置建议、可复现脚本 |

---

## License

MIT
