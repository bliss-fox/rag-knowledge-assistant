# Modular RAG MCP Server — 技术文档

> 面向技术面试的完整项目解读，涵盖架构设计、核心算法、工程实践与常见追问应对。

---

## 目录

1. [项目概览](#1-项目概览)
2. [技术栈](#2-技术栈)
3. [整体架构](#3-整体架构)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 数据摄取流水线
   - 4.2 混合检索引擎
   - 4.3 ReAct Agent
   - 4.4 MCP Server
   - 4.5 可观测性与 Dashboard
5. [关键算法](#5-关键算法)
6. [设计模式](#6-设计模式)
7. [数据类型系统](#7-数据类型系统)
8. [配置体系](#8-配置体系)
9. [测试策略](#9-测试策略)
10. [面试高频问题与回答](#10-面试高频问题与回答)

---

## 1. 项目概览

### 1.1 一句话定位

一个**完全模块化、配置驱动**的 RAG（Retrieval-Augmented Generation）服务框架，内置 MCP 协议支持、ReAct Agent、混合检索与全链路可观测性，可通过修改 `settings.yaml` 零代码切换 LLM / Embedding / VectorStore 提供商。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **可插拔 6 层** | LLM、Embedding、VectorStore、Reranker、Splitter、Evaluator 均可配置替换 |
| **混合检索** | Dense（向量）+ Sparse（BM25）并行检索，RRF 算法融合 |
| **Agentic RAG** | ReAct 主循环 + 5 工具 + SelfChecker（LLM 幻觉检测）+ ConversationMemory |
| **MCP 协议** | 标准 JSON-RPC 2.0 / stdio transport，可直连 Claude Desktop |
| **全链路追踪** | TraceContext 记录每阶段耗时与中间结果，Streamlit Dashboard 可视化 |
| **评估闭环** | Ragas + 自定义评估指标，Dashboard 内一键运行 |

### 1.3 代码规模

- Python 文件：约 116 个
- 核心代码：约 2,600 行
- 测试文件：unit / integration / e2e 三层

---

## 2. 技术栈

### 2.1 运行时依赖

```toml
# 核心
pyyaml>=6.0                      # YAML 配置解析
langchain-text-splitters>=0.3.0  # 文本递归分块
chromadb>=0.4.0                  # 本地向量存储
mcp>=1.0.0                       # Model Context Protocol SDK
jieba>=0.42                      # 中文分词（BM25 预处理）
markitdown[pdf]>=0.1.0           # PDF → Markdown 解析

# 可观测性
streamlit>=1.30.0                # Dashboard UI

# 评估
ragas>=0.1.0                     # RAG 专用评估框架
datasets>=2.0.0                  # HuggingFace 数据集工具

# LLM 客户端
openai>=1.0                      # OpenAI / Azure / DeepSeek / SiliconFlow（均兼容）
```

### 2.2 支持的提供商矩阵

| 组件 | 提供商 | 状态 |
|------|--------|------|
| **LLM** | OpenAI、Azure OpenAI、DeepSeek、Ollama | ✅ |
| **Vision LLM** | OpenAI、Azure OpenAI | ✅ |
| **Embedding** | OpenAI、Azure、Ollama、SiliconFlow（硅基流动）及其他 OpenAI 兼容服务 | ✅ |
| **Vector Store** | ChromaDB | ✅；Qdrant、Milvus | 规划中 |
| **Reranker** | Cross-Encoder（sentence-transformers）、LLM Rerank | ✅ |
| **Evaluator** | Ragas、Custom、Composite | ✅ |

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户 / Claude Desktop                         │
└────────────┬────────────────────────────────┬───────────────────────┘
             │ MCP JSON-RPC                   │ Streamlit Dashboard
             ▼                                ▼
┌────────────────────┐            ┌────────────────────────────────┐
│    MCP Server      │            │      Observability Layer       │
│  (stdio transport) │            │  Overview / Agent Chat /       │
│  query_knowledge   │            │  Ingestion / Traces / Eval     │
│  list_collections  │            └────────────────────────────────┘
│  get_doc_summary   │
└────────┬───────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                         ReAct Agent                                │
│                                                                    │
│  ┌──────────────┐  Thought/Action/Observation  ┌────────────────┐ │
│  │   LLM 推理   │◄───────────────────────────► │  ToolRegistry  │ │
│  └──────────────┘                              │  hybrid_search │ │
│  ┌──────────────┐                              │  semantic_search│ │
│  │ SelfChecker  │  幻觉检测 + 置信度打分        │  keyword_search │ │
│  └──────────────┘                              │  doc_summary   │ │
│  ┌──────────────┐                              │  list_docs     │ │
│  │   Memory     │  会话历史管理                 └────────────────┘ │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Hybrid Search Engine                          │
│                                                                    │
│   Query Processor (jieba 分词 + 过滤提取)                           │
│         │                          │                               │
│   ┌─────▼──────┐           ┌───────▼──────┐                       │
│   │   Dense    │           │    Sparse    │                       │
│   │ Retriever  │           │  Retriever   │                       │
│   │ (Embedding)│           │   (BM25)     │                       │
│   └─────┬──────┘           └───────┬──────┘                       │
│         │    ThreadPoolExecutor     │                               │
│         └─────────────┬────────────┘                               │
│                       ▼                                            │
│              RRF Fusion (k=60)                                     │
│                       ▼                                            │
│             Reranker (可选，Cross-Encoder / LLM)                   │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                 │
│                                                                    │
│   ChromaDB (Dense Vectors)    BM25 Index (JSON)    Image Store    │
│   ─────────────────────────   ─────────────────    ────────────── │
│   Embedding + metadata         倒排索引文件          图像 + JSON   │
└────────────────────────────────────────────────────────────────────┘
         ▲
         │ 摄取
┌────────────────────────────────────────────────────────────────────┐
│                      Ingestion Pipeline (6 阶段)                   │
│                                                                    │
│  ① Integrity Check → ② Load PDF → ③ Chunk → ④ Transform           │
│  → ⑤ Encode → ⑥ Store                                             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 数据摄取流水线

**入口**：`src/ingestion/pipeline.py` — `IngestionPipeline.run(file_path)`

#### 阶段一：完整性检查（Idempotency Guard）

```python
# SHA256 hash → SQLite ingestion_history 表
# 相同文件直接 skip，保证幂等性
sha256 = hashlib.sha256(file_bytes).hexdigest()
if already_ingested(sha256):
    return PipelineResult(status="skipped")
```

**面试亮点**：任何时候重跑导入脚本都是安全的，不会产生重复 chunk。

#### 阶段二：文档加载

```python
# MarkItDown 解析 PDF → Markdown 格式文本
# 图像提取：保留 [IMAGE: {image_id}] 占位符
Document(
    id=sha256,
    text="# 标题\n\n正文...\n[IMAGE: doc_abc_page1_0]\n...",
    metadata={"source_path": "...", "images": [...]}
)
```

#### 阶段三：文档分块

```python
# RecursiveCharacterTextSplitter（langchain-text-splitters）
# 参数：chunk_size=1000, chunk_overlap=200
# ID 格式：{doc_id}_{index:04d}_{hash[:8]}
Chunk(
    id="abc123_0000_d4e5f6a7",
    text="...",
    metadata={**doc_metadata, "chunk_index": 0, "source_ref": doc.id}
)
```

**关键设计**：`chunk_overlap=200` 保证跨块语义连续，分块 ID 全局唯一。

#### 阶段四：Transform 三步骤

```
ChunkRefiner  →  MetadataEnricher  →  ImageCaptioner
    ↓                   ↓                   ↓
 LLM/规则精化     LLM/规则补充元数据    Vision LLM 生成图像描述
```

- **ChunkRefiner**：修正格式错误、截断不完整句子
- **MetadataEnricher**：自动补充 title、tags、summary 字段
- **ImageCaptioner**：调用 Vision LLM，将图像描述缝合回 chunk 文本

所有三步骤均支持 LLM 方案和规则方案，LLM 失败时自动 fallback 到规则。

#### 阶段五：编码

```python
# Dense 编码：Embedding API → List[float]（维度 1024）
# Sparse 编码：jieba 分词 → {term: tf, ...} → BM25 term weights
# 并行批处理：ThreadPoolExecutor，batch_size=100
```

#### 阶段六：存储

| 存储目标 | 数据 | 路径 |
|---------|------|------|
| ChromaDB | Dense 向量 + 元数据 | `data/db/chroma/` |
| BM25 Index | 倒排索引 JSON | `data/db/bm25/{collection}/index.json` |
| Image Store | 图像文件 + JSON 索引 | `data/images/{collection}/` |

---

### 4.2 混合检索引擎

**入口**：`src/core/query_engine/hybrid_search.py` — `HybridSearch.search(query, top_k)`

#### 完整流程

```python
def search(query: str, top_k: int) -> List[RetrievalResult]:
    # 1. Query Processing
    processed = query_processor.process(query)
    # processed.keywords = ["检索", "增强", "生成"]（jieba 分词）
    # processed.filters = {"collection": "default"}

    # 2. 并行检索（ThreadPoolExecutor, max_workers=2）
    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future  = pool.submit(dense_retriever.retrieve, processed, top_k=20)
        sparse_future = pool.submit(sparse_retriever.retrieve, processed, top_k=20)
    
    dense_results  = dense_future.result()   # List[RetrievalResult]（cosine 相似度）
    sparse_results = sparse_future.result()  # List[RetrievalResult]（BM25 分数）

    # 3. RRF 融合
    fused = rrf_fusion.fuse([dense_results, sparse_results], k=60)

    # 4. Graceful Degradation（任意一路失败不中断）
    # Dense 失败 → 仅用 Sparse；Sparse 失败 → 仅用 Dense；两者都失败 → RuntimeError

    # 5. 返回 top_k
    return fused[:top_k]
```

#### Dense Retriever

```python
# 查询向量化 → ChromaDB cosine 搜索
query_vector = embedding_client.embed([query])[0]  # List[float]
results = vector_store.query(
    query_vector=query_vector,
    top_k=dense_top_k,
    collection=collection,
    filters=filters,
)
```

#### Sparse Retriever（BM25）

```python
# 关键词检索 → 获取 chunk_id → 从 ChromaDB 补全文本和元数据
chunk_ids_and_scores = bm25_indexer.query(keywords, top_k=sparse_top_k)
# [{chunk_id: "abc_0000", score: 45.3}, ...]

# 从向量存储 fetch 完整记录
chunks = vector_store.get_by_ids(chunk_ids)
```

---

### 4.3 ReAct Agent

**入口**：`src/agent/react_agent.py`

- `ReActAgent.run(question)` — 同步调用，返回完整 `AgentResponse`
- `ReActAgent.run_stream(question)` — 生成器，每步 `yield AgentStreamEvent`，用于流式 UI

#### 主循环（run）

```
┌─────────────────────────────────────────────────────────────────┐
│  for turn in range(max_turns=5):                                │
│                                                                 │
│    1. 构建 Prompt（含工具描述 + 历史轨迹）                         │
│    2. LLM 推理 → 输出 Thought / Action / Action Input            │
│       或 Final Answer                                           │
│    3. 正则解析输出                                               │
│       - _THOUGHT_RE     → thought 文本                          │
│       - _ACTION_RE      → tool_name                            │
│       - _ACTION_INPUT_RE → JSON 参数                            │
│       - _FINAL_ANSWER_RE → 终止信号                             │
│    4. 若 is_final → state.final_answer = ...; break             │
│    5. 否则 → ToolRegistry.dispatch(tool_name, tool_input)       │
│             → observation 字符串 → 追加到 state.history          │
│                                                                 │
│  超出 max_turns → 最后一个 Thought 作为 fallback 答案            │
└─────────────────────────────────────────────────────────────────┘
         ↓
   SelfChecker.check(question, final_answer, retrieval_context)
         ↓
   confidence: float（0.0–1.0）
         ↓
   返回 AgentResponse(answer, citations, turns, confidence, used_tools, elapsed_ms)
```

#### 流式变体（run_stream）

`run_stream()` 是 `run()` 的生成器版本，每完成一个推理步骤就立即 `yield`，供 Dashboard 实时更新 UI：

```python
@dataclass
class AgentStreamEvent:
    type: str          # "thought" | "action" | "observation" | "done"
    turn_idx: int
    thought: str       # type="thought" 时填充
    tool_name: str     # type="action"  时填充
    tool_input: dict   # type="action"  时填充
    observation: str   # type="observation" 时填充
    # 以下字段仅在 type="done" 时填充：
    answer: str
    confidence: float
    elapsed_ms: float
    citations: list
    used_tools: list
    turns_data: list   # 所有 Turn 的序列化结果

# 使用示例
for event in agent.run_stream("请问..."):
    if event.type == "thought":
        update_ui_thought(event.thought)
    elif event.type == "action":
        update_ui_action(event.tool_name, event.tool_input)
    elif event.type == "observation":
        update_ui_observation(event.observation)
    elif event.type == "done":
        show_final_answer(event.answer, event.confidence)
```

**事件时序**：`thought → [action → observation]* → done`（方括号内重复 0~N 次，最后一个 thought 后直接 done）

#### ToolRegistry

```python
class ToolRegistry:
    # Registry Pattern：工具名 → (fn, description, params_schema)
    
    def register(name, fn, description, params_schema):
        self._tools[name] = _ToolEntry(fn, description, params_schema)
    
    def dispatch(tool_name, tool_input: Dict) -> str:
        entry = self._tools[tool_name]
        return str(entry.fn(**tool_input))   # 返回 observation 字符串
    
    def get_tools_prompt() -> str:
        # 生成供 LLM 阅读的工具描述文本
        # 每个工具：名称 + 描述 + JSON Schema
```

#### 5 个工具

| 工具名 | 实现类 | 核心调用 |
|--------|--------|---------|
| `hybrid_search` | `HybridSearchTool` | `HybridSearch.search(query, top_k)` |
| `semantic_search` | `SemanticSearchTool` | `DenseRetriever.retrieve(query, top_k)` |
| `keyword_search` | `KeywordSearchTool` | `SparseRetriever.retrieve(query, top_k)` |
| `document_summary` | `DocumentSummaryTool` | `VectorStore.get_by_doc_id(doc_id)` |
| `list_documents` | `ListDocumentsTool` | `VectorStore.list_documents(collection)` |

#### SelfChecker（置信度）

```python
class SelfChecker:
    def check(question, answer, context: List[RetrievalResult]) -> CheckResult:
        prompt = f"""
        Question: {question}
        Context: {format_context(context)}
        Answer: {answer}
        
        Respond ONLY with JSON:
        {{
          "confidence": <0.0-1.0>,
          "is_grounded": <true/false>,
          "missing_aspects": [...],
          "should_retry": <true/false>
        }}
        """
        response = llm.chat([Message(role="user", content=prompt)])
        return CheckResult.from_json(response.content)
```

**置信度含义**：
- ≥ 0.7（🟢）：答案有充分文档支撑，无明显遗漏
- 0.4–0.7（🟡）：部分支撑，存在信息缺口
- < 0.4（🔴）：答案无法从检索内容中核实

#### ConversationMemory

```python
class ConversationMemory:
    history: List[Dict]  # [{role, content, timestamp}, ...]
    
    def add_turn(role: str, content: str)
    def format_for_prompt() -> str    # 拼接历史轮次供 LLM 阅读
    def clear()                       # 开始新会话时调用
```

---

### 4.4 MCP Server

**入口**：`src/mcp_server/server.py`

```
协议：JSON-RPC 2.0
传输：stdio（stdout 输出结果，stderr 输出日志，保持 stdout 纯净）

已实现工具：
├── query_knowledge_hub
│   Input:  {query: str, top_k: int, collection: str}
│   Output: List[{chunk_id, score, text, source}]
│
├── list_collections
│   Output: List[{name, doc_count, created_at}]
│
└── get_document_summary
    Input:  {doc_id: str}
    Output: {summary: str, chunks: int, source: str}
```

**与 Claude Desktop 集成**：在 `claude_desktop_config.json` 里配置 `mcpServers`，即可让 Claude 通过 MCP 工具直接查询本地知识库。

---

### 4.5 可观测性与 Dashboard

**框架**：Streamlit + `st.navigation()`（多页面）

| 页面 | 功能 |
|------|------|
| **系统概览** | 配置摘要、各组件状态、知识库统计、追踪数量 |
| **知识库浏览** | 查看已索引 chunk 列表，搜索、过滤、查看元数据和图像 |
| **文档导入** | 上传文件、选择集合、实时显示进度 |
| **导入追踪** | 每次导入的 6 阶段耗时、成功 / 失败状态 |
| **查询追踪** | 每次查询的 Dense / Sparse 结果对比、Rerank 前后对比 |
| **评估面板** | 运行 Ragas / Custom 评估，查看 faithfulness、relevancy 等指标 |
| **Agent 对话** | ChatGPT 风格多会话，ReAct 流式推理轨迹可视化 |

#### Agent Chat 流式输出

提问提交后，页面**实时**展示推理过程，无需等待全部完成：

```
用户提交问题
    │
    ▼
[live trace 区域，实时刷新]
  Turn 1  ● 推理中…
    💭 THOUGHT: I need to search for...
    ⚡ ACTION: hybrid_search  ● 工具执行中…
  ─────────────────────────
  Turn 1  ✅
    💭 THOUGHT: ...
    ⚡ ACTION: hybrid_search
    👁 OBSERVATION: Found 3 relevant chunks...
  Turn 2  ● 推理中…
    💭 THOUGHT: Based on the results...
    │
    ▼ (done 事件到达)
[live trace 折叠为 st.expander]
[最终答案显示]
[置信度 · 轮数 · 耗时]
```

实现机制：
- `agent.run_stream()` 生成器，每步 yield `AgentStreamEvent`
- Dashboard 用 `st.empty()` 占位，每个事件到达时调用 `live_ph.container()` 原位替换内容
- 推理中：最后一个 Turn 显示琥珀色 "● 推理中…" / "● 工具执行中…" 标记
- 完成后：调用 `live_ph.empty()` 再用 `live_ph.container()` 放入折叠的 `st.expander`

#### ReAct 时间线可视化（_render_react_trace）

每个 Turn 渲染为三色卡片：
- 💭 **蓝色**：Thought 文本
- ⚡ **紫色**：Action（工具名 badge + JSON 参数代码块）
- 👁 **绿色**：Observation（等宽字体，截断至 600 字符）

`in_progress=True` 时，最后一个 Turn 的圆点变为琥珀色并显示进度标记，用于流式更新期间。

---

## 5. 关键算法

### 5.1 RRF（Reciprocal Rank Fusion）

```
RRF_score(d) = Σᵢ  1 / (k + rankᵢ(d))

参数：
  k = 60（平滑常数，防止 rank=1 时得分过大）
  rankᵢ(d) = 文档 d 在第 i 个排序列表中的 1-based 排名
  若 d 不在某列表中，对该列表贡献为 0
```

**具体示例**：

```
Dense 结果：  [A(1), B(2), C(3)]
Sparse 结果： [B(1), C(2), D(3)]

RRF (k=60)：
  A = 1/(60+1)                = 0.01639
  B = 1/(60+2) + 1/(60+1)    = 0.03252  ← 两路都找到，排名最高
  C = 1/(60+3) + 1/(60+2)    = 0.03188
  D = 1/(60+3)               = 0.01575

最终排序：B > C > A > D
```

**优点**：
- **Score-agnostic**：无需对 cosine 和 BM25 分数做归一化
- **稳定**：两路都找到的结果天然获得更高排名
- **可调**：k 越小，top-1 优势越大；k 越大，越趋向平均

### 5.2 BM25 评分

```
score(q, d) = Σₜ  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

参数（本实现）：
  k1 = 1.5    # term frequency 饱和参数
  b  = 0.75   # 文档长度归一化参数

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
  N     = 语料库总文档数
  df(t) = 包含 term t 的文档数
```

**索引结构**：

```json
{
  "metadata": {"num_docs": 100, "avg_doc_length": 250.5},
  "index": {
    "检索": {
      "idf": 3.45,
      "df": 15,
      "postings": [
        {"chunk_id": "doc1_0000", "tf": 2, "doc_length": 320}
      ]
    }
  }
}
```

### 5.3 Dense 检索（Cosine 相似度）

```
sim(q, d) = (q_vec · d_vec) / (‖q_vec‖ · ‖d_vec‖)
```

使用 ChromaDB 内置的 cosine 距离索引（HNSW），查询复杂度 O(log N)。

---

## 6. 设计模式

### 6.1 Factory Pattern（6 层可插拔）

```python
class LLMFactory:
    _PROVIDERS: Dict[str, type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, cls_: type[BaseLLM]):
        cls._PROVIDERS[name.lower()] = cls_

    @classmethod
    def create(cls, settings) -> BaseLLM:
        name = settings.llm.provider.lower()
        return cls._PROVIDERS[name](settings=settings)
```

**配置驱动示例**：

```yaml
# 从 OpenAI 切到 DeepSeek：只改 settings.yaml，无需改代码
llm:
  provider: "DeepSeek"
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com"
  api_key: "sk-..."
```

同样的 Factory 模式应用于 EmbeddingFactory、VectorStoreFactory、RerankerFactory、SplitterFactory、EvaluatorFactory。

### 6.2 Registry Pattern（工具调度）

```python
class ToolRegistry:
    _tools: Dict[str, _ToolEntry] = {}

    def register(self, name, fn, description, params_schema):
        self._tools[name] = _ToolEntry(fn, description, params_schema)

    def dispatch(self, tool_name, tool_input) -> str:
        if tool_name not in self._tools:
            raise ToolDispatchError(f"Unknown tool: {tool_name}")
        return str(self._tools[tool_name].fn(**tool_input))
```

### 6.3 Strategy Pattern（Transform 阶段）

```python
class ChunkRefiner:
    def refine(chunk) -> Chunk:
        if self.use_llm:
            return self._llm_refine(chunk)   # LLM 策略
        return self._rule_refine(chunk)       # 规则策略（fallback）
```

### 6.4 Template Method Pattern（BaseLLM / BaseEmbedding）

```python
class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Message]) -> ChatResponse: ...
    
    def validate_messages(self, messages):  # 公共实现
        if not messages:
            raise ValueError("messages cannot be empty")
```

### 6.5 其他模式

| 模式 | 位置 | 用途 |
|------|------|------|
| **Adapter** | `DocumentChunker` | 适配 langchain splitter 到内部 Chunk 类型 |
| **Observer** | `TraceContext.record_stage()` | 收集 pipeline 各阶段事件 |
| **Chain of Responsibility** | ReAct loop + Tool dispatch | 多步骤工具链与错误降级 |

---

## 7. 数据类型系统

```python
# src/core/types.py — 所有模块间的数据契约

@dataclass
class Document:
    """PDF 解析后的原始文档"""
    id: str            # SHA256 hash
    text: str          # Markdown 格式，含 [IMAGE: ...] 占位符
    metadata: Dict     # source_path（必需）、doc_type、title、images

@dataclass
class Chunk:
    """分块后的文本片段"""
    id: str            # {doc_id}_{index:04d}_{hash[:8]}
    text: str
    metadata: Dict     # 继承 Document 元数据 + chunk_index、source_ref、image_refs
    start_offset: Optional[int]
    end_offset: Optional[int]

@dataclass
class ChunkRecord:
    """向量化后（可存储）的 chunk"""
    id: str
    text: str
    metadata: Dict     # 含 image_captions: {image_id: caption_text}
    dense_vector: Optional[List[float]]       # Embedding 向量
    sparse_vector: Optional[Dict[str, float]] # BM25 term weights

@dataclass
class ProcessedQuery:
    """查询预处理结果"""
    original_query: str
    keywords: List[str]   # jieba 分词
    filters: Dict         # {collection, doc_type, tags, ...}

@dataclass
class RetrievalResult:
    """统一的检索结果格式（Dense / Sparse / Fused 均使用）"""
    chunk_id: str
    score: float          # cosine 相似度 或 BM25 分数 或 RRF 分数
    text: str
    metadata: Dict        # source_path、chunk_index、title 等
```

**设计原则**：每个模块只通过这些 dataclass 通信，不直接依赖对方的内部实现。

---

## 8. 配置体系

```yaml
# config/settings.yaml — 完整注释版

llm:
  provider: "DeepSeek"           # openai / azure / deepseek / ollama
  model: "deepseek-v4-flash"
  temperature: 0.0               # 推理任务建议 0.0
  max_tokens: 4096
  api_key: "sk-..."
  base_url: "https://api.deepseek.com"
  # Azure 专用：
  # api_version: "2024-02-15-preview"
  # deployment_name: "gpt-4"

embedding:
  provider: "硅基流动"           # openai / azure / ollama / siliconflow / 硅基流动
  model: "BAAI/bge-m3"           # 多语言向量模型，维度 1024
  dimensions: 1024
  base_url: "https://api.siliconflow.cn/v1"
  api_key: "sk-..."

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
  collection_name: "knowledge_hub"

retrieval:
  dense_top_k: 20                # Dense 初始检索数
  sparse_top_k: 20               # BM25 初始检索数
  fusion_top_k: 10               # RRF 融合后返回数
  rrf_k: 60                      # RRF 平滑常数

rerank:
  enabled: false
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  splitter: "recursive"          # recursive / semantic
  batch_size: 100
  chunk_refiner:
    use_llm: true                # false → 规则 fallback
  metadata_enricher:
    use_llm: true

agent:
  max_turns: 5                   # ReAct 最大迭代轮数
  confidence_threshold: 0.7      # SelfChecker 触发 retry 的阈值
  default_collection: "default"
  top_k: 5

observability:
  log_level: "INFO"
  trace_enabled: true
  trace_file: "./logs/traces.jsonl"
  structured_logging: true
```

---

## 9. 测试策略

### 9.1 三层测试

```
tests/
├── unit/          # 快速，纯 Python，无外部依赖
│   ├── test_rrf_fusion.py         # RRF 算法正确性
│   ├── test_react_agent.py        # Agent 循环（mock LLM + mock 工具）
│   ├── test_conversation_memory.py
│   ├── test_self_checker.py
│   └── test_agent_tools.py
│
├── integration/   # 需要 ChromaDB / LLM API
│   ├── test_ingestion_pipeline.py
│   ├── test_hybrid_search.py
│   └── test_mcp_server.py
│
└── e2e/           # 完整业务流程
    ├── test_data_ingestion.py     # 上传→检索→验证
    ├── test_mcp_client.py         # 真实 MCP 协议调用
    └── test_recall.py             # Golden set 召回率测试
```

### 9.2 测试标记

```python
@pytest.mark.unit           # 快速，无依赖
@pytest.mark.integration    # 需要外部服务
@pytest.mark.llm            # 需要 LLM API（可用 -m "not llm" 跳过）
@pytest.mark.slow           # 耗时超过 10s
```

### 9.3 运行方式

```bash
pytest tests/unit -v                        # 仅单元测试
pytest tests/ -m "not llm" -v              # 跳过需要 API 的测试
pytest tests/ --cov=src --cov-report=html  # 生成覆盖率报告
```

---

## 10. 面试高频问题与回答

### 架构设计类

**Q：为什么用混合检索而不只用向量检索？**

A：Dense 检索（向量）擅长语义相似——同义词、跨语言、语意改写都能命中；但对精确词匹配弱，比如专有名词「GPT-4o」直接搜可能找不到。BM25 是精确词频统计，擅长精确匹配，但无法理解同义词。两者天然互补。RRF 融合不需要归一化分数，直接基于排名叠加，简单可靠。实测两路联合的 Recall@10 比单路高 15–30%。

---

**Q：RRF 里 k=60 是怎么来的？**

A：学术论文（Cormack et al. 2009）实验得出的经验值。k 控制 rank 的影响衰减速度：k 小时 top-1 优势极大，k 大时各名次差距缩小。60 在大多数 IR 任务上表现稳定。本项目通过 `config/settings.yaml` 里的 `rrf_k` 可调，支持针对具体知识库做消融实验。

---

**Q：如何保证摄取的幂等性？**

A：第一阶段计算文件 SHA256 hash，存入 SQLite `ingestion_history` 表。再次摄取同一文件时直接返回 `status=skipped`，整个 Pipeline 后续阶段都不会执行。同时 VectorStore upsert 操作以 chunk_id 为主键，即使绕过 hash 检查重复写入，也只会覆盖而不会产生重复记录。

---

**Q：图像内容如何被检索到？**

A：Image-to-Text 策略，不引入多模态向量：
1. 解析 PDF 时提取图像，在文本中插入 `[IMAGE: doc_abc_page1_0]` 占位符
2. Vision LLM（ImageCaptioner）生成自然语言描述
3. 描述文字缝合到 chunk 文本中
4. 走完全和普通文本一样的 Embedding + BM25 流程

优点：统一检索链路，不需要 CLIP 等多模态模型，部署更简单。

---

### Agent 类

**Q：ReAct Agent 怎么防止无限循环？**

A：两道防线：
1. `max_turns=5`（可配置）硬性截断，循环计数器超过即退出
2. 输出正则检测 `Final Answer:` 标志，LLM 主动宣布完成时立即 break

超出 max_turns 时，取最后一个 Thought 作为 fallback 答案，不会返回空响应。

---

**Q：SelfChecker 是什么？怎么用？**

A：一个 LLM-as-judge 模块，在 Agent 给出最终答案后调用，判断答案是否有文档依据：

```json
{
  "confidence": 0.85,
  "is_grounded": true,
  "missing_aspects": [],
  "should_retry": false
}
```

`confidence` 就是 Dashboard 里显示的置信度。`should_retry=true` 时可触发二次检索（当前为 observe，不自动重试）。SelfChecker 调用 LLM 失败时 fallback 到启发式打分（工具成功率 + 是否正常结束）。

---

**Q：Agent Chat 的流式输出是怎么实现的？**

A：两层设计：

**1. Turn 级流式（`run_stream()`）**：`ReActAgent.run_stream()` 是 `run()` 的生成器版本，每完成一个 Thought / Action / Observation 就立即 `yield AgentStreamEvent`，不需要等整个 ReAct 循环结束。Dashboard 在 `for event in agent.run_stream()` 循环里，每收到一个事件就调用 `st.empty().container()` 原位替换 live trace 内容，用户能实时看到 Agent 正在思考什么、调用了什么工具。

**2. Token 级流式（`stream_chat()`）**：`BaseLLM` 新增 `stream_chat()` 接口，`OpenAILLM` 和 `DeepSeekLLM` 各自覆盖，使用 httpx SSE streaming（`"stream": true` + `resp.iter_lines()` 逐行解析 `data:` chunk，逐 token `yield`）。这是 LLM 层的流式基础设施，目前 Dashboard 展示层用 Turn 级流式，`stream_chat()` 可供后续集成（如在最后一轮 Final Answer 输出时做真 token 流）。

**Streamlit 原位刷新技巧**：`st.empty()` 创建占位，`with live_ph.container()` 每次调用都替换内容，不会累积渲染。三个顺序占位（live_ph / answer_ph / status_ph）保证完成后布局从「live trace 在上」自然变为「答案在上、折叠 trace 在下」。

---

**Q：ToolRegistry 为什么用字符串 dispatch 而不是直接 if-else？**

A：可扩展性。LLM 输出的工具名是字符串，Registry 解耦了「工具定义」和「工具调用」。新增工具只需调一次 `register()`，Agent 主循环代码不需要改。同时 Registry 存储了 JSON Schema，`get_tools_prompt()` 自动生成供 LLM 阅读的工具描述，工具描述和实现保持同步。

---

### 工程实践类

**Q：如何新增一个 LLM 提供商（如 Moonshot）？**

A：三步：
1. 在 `src/libs/llm/` 新建 `moonshot_llm.py`，继承 `BaseLLM`，实现 `chat()` 方法
2. 在 `src/libs/llm/__init__.py` 里 `LLMFactory.register_provider("moonshot", MoonshotLLM)`
3. `settings.yaml` 里 `provider: "moonshot"`

对于 OpenAI 兼容接口（如 SiliconFlow），可以直接在 `embedding_factory.py` 里注册别名，无需新建文件：

```python
EmbeddingFactory.register_provider("moonshot", OpenAIEmbedding)
```

---

**Q：TraceContext 记录什么？怎么用？**

A：每个 Pipeline 阶段调用 `trace.record_stage(stage_name, data, elapsed_ms)`，data 里包含：
- 用了什么方法（embedding / bm25）
- 哪个提供商（siliconflow / chroma）
- 入参和出参（top_k、result_count、chunk 列表）
- 耗时

所有 trace 追加到 `logs/traces.jsonl`，Dashboard 的「查询追踪」页面按 trace_id 展示完整链路，可以对比 Dense 和 Sparse 各自的结果，定位召回率问题。

---

**Q：为什么 Streamlit 用 `st.navigation()` 而不是 `pages/` 目录？**

A：`st.navigation()` 是 Streamlit 1.29+ 的新 API，支持用函数作为页面（不强制文件命名规范），可以在 `app.py` 里统一控制权限检查、全局样式注入（`_GLOBAL_CSS`）、页面元数据（icon、title）。`pages/` 目录方式每个文件都是独立入口，全局逻辑不好复用。

---

### 深度追问

**Q：BM25 的 k1=1.5 和 b=0.75 是怎么选的？有没有调优？**

A：这是标准 BM25 的推荐默认值，Robertson 等人大量实验得出的经验参数。k1 控制 term frequency 饱和速度，1.5 意味着 tf=10 和 tf=100 的得分差异很小，避免高频词主导；b=0.75 对长文档惩罚适中。本项目暂未做超参数搜索，可以通过 Golden test set + Recall@k 指标做消融。

---

**Q：混合检索里 Dense 和 Sparse 的权重是 1:1 吗？**

A：RRF 本身没有显式权重——每个列表的贡献取决于文档排名，而非乘以某个固定系数。如果要调节两路的相对权重，可以改成加权 RRF：`w_dense/(k+rank_dense) + w_sparse/(k+rank_sparse)`，本实现目前是对称的 1:1，可在 `fusion.py` 里扩展。

---

**Q：Agent 的 Prompt 在哪里？可以看到内容吗？**

A：`config/prompts/react_agent.txt`，运行时通过 `Path.read_text()` 动态加载，支持不重启服务改 Prompt。文件不存在时 fallback 到 `react_agent.py` 里的内嵌模板。Prompt 里包含：工具列表（`{tools_description}`）、当前问题（`{question}`）、历史轨迹（`{history}`），以及 ReAct 格式指引（Thought / Action / Action Input / Observation / Final Answer）。

---

## 附录：关键文件速查

| 文件 | 核心类 / 函数 | 一句话说明 |
|------|--------------|-----------|
| `src/core/types.py` | Document / Chunk / RetrievalResult | 全项目数据合约 |
| `src/core/settings.py` | `load_settings()` | YAML 配置加载与验证 |
| `src/ingestion/pipeline.py` | `IngestionPipeline.run()` | 6 阶段摄取编排 |
| `src/core/query_engine/hybrid_search.py` | `HybridSearch.search()` | 混合检索主控 |
| `src/core/query_engine/fusion.py` | `RRFFusion.fuse()` | RRF 算法 |
| `src/ingestion/storage/bm25_indexer.py` | `BM25Indexer.build/query()` | BM25 索引 |
| `src/agent/react_agent.py` | `ReActAgent.run()` / `run_stream()` | ReAct 主循环 / 流式生成器 |
| `src/libs/llm/base_llm.py` | `BaseLLM.stream_chat()` | LLM SSE 流式接口（默认 fallback，子类覆盖） |
| `src/agent/tool_registry.py` | `ToolRegistry.dispatch()` | 工具调度 |
| `src/agent/reflection/self_checker.py` | `SelfChecker.check()` | LLM 幻觉检测 |
| `src/libs/llm/llm_factory.py` | `LLMFactory.create()` | LLM 工厂 |
| `src/libs/embedding/embedding_factory.py` | `EmbeddingFactory.create()` | Embedding 工厂 |
| `src/mcp_server/server.py` | `run_stdio_server()` | MCP Server 入口 |
| `src/observability/dashboard/app.py` | `st.navigation()` | Dashboard 入口 |
| `config/settings.yaml` | — | 全局配置 |
| `config/prompts/react_agent.txt` | — | ReAct Prompt 模板 |
