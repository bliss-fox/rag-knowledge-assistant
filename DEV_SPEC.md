# 系统设计文档 (Design Specification)

> Modular RAG MCP Server — 架构决策、技术选型与关键权衡

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **可插拔** | 每层核心组件可通过配置切换，不修改业务代码 |
| **可观测** | 每次查询的全链路追踪，记录各阶段耗时与中间结果 |
| **可评估** | 内置 Ragas + 自定义指标，支持 A/B 对比不同策略 |
| **可扩展** | Agent 工具、检索策略、LLM Provider 均可独立横向扩展 |

---

## 2. 架构分层

```
配置层  config/settings.yaml
   │
抽象层  libs/  (BaseLLM, BaseEmbedding, BaseReranker, BaseSplitter)
   │
实现层  libs/llm/openai_llm.py, libs/embedding/*, ...
   │
核心层  ingestion/, retrieval/, agent/
   │
协议层  mcp_server/  (JSON-RPC 2.0)
   │
展示层  observability/dashboard/  (Streamlit)
```

依赖方向严格向下，核心层不依赖协议层或展示层。

---

## 3. 关键技术决策

### 3.1 混合检索 — 为什么选 RRF 而非加权求和

**问题**：Dense（向量余弦相似度，范围约 0~1）与 BM25（IDF 加权词频，无固定范围）的分数量纲不同，直接线性加权需要人工校准权重。

**决策**：使用 Reciprocal Rank Fusion（RRF，k=60），公式：

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

**理由**：
- 基于排名而非原始分数，天然免疫两路检索的量纲差异
- k=60 在多项研究中被验证为鲁棒超参数，无需 per-dataset 调整
- 实现简单（10 行代码），效果稳定

**代价**：丢失了分数的绝对大小信息（如置信度差异）。

---

### 3.2 两阶段检索 — 粗排 + 精排

```
Query
  │
  ├─ Dense Recall (Top-20)  ─┐
  │                          ├─ RRF Merge → Top-20 candidates
  └─ BM25 Recall (Top-20)   ─┘
                              │
                    Cross-Encoder Rerank → Top-5 final
```

**理由**：
- Cross-Encoder 逐对打分精度高，但时间复杂度 O(n)，不能对全库运行
- 先用低成本双路召回缩小候选集，再用高成本精排，是工业界标准范式
- 实测：Hybrid 召回 Hit@5 = 100%，精排只需在已高质量候选上工作

---

### 3.3 ReAct Agent — 为什么不用单次 RAG

**单次 RAG 的局限**：
- 多跳问题（"A 和 B 相比，哪个在 C 场景更优？"）需要先检索 A，再检索 B，再综合
- 无法感知检索结果的充分性，"不知道自己不知道"

**ReAct 方案**：
```
Thought → Action(tool) → Observation → Thought → ... → Answer
```

- 每轮 Thought 决定是否继续检索或已有足够信息
- SelfChecker 在生成答案后做二次验证（LLM-as-judge），检测幻觉
- ConversationMemory 保留最近 N 轮对话，支持追问

**代价**：延迟增加（每个 Thought-Action 轮需一次 LLM 调用），最大轮数限制为 8 防止无限循环。

---

### 3.4 MCP 协议 — 为什么选 stdio transport

**选型**：JSON-RPC 2.0 over stdio（而非 HTTP/SSE）

**理由**：
- stdio transport 是 Claude Desktop 的标准集成方式，零网络配置
- 进程间通信，天然隔离，无需管理端口和认证
- 符合 MCP 规范，工具定义（inputSchema）与 Claude 直接对接

**代价**：不支持多客户端并发（单进程单连接），但对个人知识库场景足够。

---

### 3.5 可观测性 — TraceContext 设计

每次查询创建一个 `TraceContext` 对象，在管道各阶段注入：

```python
trace = TraceContext(query_id=uuid)
trace.record("retrieval", latency_ms=142, results_count=20)
trace.record("rerank", latency_ms=38, results_count=5)
trace.record("generation", latency_ms=810, tokens=320)
```

Trace 落盘为 JSONL（`traces.jsonl`），Dashboard 实时读取并可视化。

**设计原则**：Trace 记录与业务逻辑解耦，通过参数传递而非全局变量，便于单元测试。

---

## 4. 可插拔架构实现

### 4.1 抽象接口

```python
# 以 LLM 为例
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
```

所有实现（`OpenAILLM`、`DeepSeekLLM`、`OllamaLLM`）继承 `BaseLLM`，上层代码只依赖接口。

### 4.2 工厂注册

```python
LLM_REGISTRY = {
    "openai":   OpenAILLM,
    "deepseek": DeepSeekLLM,
    "ollama":   OllamaLLM,
}

def create_llm(settings: LLMSettings) -> BaseLLM:
    return LLM_REGISTRY[settings.provider](settings)
```

新增 Provider 只需：① 实现 `BaseLLM`，② 在注册表加一行。

### 4.3 配置驱动

```yaml
# config/settings.yaml — 切换 Provider 无需改代码
llm:
  provider: deepseek          # openai | azure | deepseek | ollama
  model: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}

embedding:
  provider: siliconflow       # openai | azure | siliconflow | ollama
  model: BAAI/bge-m3
```

---

## 5. 测试策略

| 层级 | 范围 | 隔离方式 |
|------|------|---------|
| **Unit** | 单模块、纯逻辑 | Mock 外部依赖（LLM、向量库） |
| **Integration** | 跨模块流水线 | 真实 ChromaDB（内存模式）+ Mock LLM |
| **E2E** | 完整查询链路 | 真实 API（标记 `@pytest.mark.llm`，默认跳过） |

原则：单元测试覆盖所有边界条件；集成测试验证组件接口契约；E2E 测试仅在 CI 指定 Job 中运行。

---

## 6. 扩展点

| 扩展方向 | 实现位置 | 难度 |
|---------|---------|------|
| 新 LLM Provider | `src/libs/llm/` + 注册表 | 低 |
| 新 Embedding Provider | `src/libs/embedding/` + 注册表 | 低 |
| 新 VectorStore（Qdrant/Milvus） | `src/libs/vectorstore/` | 中 |
| 新 Agent 工具 | `src/agent/tools/` + `tool_registry.py` | 低 |
| 新 Evaluator | `src/libs/evaluator/` | 中 |
| HTTP/SSE MCP transport | `src/mcp_server/` | 中 |
| 多租户 / 权限隔离 | `src/core/` + Collection 命名空间 | 高 |
