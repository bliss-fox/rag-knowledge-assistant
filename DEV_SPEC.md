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

### 3.3 ReAct Agent — 研究与多步规划边界

**单次 RAG 的局限**：
- 多跳问题（"A 和 B 相比，哪个在 C 场景更优？"）需要先检索 A，再检索 B，再综合
- 单次检索不适合需要多轮改写与工具选择的探索性任务

**ReAct 方案**：
```
Thought → Action(tool) → Observation → Thought → ... → Answer
```

- 每轮 Thought 决定是否继续检索或改写查询
- SelfChecker 只提供研究性二次检查和可观察信号，不能替代确定性的 EvidenceGate，也不能把未经证据支持的答案自动变成正确答案
- ConversationMemory 保留最近 N 轮对话，支持追问

**代价**：延迟增加（每个 Thought-Action 轮需一次 LLM 调用），最大轮数限制为 8 防止无限循环。

**生产边界**：ReAct/SelfChecker 保留为研究与规划组件，不是生产最终回答入口。`scripts/agent.py` 是认证 FastAPI 客户端，不会在进程内直接构建 ReAct、LLM 或检索器；所有生产最终答案必须经过 `/answer` 与统一的 `RAGApplicationService.answer()`，在生成前通过 EvidenceGate，并在生成后执行 claim/citation 覆盖检查。MCP 的 `query_knowledge_hub` 只返回复用应用服务层得到的检索证据，不绕过该边界生成最终答案。

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
trace = TraceContext(trace_type="query")
trace.record_stage("retrieval", {"results_count": 20}, elapsed_ms=142)
trace.record_stage("rerank", {"results_count": 5}, elapsed_ms=38)
trace.record_stage("generation", {"usage": {"completion_tokens": 320}}, elapsed_ms=810)
trace.finish()
```

生产主存储是 SQLite WAL 的 `traces` 与 `trace_stages` 表，Dashboard 只通过认证
FastAPI 读取；`logs/traces.jsonl` 仅作为旧版本兼容输入和可选导出格式。旧记录使用
`scripts/import_legacy_traces.py` 逐行迁移：默认原子跳过已有 trace ID，兼容 stage 顶层
字段与新版 `stage.data`，坏行不会中断其余记录，迁移报告不回显原始正文。

**设计原则**：Trace 记录与业务逻辑解耦，通过参数传递而非全局变量；正文写入前统一
脱敏，30 天后清理正文而保留哈希、版本、排名、耗时、Token、状态和聚合指标。

---

### 3.6 黄金集治理 — 候选、审批与冻结发布分离

公开候选与私有真实业务集使用相同的评测 case schema，但生命周期不同：

- 公开候选由固定上游 revision 确定性生成，自动产物始终保持
  `eligible_for_quality_gate=false`；只有逐条核对问题、答案要点和支持文档后，才能通过
  独立提升变更进入正式公开集。
- 当前候选包由固定版本 CMRC 2018、XQuAD `xquad.zh`、SQuAD 2.0、HotpotQA、
  MultiHop-RAG 和 RAGBench FinQA 组成。所有来源固定到完整上游 commit SHA；生成器先校验
  数据卡许可证，再校验下载文件的预期字节数与 SHA-256，任何上游内容漂移都会失败。
  MultiHop-RAG 保留 ODC-By 1.0 归属信息，直接使用上游 `question_type` 区分推断、比较、
  时间和无答案问题；所有证据 URL 必须能映射到同一 revision 的完整 corpus。FinQA 使用
  CC BY 4.0，表格类别来自官方子集配置，句级相关键映射为预期文档；模型输出只能作为待
  人工核对的答案候选，不能标记为人工答案。生成后的 corpus 会清理不再被 manifest 引用
  的旧 Markdown，确保磁盘文件与证据清单一致。
- 私有草稿和逐条审批保存在本机 SQLite schema v5 的 `private_golden_sets` 与
  `private_golden_cases`，普通用户无权访问，审计日志只保存 set/case ID、状态和版本，
  不保存私有问题、答案要点或证据 ID。批准必须同时记录问题、答案、证据、类别/难度与
  split 五项人工核对证明；旧版本中缺少这些证明的批准会在迁移后回到 pending。
- 私有导出要求 50–200 条、全部批准、dev/final 均非空，且 dev 同时覆盖可回答与
  不可回答问题。任何样例修改会撤销旧审批；发布内容变化时必须提升语义版本。
- 导出先在临时文件上执行 schema 和质量门禁资格校验，再原子替换
  `evaluation/private_golden.json`。该路径被 Git 忽略，私有评测只允许本地 Ollama；
  final split 不参与日常调参。

这套工作流只证明系统具备安全整理和发布私有黄金集的能力；在 50–200 条真实样例尚未
由管理员完成复核前，不得声称私有黄金集或生产质量门禁已经完成。

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
