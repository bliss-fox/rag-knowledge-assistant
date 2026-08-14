# 面试官导读：如何讲清这个生产级 RAG 项目

这份文档用于快速评估项目的工程深度。所有数字只引用仓库中可复现的测试或验收结果；尚未完成的质量工作明确列在最后。

## 30 秒定位

我把一个模块化 RAG 原型升级成了可在单台 Windows 服务器交付的局域网产品。系统支持多来源增量摄取、BM25 + Dense + RRF + Cross-Encoder 检索、生成前后证据门控、多用户权限、持久任务、全链路 Trace、离线评测、CI 质量门禁和 Windows 启动器。设计重点是“可拒答、可解释、可恢复、可评测”，而不只是生成一段看似合理的文本。

## 5 分钟讲解顺序

### 1. 为什么要做

演示型 RAG 常见的风险是：只做向量检索、证据不足仍生成、Prompt 散落在代码里、摄取任务中断后状态不明、线上问题只有平均耗时和最终答案。这个项目把这些风险转化为明确的系统约束和可观测数据。

### 2. 请求如何流动

1. Streamlit、MCP 和 CLI 都进入 FastAPI 背后的统一应用服务层。
2. 查询规范化后并行执行 BM25 与 Qdrant Dense Search。
3. RRF 融合两路排名，保留原始名次和分数组成。
4. Cross-Encoder 对候选精排；超时或失败时显式降级到 RRF。
5. EvidenceGate 在调用 LLM 前检查最高分、有效证据数和校准状态。
6. Ollama 生成后，系统再检查 claim 的引用覆盖和引用有效性。
7. 每一步的输入、输出、耗时、Token、版本和异常写入 SQLite Trace。

### 3. 数据如何进入系统

Loader Registry 支持 PDF、Markdown、TXT 和受限网页抓取。本地知识库只读扫描，使用文件 SHA-256、mtime、稳定 document ID 和 source manifest 计算新增、修改、删除。摄取、同步、索引重建和评测进入 SQLite 持久队列，由 Worker 恢复、重试、取消并对同一来源加互斥锁。

### 4. 如何保证答案不乱编

项目不把“请根据上下文回答”当作安全边界。生成前的 EvidenceGate 可以直接拒答且不调用 LLM；生成后的 claim/citation 检查会拒绝无引用事实、无效引用或证据不能支撑的明确断言。阈值不能凭感觉设定，必须由公开与私有 dev split 校准，配置指纹改变后重新校准。

### 5. 如何证明改动没有退化

离线评测固定 corpus、chunk、query 和 k，对 BM25、Dense、Hybrid、Hybrid + Rerank 做公平消融，输出 Recall、MRR、nDCG 和 P50/P95。答案侧记录 Faithfulness、Answer Relevancy、引用覆盖与拒答指标。CI 在 Judge 缺失、数据集未人工复核或 baseline 未冻结时主动失败，不会把 mock 结果包装成质量结论。

## 值得深入讨论的设计

### 为什么用 RRF，而不是把 BM25 与向量分数直接相加？

BM25 与 cosine similarity 的量纲、范围和分布不同。RRF 只使用名次，使融合在没有成熟标注集时更稳定，也能清楚解释每个候选来自哪一路。代价是无法自动学习业务权重，因此仍需后续黄金集验证。

### 为什么 EvidenceGate 不能只写进 Prompt？

Prompt 是软约束，生成模型可能忽略它；而“没有充分证据时不得回答”是产品规则。把门控做成独立确定性组件，才能在生成前节省模型调用、输出结构化拒答原因，并通过单元测试覆盖空结果、低分、冲突和无引用 claim。

### 为什么选择 SQLite，而不是 PostgreSQL 或消息队列？

目标是单台 Windows 服务器。SQLite WAL 降低部署复杂度，同时能事务化保存账号、任务、Trace、审计和 deployment event。代码和文档明确不把它包装成集群方案；若需求扩展到多节点 Worker，任务租约、数据库和指标存储都应迁移。

### 如何处理重排器故障？

Cross-Encoder 是增强项，不应成为单点故障。启动时预热并由 readiness 报告状态；请求超时或推理失败时返回 RRF 顺序，同时在响应与 Trace 中标记 `degraded` 和原因，便于统计降级率。

### 如何回答“上周二质量为什么下降”？

Dashboard 以时间窗比较请求量、失败/拒答/降级率、引用覆盖、Token 和各阶段 P50/P95，并按 Prompt、模型、配置、索引和错误类型分组。deployment event 把这些版本切换关联到具体时间点，单次 Trace 则展示召回、融合、重排、最终 Prompt 和响应瀑布。

## 可复现证据

```powershell
# 不访问云模型的完整回归
.\.venv\Scripts\python.exe -m pytest -m "not llm"

# Prompt、黄金集和 baseline 状态校验
.\.venv\Scripts\python.exe scripts\validate_production_assets.py

# 静态与编译检查
.\.venv\Scripts\python.exe -m ruff check src scripts --select F
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

更具体的真实服务、Qdrant roundtrip、本地目录同步和 Windows 进程回收记录见 [生产验收记录](PRODUCTION_ACCEPTANCE_2026-08-13.md)。部署端口、备份、权限与安全边界见 [生产部署说明](PRODUCTION_DEPLOYMENT.md)。

## 当前不能夸大的内容

- 正式公开黄金集尚未完成约 100 条人工复核，目前只有 6 条种子样例。
- `evaluation/baseline.pending.json` 仍使质量门禁 fail-closed；因此不能宣称 Faithfulness、Recall 或引用覆盖已经达到生产阈值。
- 21 条 legacy candidate 的检索数字只证明评测链路可复现，不是正式质量 baseline；在 CPU 上 Cross-Encoder 的延迟代价明显，当前数据也没有证明净收益。
- EvidenceGate 仍等待公开和私有 dev split 的正式校准。
- SQLite WAL 是单节点交付边界，不等同于高可用集群。
- PyInstaller 启动器不包含 Ollama、Qdrant 或模型权重，目标机器仍需准备这些依赖。

## 如果继续迭代

优先级不是再增加一个 Agent 工具，而是完成 50–200 条人工复核黄金集、在目标硬件上校准 EvidenceGate、冻结可复现 baseline，再根据失败分类决定是否调整 chunk、检索参数或 Reranker。只有这些数据稳定后，才启用阻止合并的正式质量阈值。
