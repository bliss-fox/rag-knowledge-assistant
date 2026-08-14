# Modular RAG MCP Server

面向单台 Windows 服务器、局域网多人使用的 RAG 系统：管理员导入 PDF、Markdown、TXT、网页或只读同步本地知识库，用户通过浏览器获得带稳定引用的回答；证据不足时系统在生成前后双重拒答。

这是单节点交付方案，不是高可用集群。默认日常问答只调用服务器本机的 Ollama、Qdrant 和 SQLite；只有公开语料的 PR 质量评测可调用 DeepSeek Judge。

> 面试官快速入口：[5 分钟项目讲解与高频追问](docs/INTERVIEW_GUIDE.md) · [生产验收记录](docs/PRODUCTION_ACCEPTANCE_2026-08-13.md) · [部署与安全边界](docs/PRODUCTION_DEPLOYMENT.md)

## 我解决的工程问题

这个项目的重点不是“把文档塞进向量库后调用一次大模型”，而是补齐演示型 RAG 到可交付单机产品之间的工程缺口：

- **答案可信度**：检索前置 EvidenceGate 与生成后 claim/citation 校验共同约束回答；证据不足时不调用 LLM 或返回拒答。
- **检索可解释性**：完整保留 BM25、Dense、RRF 与 Cross-Encoder 的分数、排名变化和降级原因，而不是只返回最终文本。
- **数据可维护性**：目录和网页使用稳定 ID、manifest、增量同步与精确删除；任务崩溃可恢复，同一来源互斥执行。
- **行为可复现性**：Prompt、配置、模型和索引全部带版本；评测冻结 corpus、参数和运行环境，CI 对缺失 Judge、未冻结 baseline 等情况 fail-closed。
- **问题可定位性**：SQLite Trace 记录检索、重排、Prompt、响应、Token、P50/P95、引用覆盖和 deployment event，可把异常窗口关联到具体变更。

## 可验证的工程证据

| 证据 | 当前结果 | 如何复核 |
|---|---|---|
| 自动化回归 | 1,501 passed、4 skipped，覆盖 Unit / Integration / E2E（2026-08-14） | `pytest -m "not llm"` |
| 真实检索链路 | 隔离 Qdrant collection 完成 Ollama embedding、Dense/BM25/RRF 单文档 roundtrip | [验收记录](docs/PRODUCTION_ACCEPTANCE_2026-08-13.md#qdrant--ollama-roundtrip) |
| 本地知识库同步 | 2 个允许文件成功同步；二次运行 `processed=0`、`unchanged=2` | [验收记录](docs/PRODUCTION_ACCEPTANCE_2026-08-13.md#本地知识库只读同步) |
| Windows 交付 | PyInstaller 启动 API、Worker、Dashboard；父进程退出后回收整棵子进程树 | [验收记录](docs/PRODUCTION_ACCEPTANCE_2026-08-13.md#windows-启动器) |
| Prompt 治理 | 10 个版本化 YAML Prompt，变量和 SHA-256 启动校验 | [`config/prompts`](config/prompts) |
| 质量诚信 | 公开黄金集仍为 6 条种子样例，正式 baseline 保持 pending，CI 不会伪造通过 | [测试与评测](#测试与评测) |

自动化测试数量证明的是工程回归面，不代表检索或回答质量提升；正式质量结论必须等人工复核黄金集和冻结 baseline 后再给出。

## 已实现能力

- FastAPI 统一后端；Streamlit 只通过 API 操作，MCP 与 CLI 共用 `RAGApplicationService`。
- Argon2id 本地账号、`admin`/`user` RBAC、短期访问令牌、可撤销刷新会话和审计事件。
- PDF / Markdown / TXT Loader Registry；受限同域网页抓取含 SSRF、重定向、深度、页数和响应大小防护。
- `D:\AI-KnowledgeBase\documents` 只读增量同步；SHA-256、mtime、稳定文件 ID 和精确 document→chunk 删除清单。
- 独立版本化 Qdrant collection，默认 `modular_rag_<logical>_v1`；禁止使用或清理 `local_knowledge`。
- BM25 + Dense + RRF + `BAAI/bge-reranker-v2-m3`，重排超时/失败显式降级并记录 Trace。
- 独立 EvidenceGate、生成后 claim 引用覆盖检查，以及 `answered/refused/degraded/error` 结构化状态。
- 十个版本化 YAML Prompt，启动校验变量和 SHA-256；生产路径不使用未版本化内联 Prompt。
- SQLite WAL 持久任务、Trace、阶段耗时、Token、审计、deployment event 和 30 天正文清理。
- 公开/私有黄金集 schema、dev/final 隔离、检索消融、答案质量指标与 DeepSeek CI 门禁框架。

## 架构

```text
LAN browser ──> Streamlit ──> FastAPI ──> RAGApplicationService
MCP stdio ───────────────────────┘            │
CLI ──────────────────────────────────────────┘
                                               ├─ BM25
                                               ├─ Ollama embedding ──> Qdrant
                                               ├─ Cross-Encoder rerank
                                               ├─ EvidenceGate ──> Ollama qwen3:8b
                                               └─ SQLite WAL Trace / jobs / auth

Admin API ──> persistent job queue ──> Worker ──> loaders / sync / evaluation
Windows EXE ──> API + Worker + Streamlit + browser
```

## 关键技术取舍

| 决策 | 原因 | 代价 / 边界 |
|---|---|---|
| BM25 + Dense 通过 RRF 融合 | 两路分数量纲不同，按排名融合避免脆弱的手工归一化 | RRF 不学习业务权重，仍需黄金集验证 |
| 独立 Cross-Encoder 精排 | 对候选逐对建模，并允许超时后显式退回 RRF | CPU 延迟高；当前候选评测没有证明净收益 |
| 规则化 EvidenceGate 独立于 LLM | “是否有足够证据”不能只让生成模型自我裁决 | 阈值必须基于公开和私有 dev split 校准 |
| SQLite WAL + 单 Worker | Windows 单机部署简单、任务与 Trace 事务一致 | 不支持多主机高可用或横向扩容 |
| Qdrant collection 版本 + alias 切换 | 重建索引时不原地污染旧版本，便于验证和回滚 | 需要快照、容量与旧版本清理策略 |
| Streamlit 只调用 FastAPI | UI、MCP、CLI 共享应用服务层和权限/拒答逻辑 | 多一个服务边界，需要合约与健康检查 |

## 三分钟启动

前置服务：Ollama `http://127.0.0.1:11434`、Qdrant `http://127.0.0.1:6333`。本机应已有：

```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

安装并启动：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m src.production.launcher
```

打开 `http://<服务器IP>:8501`，首次启动创建管理员（无默认密码）。API 是 `http://<服务器IP>:8766`，健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8766/health/ready
```

服务也可分开运行：

```powershell
.\.venv\Scripts\python.exe -m src.production.api
.\.venv\Scripts\python.exe -m src.production.worker
.\.venv\Scripts\python.exe -m streamlit run src/observability/dashboard/app.py
```

## API 与权限

主要接口包括 `/auth/*`、`/users`、`/data-sources`、`/documents/upload`、`/jobs/*`、`/search`、`/answer`、`/traces`、`/metrics/*` 和 `/evaluations`。管理员可在 Streamlit 的“离线评测”页面启动 dev/final 任务，查看 BM25/Dense/Hybrid/Rerank 消融、Faithfulness、Answer Relevancy、引用覆盖率、正确/错误拒答率、P50/P95 和逐查询失败分析；普通用户只可查询并查看自己的任务和 Trace。

默认绑定 `0.0.0.0` 供局域网访问，但不会配置公网入口、TLS、企业 SSO 或多节点高可用。生产 LAN 仍应由防火墙限制来源，公网访问应在受管反向代理后单独完成威胁建模。

## 测试与评测

```powershell
# 全量离线自动化测试（不调用真实云模型）
.\.venv\Scripts\python.exe -m pytest -m "not llm"

# Prompt 与黄金集 schema
.\.venv\Scripts\python.exe scripts/validate_production_assets.py

# 编译与新增代码静态检查
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m ruff check src scripts --select F
```

公开集位于 `evaluation/public_golden.json`；私有集复制 `evaluation/private_golden.example.json` 为 `evaluation/private_golden.json`，后者已被 Git 忽略。每条样例包含 query ID、类别、预期文档、答案要点、是否可回答、来源、dev/final、语言和难度。

后台 Worker 的答案 Judge 使用当前配置的模型并记录模型、Prompt 版本与 SHA-256。私有集只允许本地 `ollama` provider；配置为云端 provider 时任务会在发送内容前失败。评测报告只保留 query ID、分类、状态、耗时、分数和错误类型，不保存私有问题、答案或引用全文。Judge 失败时相关指标明确标记为不可用，不会用启发式分数冒充 Faithfulness。

当前公开集仅有 6 条种子样例，明确未达到计划中的约 100 条人工复核规模，并设置 `eligible_for_quality_gate=false`。生产资产校验和云端 Judge 现在都采用默认拒绝策略：只有同时满足 `eligible_for_quality_gate=true` 与 `review_status=human_reviewed` 的正式数据集才会运行。`evaluation/baseline.pending.json` 继续让质量门禁主动失败；只有扩充、人工复核并在固定 corpus/chunk/k/硬件/重复次数下复现后，才能冻结正式 baseline。不得把种子集结果作为生产质量指标。

`evaluation/candidates/public_golden_candidate.json` 是固定版本 CMRC 2018、SQuAD 2.0 和 HotpotQA 生成的 100 条候选原料（45 中文事实、15 英文事实、15 无答案、25 多跳），同时包含 324 个公开 corpus 文档。它明确设置 `eligible_for_quality_gate=false`，CI Judge 会拒绝将其当作正式 final 集；逐条人工复核并正式提升前不能称为黄金集。可用以下命令确定性重建：

```powershell
.\.venv\Scripts\python.exe scripts\build_public_golden_candidates.py
```

管理员可在 Dashboard 的“黄金集复核”页逐条查看问题、答案要点和证据全文，并标记为批准、退回或待定。复核状态保存在 SQLite schema v3 的 `golden_reviews` 表，并绑定候选文件 SHA-256；候选文件改变后旧审批不会沿用。只有全部候选均获批准时，Dashboard 才允许导出 `evaluation/reviewed/public_golden_reviewed-<sha>.json`。该导出物仍保持 `eligible_for_quality_gate=false`，不会自动覆盖 `evaluation/public_golden.json`；正式提升和 baseline 冻结必须作为独立、可审查的仓库变更完成。

复核者确认导出物后，用独立命令生成正式文件；该命令会核对逐条审核记录、样例 ID 和 schema，并原子替换目标文件：

```powershell
.\.venv\Scripts\python.exe scripts\promote_reviewed_golden.py `
  --reviewed evaluation/reviewed/public_golden_reviewed-<sha>.json `
  --output evaluation/public_golden.json `
  --version 2.0.0
```

正式公开集保留完整 corpus manifest。云端质量流程按每条样例的全部 `expected_document_ids` 导入证据，可正确覆盖 HotpotQA 多文档问题；无答案样例的相关公开 passage 也会被导入，避免把拒答测试简化成空语料测试。所有路径仍必须位于仓库内。

正式提升后的公开集和本机私有集都必须显式设置 `review_status=human_reviewed` 与 `eligible_for_quality_gate=true`。完成两套 dev split 建索引后，使用实际生产检索链路校准 EvidenceGate：

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_evidence_gate.py `
  --public evaluation/public_golden.json `
  --private evaluation/private_golden.json `
  --public-collection public_evidence_calibration `
  --private-collection default
```

脚本要求公开和私有 dev 集各自同时含有答案与无答案样例，拒绝使用降级检索，输出 `evaluation/evidence_calibration.json`，但不会自动修改生产配置。人工检查产物后，把其中阈值写入 `config/settings.yaml`，设置 `calibration_status: calibrated` 和 `calibration_artifact: evaluation/evidence_calibration.json`。启动时会重新验证数据集哈希、公开/私有 dev 覆盖、阈值可复现性，以及 embedding、reranker、检索、chunk 和索引配置指纹；任一不一致都会拒绝启动，要求重新校准。

质量门槛代码已固定为 Faithfulness ≥ 0.85、引用覆盖率 ≥ 0.95、Recall@5 ≥ 0.80、相对主分支回退 ≤ 0.02。DeepSeek Job 缺 Secret、Judge 失败或 baseline 未冻结都会失败，不会静默跳过。

### 真实候选检索评测（非正式 baseline）

2026-08-13 在 Windows 11、CPU-only、Qdrant `modular_rag_benchmark_v1`、Ollama `qwen3-embedding:0.6b` 与 `BAAI/bge-reranker-v2-m3` 上，对 21 条 legacy candidate query 每种模式重复 5 次。四种模式各 105 次测量，错误、降级和不可用模式均为 0：

| 模式 | Hit@1 | Hit@5 | MRR@10 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| Dense | 66.67% | 100% | 0.8135 | 2149.8 ms | 2182.0 ms |
| BM25 | 90.48% | 100% | 0.9524 | 8.6 ms | 30.7 ms |
| Hybrid RRF | 85.71% | 100% | 0.9087 | 2158.2 ms | 2177.2 ms |
| Hybrid + Cross-Encoder | 85.71% | 100% | 0.9206 | 16349.0 ms | 17144.9 ms |

报告位于 `data/eval_results/benchmark_candidate_5x.json`（运行产物，不提交语料全文），固定了 corpus/query SHA-256、模型、维度、RRF k、Python 与平台。该小样本只证明评测链路可复现；其中 BM25 最优，Cross-Encoder 仅轻微改善 MRR，却把 P50 增加到约 16.3 秒，因此不能声称 Rerank 在当前 CPU 部署上带来净收益，更不能替代约 100 条人工复核公开 final 集的正式 baseline。

## Windows EXE

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\scripts\build_windows.ps1
```

EXE 是服务器启动器，不把 Ollama、Qdrant 或模型权重打包进单文件。它检查端口、启动 API/Worker/Dashboard、等待 API 存活检查、打开浏览器并在退出时安全终止子进程。Windows Job Object 会在启动器正常关闭或异常退出时回收整个子进程树，避免 API、Worker 或 Dashboard 残留占用端口。生产就绪状态仍由 `/health/ready` 单独报告；EvidenceGate 未校准或 Reranker 不可用时返回 503，但不会阻断管理员进入首次配置界面。未校准时 `/answer` 和共享应用服务会在检索/LLM 调用前 fail-closed；`/search` 保持可用，以支持索引诊断和阈值校准。

Cross-Encoder 模型默认缓存于 `%LOCALAPPDATA%\ModularRAG\models\huggingface`，不会因替换 `dist\ModularRAGLauncher` 而丢失。首次下载和预热在 API 后台执行：`/health/live` 与管理员界面立即可用，`/health/ready` 在模型完成预热前保持 503，避免把未加载好 Reranker 的实例接入生产流量。

`rerank.cache_dir` 可配置为其他明确目录；默认值 `user` 会覆盖机器上遗留的 `HF_HOME` 路径选择，不依赖或写入原知识库的模型目录。Windows 构建脚本也把依赖分析阶段的 Hugging Face 缓存显式指向同一用户缓存目录。

## 已知限制

- 需要独立启动 Qdrant；当前环境若 6333 未监听，摄取和查询会返回明确的服务不可用错误。
- SQLite WAL 适合单节点服务，不支持跨主机 Worker 或自动故障转移。
- EvidenceGate 阈值目前是保守配置值，`calibration_status` 仍为 `pending`；校准工具和启动校验已就绪，但目标规模公开/私有 dev 集尚未完成复核和实测，因此不能宣称质量门槛已经达成。
- 公开黄金集尚未扩充到 50–200 条，私有黄金集必须由管理员在本机人工整理。
- DeepSeek 公共 final 评测只允许仓库公开语料；私有知识库不得进入该流程。
- Windows EXE 已完成当前工作树 PyInstaller 构建、API/Worker/Dashboard 联合启动、健康探测和父进程异常退出的整棵子进程树回收验证；首次模型下载会正确写入用户缓存。
- Qdrant + Ollama 单文档生产链路已用隔离的 `modular_rag_production_acceptance_20260813_v1` collection 完成真实摄取和 Dense/BM25/RRF 查询验收，三路均命中预期稳定文档。`--no-rerank` 已验证不会实例化 Cross-Encoder，且会明确标记最终顺序来自 RRF。
- `D:\AI-KnowledgeBase\documents` 已通过只读目录数据源同步到隔离的 `modular_rag_production_local_acceptance_20260813_v1`：2 个允许文件、2 个 Qdrant point、0 失败；第二次同步为 `unchanged=2`、`processed=0`。中文查询的 Dense、BM25 和 RRF 均命中正确 Markdown。公开质量 baseline 仍需在人工标注完成后执行。

## MCP

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "D:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "D:\\path\\to\\MODULAR-RAG-MCP-SERVER"
    }
  }
}
```

暴露 `query_knowledge_hub`、`list_collections` 和 `get_document_summary` 三个工具；依赖服务不可用时返回结构化 Tool 错误而不会破坏 stdio 协议。
