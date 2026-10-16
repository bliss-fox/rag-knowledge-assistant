# 生产验收记录（2026-08-13）

本记录只陈述真实执行结果，不代替尚未完成人工复核的正式质量 baseline。

## 发布前回归（2026-08-14）

- 在 Python 3.14.6 / Windows 上执行 `pytest -m "not llm"`：`1568 passed, 4 skipped, 36 deselected`，耗时 117.95 秒。该轮包含旧 JSONL Trace 迁移、SQLite schema v5、公开/私有黄金集五项结构化复核证明、Dashboard/API、公开质量 fail-closed 预检、固定版本 CMRC 2018 / XQuAD 中文 / SQuAD 2.0 / HotpotQA / MultiHop-RAG / RAGBench FinQA 公开候选生成回归、下载哈希防漂移、完整公开 corpus 摄取约束、统一本机依赖预检、不可变 Cross-Encoder snapshot 加载，以及 Windows 原生 Qdrant 安装、归档安全、PID 防误杀和交付包入口测试。
- 该数字证明代码、API 合约、Dashboard、MCP、任务与数据生命周期的自动化回归结果，不代表正式 Faithfulness、Recall 或引用覆盖率已经达标。
- 离线回归在 Qdrant 停止时执行，以保持原有确定性 fake/mock 口径。Qdrant 运行时旧 E2E 摄取用例会进入真实模型与索引路径，单个样例约 30 秒；真实依赖联调单独记录在下节，不能与离线回归耗时混为一谈。

## 本机依赖统一预检（2026-08-14）

- 新增 `scripts/preflight_local_dependencies.py`，真实检查 Ollama 模型清单、Embedding、
  短生成、Qdrant collections API 和 Cross-Encoder；报告不包含探针、回答、collection
  名称或知识库正文，失败时返回非零退出码。
- 实测 `qwen3-embedding:0.6b` 返回 1024 维，耗时约 2178 ms；`qwen3:8b` 短生成调用
  成功，输入/输出 Token 为 19/8，热缓存耗时约 209 ms。这是单次依赖诊断，不是
  P50/P95 性能结论。
- `BAAI/bge-reranker-v2-m3` 的已有缓存最初只有 config 和权重，缺少 tokenizer。补全
  4 个 tokenizer 文件后，加载器会识别完整 immutable snapshot 并以
  `local_files_only` 加载，不再在服务启动时依赖 Hub 锁写入或网络元数据探测。
- CPU 上 Cross-Encoder 预热和双候选排序成功，阶段耗时约 5806 ms；中文相关候选得分
  约 0.9997，无关英文候选约 0.0000。该探针只证明模型可加载并具有基本排序能力，
  不代替正式黄金集上的净收益评测。
- Docker Desktop 仍因主机未启用 Virtual Machine Platform、无可用虚拟化而不能提供
  Linux Engine。新增 Windows 原生 Qdrant 路径后，统一预检中的 Qdrant collections API
  检查耗时约 40 ms、`ok=true`；完整预检五项全部通过，最终 `ok=true`、退出码 0。
- 同一轮预检中，Ollama 清单约 2145 ms、1024 维 Embedding 约 3438 ms、8 Token 短生成
  约 1600 ms、CPU Cross-Encoder 预热与排序约 6063 ms。它们是单次依赖诊断，不是
  P50/P95 性能结论。

## Windows 原生 Qdrant（2026-08-14）

- 原生管理器固定 Qdrant `v1.15.4` 官方 Windows MSVC ZIP，下载前后校验 SHA-256
  `5348af719d5d9f3658597c8d3a8231bb8c38e7139a8076b69bd3dd59151d06d5`；真实安装到
  `%LOCALAPPDATA%\ModularRAG\qdrant\v1.15.4` 成功。
- 进程环境固定 REST/gRPC 为 `127.0.0.1:6333/6334`、关闭遥测，storage 独立位于
  `%LOCALAPPDATA%\ModularRAG\qdrant\storage`。首次启动、状态检查均返回 healthy；没有
  创建、读取或修改保留的 `local_knowledge` collection。
- 创建隔离测试 collection `modular_rag_native_manager_acceptance_v1` 并写入 1 个合成
  point；安全停止后重新启动，精确计数仍为 1，证明 storage 跨进程重启保留。验收后已
  删除该合成 collection。
- 首次真实停机暴露 Windows 进程退出瞬间 `QueryFullProcessImageNameW` 可能返回
  `WinError 5`。轮询已改为在进程路径已验证且健康端点与端口均消失时安全清理自己的
  PID 记录，并增加回归；修复后真实 start/stop/start 全部成功，最终 Qdrant 保持运行。
- `qdrant-client 1.19.0` 会警告与服务端 1.15.4 超出兼容窗口，依赖现已收紧并实装为
  `qdrant-client 1.15.1`。使用项目 `QdrantStore` 完成隔离 collection 的 upsert、query、
  stats 和删除，Top-1 返回预期稳定 ID；未触碰原知识库 collection。

## Windows 启动器

- PyInstaller 成功生成 `dist/ModularRAGLauncher/ModularRAGLauncher.exe`。
- 2026-08-14 21:06:23 完成包含原生 Qdrant 管理入口的重新构建；二进制大小为
  105,609,081 字节，SHA-256 为
  `61087415f8a684c05e2d4adc3e3ddcc82876288a151c8e219986db312270e2b6`。
- 新 EXE 执行 `--qdrant status` 成功返回安装版本 1.15.4、healthy、运行 PID 和
  `storage_preserved=true`，证明管理模块已进入交付包而不是只存在于源码脚本。
- 使用 `RAG_OPEN_BROWSER=0` 启动最终 EXE，API `/health/live = 200`、Dashboard `/_stcore/health = 200`；`GET /golden-candidates/summary` 与 `GET /private-golden-sets` 在未认证时均返回 401，证明新路由已进入交付包且仍受认证保护。
- `/health/ready = 503`，正文同时返回 `ok=false`、`schema_version=5`、`database_ok=true`、`dependencies_ok=false`、`bootstrap_required=true`，并报告 Qdrant `127.0.0.1:6333` 拒绝连接和 EvidenceGate `pending`；HTTP 状态与正文语义一致，没有把未就绪实例接入回答流量。
- 交付包包含 `src/observability/dashboard/legacy_pages` 供兼容测试使用，不包含 Streamlit 会自动发现的 `src/observability/dashboard/pages` 目录。
- Cross-Encoder 缓存创建于 `%LOCALAPPDATA%\ModularRAG\models\huggingface`；运行日志没有访问 `D:\AI-KnowledgeBase\models` 或 `dist\data\models`。
- 最终验收启动 API、Worker 与 Dashboard；强制结束实际父启动器后，在 15 秒验收窗口内确认 Windows Job Object 已清理全部子进程，8501/8766 的剩余监听数为 0，且没有残留本次启动的 `ModularRAGLauncher` 进程。
- Qdrant 恢复后启动新 EXE，API `/health/live = 200`、Dashboard `/_stcore/health = 200`；
  后台 warmup 为 `completed`，Cross-Encoder `reranker_ready=true`、`reranker_error=null`。
  `/health/ready` 仍为 503，因为新交付目录尚未创建管理员且 EvidenceGate 校准为
  `pending`，不是 Qdrant 或 Reranker 初始化失败。

## Fail-closed 与数据生命周期

- EvidenceGate 未校准时，统一应用服务的 `answer()` 会在创建检索 pipeline 和调用 LLM 之前直接拒答，原因标记为 `evidence_gate_uncalibrated`；该约束覆盖 FastAPI、评测 Worker 和其他直接复用应用服务的入口。
- `scripts/agent.py` 已改为认证 FastAPI 客户端，只调用 `/auth/login`、`/auth/refresh`、`/auth/logout` 和 `/answer`；密码不接受命令行参数，401 时使用可撤销 refresh session 重试。它不再直接实例化 LLM、VectorStore、HybridSearch 或 ReAct，并保留服务端 `trace_id`、EvidenceGate 与 Prompt/模型版本。
- MCP `query_knowledge_hub` 保持为薄检索工具，复用 `RAGApplicationService.search()` 且不直接生成最终答案；研究用 ReAct/SelfChecker 不能替代生产 EvidenceGate。
- `/answer` 在阈值未校准时返回 503；只读 `/search` 仍可用于检索诊断和阈值校准，Reranker 不可用时按原设计显式降级到 RRF。
- 删除数据源改为持久化后台任务，并与同源同步共用互斥资源键。Worker 先读取 manifest，清理该来源独占的 Qdrant chunk 和 BM25 文档，再删除 SQLite 数据源；共享 chunk/document 不会被误删。
- schema v5、公开/私有审批、提升脚本、API 与 Dashboard 的定向回归 62 项全部通过；完整非 LLM 回归为 `1553 passed, 4 skipped, 36 deselected`。

## Qdrant + Ollama roundtrip

环境：

- Qdrant `http://127.0.0.1:6333`
- Ollama `qwen3-embedding:0.6b`
- 独立物理 collection `modular_rag_production_acceptance_20260813_v1`
- 测试文档 `evaluation/candidates/public_corpus/cmrc2018/DEV_1008_QUERY_1-e8b4aa41.md`
- 查询“卒本扶余是哪个国家的一个延续？”

实际结果：

- 摄取成功：1 个稳定文档、1 个 chunk、1024 维向量、1 个 BM25 文档。
- Dense、BM25、RRF 均把预期 chunk 排在第 1 位。
- Dense 分数约 `0.535`，RRF 分数约 `0.0328`；答案证据正文包含“北扶余国”。
- `--no-rerank` 不再加载 Cross-Encoder，控制台明确显示最终顺序来自 RRF；修复前同一命令仍会尝试加载模型并把融合顺序误显示为 Rerank。

## 本地知识库只读同步

- 扫描根目录：`D:\AI-KnowledgeBase\documents`。
- 发现 3 个文件，其中允许的 1 个 PDF 和 1 个 Markdown 共 2194 字节；未发现密钥命名文件。
- 持久化任务 `9a45a00c-5b4c-41f7-8331-53a609eeacb9` 成功，`processed=2`、`failed=[]`。
- SQLite `source_files` 中两个文件均为 `indexed`，`indexed_documents` 保存了稳定 document/chunk ID。
- Qdrant `modular_rag_production_local_acceptance_20260813_v1` 状态为 green，包含 2 个 point。
- 重复同步任务成功，`added=[]`、`modified=[]`、`deleted=[]`、`unchanged=2`、`processed=0`，证明当前快照幂等。
- 查询“这个本地知识库使用什么模型生成检索向量？”时，Markdown 在 Dense、BM25、RRF 中均排名第 1，证据包含 `qwen3-embedding:0.6b`。

## 尚未完成

- 100 条公开候选尚未完成人工复核；私有集编辑与审批流程已实现，但本机尚未整理出 50–200 条正式私有样例。
- EvidenceGate 尚未基于公开、私有 dev split 校准，因此继续 fail-closed。
- 正式 Hybrid + Cross-Encoder 质量评测和冻结 baseline 尚未执行。
- Qdrant、Ollama 和 Cross-Encoder 的真实依赖预检及 EXE 后台 warmup 已完成；生产 readiness 仍因首次管理员 bootstrap 和 EvidenceGate 未校准而 fail-closed。
- Docker Compose 资产继续保留为备选，但当前主机虚拟化条件不满足；Windows 原生 Qdrant 已解除 Docker 对本机交付的阻塞。

## 旧 JSONL Trace 迁移

- `SQLiteTraceStore.import_jsonl()` 已提供逐行迁移，兼容旧 stage 顶层字段和新版
  `stage.data`；嵌套字段按新版布局优先合并。
- 默认使用原子 `INSERT OR IGNORE` 跳过已有 trace ID，不会以较旧 JSONL 记录覆盖
  当前生产 Trace；只有 CLI 明确传入 `--replace-existing` 才允许替换。
- 正常记录仍复用在线 `save()` 路径，因此 query、prompt、response、stage 内密钥和正文
  执行相同脱敏。坏 JSON、缺少字段和数据库错误按行隔离，报告只保留行号、类别和安全
  摘要，不回显原始敏感正文。
- `scripts/import_legacy_traces.py` 默认读取生产配置路径；部分失败仍完成可恢复记录的导入，
  同时返回退出码 2，避免静默丢行。定向迁移与数据库测试共 10 项通过。

## 私有黄金集本机整理流程

- SQLite schema v5 包含 `private_golden_sets`、`private_golden_cases` 以及公开/私有审批的
  五项结构化人工核对证明；已有 schema v3/v4 数据库启动时会保留原数据并补齐表/列。
  升级前没有完整证明的历史批准会自动回到 pending，防止旧布尔状态被误当成已完整复核。
- 管理员可在 Dashboard 的“私有黄金集”页创建数据集、编辑样例、查看 dev/final 与
  可回答性统计、逐条批准/退回/待定，以及确认后删除样例；普通用户导航不显示该页面，
  对应 API 也会返回 403。
- 样例内容变化会清除 reviewer、五项证明与旧审批并回到 pending。批准前必须分别确认
  问题、答案、证据、类别/难度和 split；只有 50–200 条、全部带完整证明的批准、dev/final
  均非空且 dev 同时包含可回答和不可回答样例时，服务端才允许导出。
- 导出先校验临时文件，再原子替换被 Git 忽略的
  `evaluation/private_golden.json`。同版本内容或名称变化会拒绝覆盖；新版本必须高于已有
  发布版本。幂等重导出会校验现有产物并恢复数据集的 reviewed 状态。
- 私有问题、答案要点和证据 ID 不写入审计详情；私有评测仍只允许本地 Ollama。
- 当前验收只证明整理、审批和安全导出能力已经实现，不代表真实私有黄金集已经完成，
  也不提供任何未经实跑的 Faithfulness、Recall 或延迟结论。

## 公开候选集扩充

- `evaluation/candidates/public_golden_candidate.json` 已确定性重建为版本
  `2026.09.02.3`，保持严格 100 条、`requires_human_review` 和
  `eligible_for_quality_gate=false`。
- 新增固定 SHA `51adfef1c1287aab1d2d91b5bead9bcfb9c68583` 的 XQuAD 中文配置，
  使用 CC BY-SA 4.0；16 条人工翻译中文问答使候选语言分布从 22/78 调整为
  中文/英文 38/62。
- 新增固定 SHA `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82` 的 MultiHop-RAG，
  使用 ODC-By 1.0 并在 manifest 保留归属信息；24 条样例按官方
  `question_type` 覆盖推断、跨文档比较、时间和无答案各 6 条。
- 新增固定 SHA `97808f3e5fd16ede40bbff6c2949af8139b2eb7b` 的 RAGBench FinQA，
  使用 CC BY 4.0；8 条表格/数值候选只选择 adherent、无 unsupported sentence 且
  completeness=1 的固定模型输出，并明确标记答案仍需人工复核。句级相关键映射到预期
  文档索引，越界或空映射会让生成失败。
- 整体 dev/final 为 50/50，中文/英文为 38/62；100 条仍保持独立稳定 query ID。
- 所有上游数据卡和下载文件现在都固定完整 commit SHA、字节数与 SHA-256；重建时会
  核对数据卡许可证并对内容漂移默认失败。RGB 因官方限制为 CC BY-NC-SA 4.0
  非商业用途而未纳入，选择记录在 `evaluation/candidates/SOURCE_RESEARCH.md`。
- 255 个 manifest corpus 文档均存在，磁盘上没有额外陈旧 Markdown；MultiHop-RAG
  所有证据 URL 均能映射到同 revision 的 609 篇官方 corpus。定向候选测试 13 项通过，
  生产资产校验现在同时验证 6 条正式 seed 和 100 条候选及其完整 255 份 corpus manifest；候选仍保持 `eligible_for_quality_gate=false`，不会被误当成正式集。质量 Job 必须摄取 manifest 中的全部文档（含干扰文档），不能只摄取标签中的正例。
- 这些结果只证明候选原料和许可证/证据映射治理已完成。100 条仍需逐条人工复核，
  正式公开黄金集、EvidenceGate 校准和质量 baseline 仍未完成。

## 网页与目录增量同步补充验收

- 网页页面身份现在由 `source_id + canonical URL` 的 SHA-256 决定；内容变化不会改变快照路径或来源身份。
- 网页同步返回 `added`、`modified`、`deleted`、`unchanged`，只有新增和修改页面进入摄取；连续完整抓取相同内容时 `processed=0`。
- 抓取达到 `max_pages` 且仍有未访问的同源链接时，结果标记 `complete=false`、`deletion_skipped_reason=page_limit_reached`，不会把未返回页面误判为删除。
- 修改或删除页面时只移除该 manifest 独占的 Qdrant chunk 和 BM25 文档；同一来源或其他来源仍引用的 ID 保留。
- 删除网页数据源时，Worker 在索引和 SQLite 清理完成后只删除 `data/uploads/web/<source_id>` 下由应用生成的快照；路径逃逸、符号链接或非目录目标会 fail-closed，其他来源快照保持不变。
- 任务取消发生在摄取期间时会中止同步，不执行随后页面删除；网页快照仅从应用自身目录删除，不修改远端网页。
- 目录同步会重试状态为 `discovered` 或 `failed` 的文件，并由来源 manifest 负责幂等；全局摄取历史不会再阻止相同内容索引到另一个独立 collection。
- 新增的网页、SSRF、manifest、取消和安全删除定向测试共 17 项通过；完整非模型单元/集成回归为 `1462 passed, 1 skipped, 40 deselected`。

## EvidenceGate 校准完整性加固（2026-09-02）

- 校准产物升级为 `1.1.0`：公开和私有 dev split 各至少需要 20 条观测，并各有至少
  5 条检索支持与 5 条不支持样例；只有分类约束与样本规模同时满足才会标记
  `calibrated`。
- 每条观测必须携带非空 query ID、visibility、对应数据集 SHA-256、布尔支持标签和有限
  top score；重复 query ID、数据集哈希错配或降级检索都会被拒绝。
- 启动校验不再信任产物自报的覆盖率或选择结果，而是从观测重算覆盖数、阈值、拒答指标
  和正负样本数，并确认观测 query ID 完整覆盖人工复核数据集的 dev split。
- 校准与生产服务定向回归为 `35 passed`；完整非 LLM 回归为
  `1577 passed, 4 skipped, 36 deselected`，Ruff 和 `git diff --check` 均通过。
- 本机真实依赖探针确认 Qdrant 1.15.4、Ollama `qwen3-embedding:0.6b`、
  `qwen3:8b` 和 CPU Cross-Encoder 均可用；1024 维 embedding、8-token 生成与排序探针
  全部成功。本结果只证明依赖可运行，不代表正式质量阈值已经校准。
