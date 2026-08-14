# 生产验收记录（2026-08-13）

本记录只陈述真实执行结果，不代替尚未完成人工复核的正式质量 baseline。

## 发布前回归（2026-08-14）

- 在 Python 3.14.6 / Windows 上执行 `pytest -m "not llm"`：`1501 passed, 4 skipped, 36 deselected`，耗时 80.88 秒。
- 该数字证明代码、API 合约、Dashboard、MCP、任务与数据生命周期的自动化回归结果，不代表正式 Faithfulness、Recall 或引用覆盖率已经达标。

## Windows 启动器

- PyInstaller 成功生成 `dist/ModularRAGLauncher/ModularRAGLauncher.exe`。
- 21:55 完成包含最新子进程管理、统一 fail-closed、安全数据源删除和网页增量同步的重建；对该最新二进制再次执行了真实启动验收（105,551,334 字节）。
- 最新 EXE 在 60 秒内返回 API `/health/live = 200` 和 Dashboard `/_stcore/health = 200`。
- `/health/ready = 503`，正确报告首次管理员尚未创建、EvidenceGate `pending`，以及当前受控执行账户对 `%LOCALAPPDATA%` 模型缓存只有读取权限导致的 Reranker 预热失败；没有把未就绪实例接入回答流量。
- Cross-Encoder 缓存创建于 `%LOCALAPPDATA%\ModularRAG\models\huggingface`；运行日志没有访问 `D:\AI-KnowledgeBase\models` 或 `dist\data\models`。
- 强制终止父启动器后，Windows Job Object 在 3 秒内清理 API、Worker、Dashboard，8766/8501 不再监听。

## Fail-closed 与数据生命周期

- EvidenceGate 未校准时，统一应用服务的 `answer()` 会在创建检索 pipeline 和调用 LLM 之前直接拒答，原因标记为 `evidence_gate_uncalibrated`；该约束覆盖 FastAPI、评测 Worker 和其他直接复用应用服务的入口。
- `/answer` 在阈值未校准时返回 503；只读 `/search` 仍可用于检索诊断和阈值校准，Reranker 不可用时按原设计显式降级到 RRF。
- 删除数据源改为持久化后台任务，并与同源同步共用互斥资源键。Worker 先读取 manifest，清理该来源独占的 Qdrant chunk 和 BM25 文档，再删除 SQLite 数据源；共享 chunk/document 不会被误删。
- 相关定向与 Dashboard 合约测试 34 项全部通过；完整非模型单元/集成回归为 `1453 passed, 1 skipped, 40 deselected`。

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

- 100 条公开候选和私有黄金集尚未完成人工复核。
- EvidenceGate 尚未基于公开、私有 dev split 校准，因此继续 fail-closed。
- 正式 Hybrid + Cross-Encoder 质量评测和冻结 baseline 尚未执行。
- 当前受控执行账户无法写入用户 Hugging Face 缓存，尚未完成 Cross-Encoder 全量下载和成功预热；正常交互式 Windows 用户拥有该目录权限，仍需在最终部署账户下复验。

## 网页与目录增量同步补充验收

- 网页页面身份现在由 `source_id + canonical URL` 的 SHA-256 决定；内容变化不会改变快照路径或来源身份。
- 网页同步返回 `added`、`modified`、`deleted`、`unchanged`，只有新增和修改页面进入摄取；连续完整抓取相同内容时 `processed=0`。
- 抓取达到 `max_pages` 且仍有未访问的同源链接时，结果标记 `complete=false`、`deletion_skipped_reason=page_limit_reached`，不会把未返回页面误判为删除。
- 修改或删除页面时只移除该 manifest 独占的 Qdrant chunk 和 BM25 文档；同一来源或其他来源仍引用的 ID 保留。
- 删除网页数据源时，Worker 在索引和 SQLite 清理完成后只删除 `data/uploads/web/<source_id>` 下由应用生成的快照；路径逃逸、符号链接或非目录目标会 fail-closed，其他来源快照保持不变。
- 任务取消发生在摄取期间时会中止同步，不执行随后页面删除；网页快照仅从应用自身目录删除，不修改远端网页。
- 目录同步会重试状态为 `discovered` 或 `failed` 的文件，并由来源 manifest 负责幂等；全局摄取历史不会再阻止相同内容索引到另一个独立 collection。
- 新增的网页、SSRF、manifest、取消和安全删除定向测试共 17 项通过；完整非模型单元/集成回归为 `1462 passed, 1 skipped, 40 deselected`。
