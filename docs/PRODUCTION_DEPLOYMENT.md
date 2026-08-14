# 单机生产部署与安全边界

1. 服务器只在受信局域网开放 8501/8766；6333 与 11434 仅监听本机或由主机防火墙阻断外部访问。
2. 首次启动在 Dashboard 创建管理员。系统无默认账号或默认密码。
3. `D:\AI-KnowledgeBase\documents` 仅作为只读源；应用写入仅限项目 `data/`、Qdrant 项目 collection 和日志目录。
4. 备份 SQLite `data/db/production.db*`、Qdrant snapshot、`config/` 和 `config/prompts/`。恢复时保持 Prompt、配置、模型和索引版本一致。
5. 每次 Prompt、模型、配置或索引切换应写 deployment event；先在 dev split 校准，再验证 final split。
6. Trace 正文保存 30 天后清理；长期聚合只保留哈希、版本、Token、延迟、状态和错误类别。
7. 网页摄取只允许 HTTP(S)、同域、公开 IP 和配置路径；禁止 localhost、私网 IP、跨域跳转和无限抓取。
8. SQLite WAL 是单节点边界。不要在共享网络文件系统运行数据库，也不要启动多个独立 Worker 争用相同部署。
9. 私有黄金集的答案评测只允许本地 Ollama Judge。公开 PR 的 DeepSeek Judge 只能读取仓库内公开 final split；两条链路禁止混用。
10. Judge 的模型、温度、Prompt ID、版本和哈希写入报告；Judge 解析或调用失败时 Faithfulness、Answer Relevancy 和引用覆盖率标为不可用，不得冻结 baseline。
11. 启动器以 `/health/live` 判断 API 进程已经启动；部署监控和流量切换必须使用 `/health/ready`。后者在 Reranker 不可用或 EvidenceGate 未校准时返回 503，这是预期的 fail-closed 行为。
12. EvidenceGate 不能靠手工把状态改成 `calibrated`。先在人工复核且显式允许质量门禁的公开/私有 dev 集上运行 `scripts/calibrate_evidence_gate.py`，检查 `evaluation/evidence_calibration.json`，再同步阈值和产物路径。启动会验证数据集哈希、阈值、观察值和检索配置指纹；校准后修改 embedding、reranker、chunk、检索或索引版本必须重新校准。
13. Cross-Encoder 权重默认缓存到 `%LOCALAPPDATA%\ModularRAG\models\huggingface`。服务账户必须对该目录有读写权限；不要把缓存指向只读的 `D:\AI-KnowledgeBase` 原知识库目录。
