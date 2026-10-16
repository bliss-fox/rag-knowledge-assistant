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
12. EvidenceGate 不能靠手工把状态改成 `calibrated`。先在人工复核且显式允许质量门禁的公开/私有 dev 集上运行 `scripts/calibrate_evidence_gate.py`，检查 `evaluation/evidence_calibration.json`，再同步阈值和产物路径。每种 visibility 至少需要 20 条观测，并各有至少 5 条检索支持与 5 条不支持样例；观测必须按数据集 SHA-256 和 query ID 完整覆盖 dev split。启动会重算覆盖率、阈值和选择指标，并验证数据集身份/哈希与检索配置指纹；校准后修改 embedding、reranker、chunk、检索或索引版本必须重新校准。
13. Cross-Encoder 权重默认缓存到 `%LOCALAPPDATA%\ModularRAG\models\huggingface`。首次下载或补全 snapshot 时服务账户必须可写；snapshot 完整后运行时会按不可变本地路径加载，可以使用只读缓存。不要把缓存指向 `D:\AI-KnowledgeBase` 原知识库目录。
14. GitHub 公开质量任务使用带 `rag-quality` 标签的 Windows self-hosted Runner；该账户必须能访问本机 Ollama、Qdrant、固定模型和 Cross-Encoder 缓存。GitHub Hosted Runner 只执行 Secret、人工复核状态和冻结 baseline 的 fail-closed 预检，不承担本地模型答案生成。

## Qdrant 本机部署

### Windows 原生方案（默认）

不具备 Docker Desktop 或硬件虚拟化的 Windows 服务器使用原生管理器。下载资产固定为
Qdrant `v1.15.4` 官方 Windows MSVC ZIP，固定 SHA-256 为
`5348af719d5d9f3658597c8d3a8231bb8c38e7139a8076b69bd3dd59151d06d5`。安装器在
解压前验证完整哈希，并拒绝绝对路径、父目录穿越和符号链接成员：

```powershell
.\.venv\Scripts\python.exe scripts\manage_qdrant_native.py install
.\.venv\Scripts\python.exe scripts\manage_qdrant_native.py start
.\.venv\Scripts\python.exe scripts\manage_qdrant_native.py status
.\.venv\Scripts\python.exe scripts\manage_qdrant_native.py logs --tail-lines 200
```

默认目录：

- 程序：`%LOCALAPPDATA%\ModularRAG\qdrant\v1.15.4`
- 数据：`%LOCALAPPDATA%\ModularRAG\qdrant\storage`
- PID：`%LOCALAPPDATA%\ModularRAG\qdrant\runtime\qdrant.pid.json`
- 日志：`%LOCALAPPDATA%\ModularRAG\qdrant\logs\qdrant.log`

服务环境强制设置 REST/gRPC 为 `127.0.0.1:6333/6334` 并关闭遥测。`stop` 在终止前会通过
Windows API 查询 PID 的实际可执行文件路径；安装收据、记录路径或实际进程路径任一不匹配
都会 fail-closed，避免误杀复用同一 PID 的其他程序。停止和 `install --force` 都不会删除
`storage`；强制替换时旧程序目录会改名保留为带时间戳的 backup。

PyInstaller 交付包可通过同一个启动器执行上述操作：

```powershell
.\ModularRAGLauncher.exe --qdrant install
.\ModularRAGLauncher.exe --qdrant start
.\ModularRAGLauncher.exe --qdrant status
```

### Docker Compose 备选方案

仓库同时保留固定版本的 `deploy/qdrant/compose.yaml`。它只把 REST/gRPC 端口绑定到
`127.0.0.1`，使用具名卷 `modular_rag_qdrant_storage` 保存数据，并禁止容器获得新的
Linux 权限。服务器安装并启动 Docker Desktop 后执行：

```powershell
.\scripts\manage_qdrant.ps1 pull
.\scripts\manage_qdrant.ps1 start
.\scripts\manage_qdrant.ps1 status
```

Qdrant 与 Ollama 就绪后，使用统一预检做真实但无业务数据的发布前检查：

```powershell
.\.venv\Scripts\python.exe scripts\preflight_local_dependencies.py --pretty
```

预检会检查 Ollama 模型清单、实际 Embedding 维度、短生成调用、本地 Cross-Encoder
预热/中英文排序和 Qdrant collections API。报告不保存探针、回答、collection 名称或
知识库正文；任一必需检查失败时退出码为 1。Cross-Encoder 缓存不完整、缓存权限错误、
模型缺失、维度不符、超时和连接失败使用不同的稳定错误类别。预检不会修改索引、SQLite
或知识源，但首次加载 Cross-Encoder 时可能下载并补全配置的模型缓存。

原生管理器的 `stop` 只终止经过路径验证的 Qdrant 进程；Compose 管理器的 `stop` 只停止
容器，不删除容器或具名卷。不要在生产主机运行 `docker compose down -v`，也不要手工删除
原生 `storage` 目录。
Qdrant 的 6333/6334 不应通过防火墙暴露给局域网用户，只有 FastAPI 的 8766 和
Dashboard 的 8501 按需开放。
