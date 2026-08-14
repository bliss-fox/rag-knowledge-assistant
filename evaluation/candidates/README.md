# 公开黄金集候选区

`public_golden_candidate.json` 和 `public_corpus/` 来自固定版本的公开数据集，只是人工标注的候选原料。

- `eligible_for_quality_gate` 必须保持 `false`。
- 自动生成器不得把 `human_reviewed` 改为 `true`。
- 每条问题、答案要点、证据文档、类别、难度和 dev/final 划分必须人工复核。
- 复核完成后通过独立变更提升到 `evaluation/public_golden.json`，并记录复核者和日期。
- 不得把 `D:\AI-KnowledgeBase` 的私有内容复制到此目录。

## Dashboard 复核和导出

管理员在 Dashboard 的“黄金集复核”页面逐条批准、退回或标记待定。每条决定保存在 SQLite `golden_reviews` 表中，并与当前候选文件的 SHA-256 绑定。候选文件发生任何变化后，旧复核记录都不能用于新版本。

全部 100 条均批准后，可点击“导出已复核产物”。系统会原子写入：

```text
evaluation/reviewed/public_golden_reviewed-<candidate-sha-prefix>.json
```

导出物包含来源候选哈希和逐条复核记录，但仍设置 `eligible_for_quality_gate=false`。它是正式提升变更的输入，不是可直接进入 CI 的正式黄金集。

经独立代码审查后运行：

```powershell
.\.venv\Scripts\python.exe scripts\promote_reviewed_golden.py `
  --reviewed evaluation/reviewed/public_golden_reviewed-<sha>.json `
  --output evaluation/public_golden.json `
  --version <new-version>
```

提升脚本验证所有 query ID 都有且只有批准记录，验证 schema 后才原子替换正式集。正式集才会设置 `review_status=human_reviewed` 与 `eligible_for_quality_gate=true`。

重建命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_public_golden_candidates.py
```
