# Modular RAG MCP Server — 检索评测报告

> **测评日期**：2026-06-19 | **Embedding**：BAAI/bge-m3 (1024-dim, SiliconFlow) | **BM25**：jieba + rank_bm25

---

## 1. 评测目标

验证四种检索模式在同一知识库上的性能差异，量化 Dense、Sparse、Hybrid 三种策略的效果，为工程配置选型提供数据支撑。

---

## 2. 测试集与知识库

### 2.1 知识库

| 文档 | 内容 | Chunks |
|------|------|--------|
| `complex_technical_doc.pdf` | Advanced RAG System 英文技术文档（架构、分块、Embedding、检索等） | 12 |
| `chinese_technical_doc.pdf` | 大语言模型应用开发技术指南（LLM 基础、RAG、BM25、MCP 等） | 12 |
| `chinese_table_chart_doc.pdf` | RAG 系统性能评测报告（含表格数据、对比实验、配置推荐） | 7 |
| `chinese_long_doc.pdf` | 大模型面试八股知识手册（Transformer、微调、Agent、评估等，30+页） | 39 |
| **合计** | | **70 chunks** |

> 摄取参数：`chunk_size=1000, chunk_overlap=200`，规则分块（无 LLM 增强），Embedding 维度 1024。

### 2.2 黄金测试集

21 条手工标注的问答对，覆盖中英双语和多种问题类型：

| 类型 | 数量 | 占比 |
|------|------|------|
| 事实型（factual） | 13 | 61.9% |
| 对比型（comparison） | 4 | 19.0% |
| 概念型（conceptual） | 3 | 14.3% |
| 多步推理型（multi_step） | 1 | 4.8% |

> **评测指标**：**文档级来源匹配**（Document-Level Source Matching）—— 若检索结果中任意 Chunk 的 `source_path` 包含期望文件名，则计为命中，无需精确到 Chunk ID。

---

## 3. 评测方法

```
模式 A：Dense Only       — 仅向量语义检索（BGE-m3），禁用 BM25
模式 B：Sparse Only      — 仅 BM25 关键词检索（jieba 分词），禁用向量检索
模式 C：Hybrid (RRF)     — Dense + BM25 双路并行，RRF 融合（k=60）
模式 D：Hybrid + Rerank  — Hybrid 基础上叠加可选重排器（当前 rerank.enabled=false，与 Hybrid 等价）
```

**Metrics**：
- **Hit@1**：Top-1 结果命中期望来源文档的比例
- **Hit@5**：Top-5 内至少 1 个结果命中的比例
- **MRR@10**：前 10 位中首次命中位置的倒数均值（Mean Reciprocal Rank）
- **平均延迟**：单次查询的端到端耗时（ms）

---

## 4. 评测结果

### 4.1 核心指标对比

| 检索模式 | Hit@1 | Hit@5 | MRR@10 | 平均延迟 | 备注 |
|---------|-------|-------|--------|---------|------|
| **Dense Only** | 66.7% | **100.0%** | 0.7937 | 315 ms | 纯语义向量检索基线 |
| **Sparse Only (BM25)** | **90.5%** | **100.0%** | **0.9524** | **14 ms** | 精确关键词匹配，最快 |
| **Hybrid (RRF)** | 76.2% | **100.0%** | 0.8810 | 259 ms | 融合检索，Hit@1 优于 Dense |
| **Hybrid + Rerank** | 76.2% | **100.0%** | 0.8810 | 276 ms | 当前 reranker 未启用，与 Hybrid 等价 |

> 测试集：21 条，Top-K=10。延迟含 Embedding API 调用开销。

### 4.2 与 Dense 基线的相对提升

| 对比 | Hit@1 提升 | MRR@10 提升 |
|------|-----------|------------|
| Hybrid vs Dense | **+9.5 pp** | **+0.0873** |
| BM25 vs Dense | **+23.8 pp** | **+0.1587** |

### 4.3 逐题 Hit@1 热力图（Dense vs Sparse vs Hybrid）

| QID | 问题摘要 | Dense | Sparse | Hybrid |
|-----|---------|-------|--------|--------|
| q001 | What embedding providers... | ✅ | ✅ | ✅ |
| q002 | Key features of modular RAG... | ✅ | ✅ | ✅ |
| q003 | How does chunking work... | ✅ | ✅ | ✅ |
| q004 | Default chunk size and overlap... | ✅ | ✅ | ✅ |
| q005 | Reranking options... | ❌ | ✅ | ✅ |
| q006 | BM25 算法基本原理... | ✅ | ✅ | ✅ |
| q007 | RRF 融合算法公式... | ❌ | ✅ | ✅ |
| q008 | ChromaDB 特点... | ❌ | ✅ | ❌ |
| q009 | Naive RAG vs Advanced RAG... | ✅ | ✅ | ✅ |
| q010 | Modular RAG 设计理念... | ❌ | ✅ | ✅ |
| q011 | Cross-Encoder vs Bi-Encoder... | ✅ | ❌ | ❌ |
| q012 | MCP 协议与工具... | ✅ | ✅ | ✅ |
| q013 | 混合检索+重排 NDCG@10... | ✅ | ✅ | ✅ |
| q014 | 推荐分块参数... | ✅ | ✅ | ✅ |
| q015 | Dense vs BM25 Precision@5... | ✅ | ✅ | ✅ |
| q016 | BGE-large-zh 维度... | ❌ | ✅ | ❌ |
| q017 | KV Cache 技术... | ✅ | ✅ | ✅ |
| q018 | LoRA 微调原理... | ✅ | ✅ | ✅ |
| q019 | 语义缓存... | ❌ | ❌ | ❌ |
| q020 | HNSW 索引特点... | ❌ | ✅ | ❌ |
| q021 | RAG 安全风险... | ✅ | ✅ | ✅ |
| **合计命中** | | **14/21** | **19/21** | **16/21** |

---

## 5. 结果分析

### 5.1 BM25 在中文技术领域的优势

**Sparse Only（BM25）取得最高 Hit@1（90.5%）** 并非偶然。原因分析：

1. **领域术语精确匹配**：测试问题中包含大量精确技术术语（"BM25"、"RRF"、"ChromaDB"、"HNSW"、"LoRA"、"KV Cache"），jieba + rank_bm25 对这类关键词的命中能力非常强。
2. **知识库规模适中**：70 chunks 的小规模知识库中，BM25 的倒排索引几乎没有稀疏性问题。
3. **速度优势显著**：14ms vs Dense 的 315ms，比率达 **22.5×**，对延迟敏感场景极具价值。

### 5.2 Hybrid RRF 的融合效果

Hybrid 相比 Dense：
- **修复了 3 条 Dense 失败的查询**（q005、q007、q010）：这些问题中关键词信号强于语义信号，BM25 路正确识别，RRF 融合提升了 Top-1 排名。
- **Hit@1 从 66.7% 提升至 76.2%（+9.5pp）**，MRR@10 从 0.7937 提升至 0.8810（+11.0%）。
- **引入 1 条回归**（q011 "Cross-Encoder vs Bi-Encoder"）：Dense 在此问题上正确 Top-1，RRF 融合后排名被调整至 Top-2，MRR 从 1.0 降为 0.5。这是典型的融合代价。

### 5.3 全局 Hit@5 = 100%

**所有模式在 Hit@5 上均达到 100%**，说明知识库覆盖度充分：每道测试题的答案都在任意模式的 Top-5 检索结果内。这验证了知识库构建质量良好，文档分块策略合理。

### 5.4 Hybrid+Rerank 说明

当前配置 `rerank.enabled: false`，Hybrid+Rerank 模式等价于 Hybrid。若启用 Cross-Encoder Reranker（`rerank.provider: cross_encoder`），可在 Hybrid 基础上进一步精排 Top-5 结果，预期可将 Hit@1 提升至 80-90%，代价是额外引入 Cross-Encoder 推理延迟（约 +50~200ms）。

### 5.5 唯一全模式失败案例：q019（语义缓存）

q019"什么是语义缓存？"在三种模式下均未 Top-1 命中（Expected: `chinese_long_doc.pdf`）。  
原因：**"语义缓存"** 这一术语同时出现在 `chinese_technical_doc.pdf`（第七章缓存策略）和 `chinese_long_doc.pdf`（第十章成本优化），前者内容更集中，语义向量更相似，但标注期望是 `chinese_long_doc.pdf`。这揭示了**跨文档内容重叠**是 RAG 检索的固有挑战，需要更细粒度的 Chunk 标注或内容去重。

---

## 6. 配置建议

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| **中文技术文档 QA（低延迟）** | Sparse Only (BM25) | 14ms，Hit@1=90.5%，适合实时响应 |
| **通用英文 QA / 语义相关** | Hybrid (RRF) | 综合精度优于 Dense，延迟可接受 |
| **高精度场景（可接受慢）** | Hybrid + Cross-Encoder Reranker | 启用 `rerank.provider: cross_encoder`，最大化 Hit@1 |
| **大规模知识库** | Hybrid (RRF) | BM25 稀疏性增加时，Dense 互补价值更大 |

**settings.yaml 快速切换示例：**

```yaml
# 启用 Cross-Encoder Reranker（Hybrid+Rerank 模式）
rerank:
  enabled: true
  provider: cross_encoder
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 5
```

---

## 7. 运行方式（可复现）

```bash
# Step 1: 摄取测试文档
python scripts/ingest_benchmark_docs.py

# Step 2: 运行 4 模式基准测试
python scripts/run_benchmark.py --skip-ingest

# 结果输出至 data/eval_results/benchmark_comparison.json
```

> **Embedding 模型**：BAAI/bge-m3 via SiliconFlow API（1024-dim）  
> **BM25 分词**：jieba（中文）  
> **测试环境**：Windows 11, Python 3.12, ChromaDB local persistence  

---

## 8. 附录：各模式逐题 MRR@10

| QID | 类别 | Dense | Sparse | Hybrid |
|-----|------|-------|--------|--------|
| q001 | 事实 | 1.000 | 1.000 | 1.000 |
| q002 | 事实 | 1.000 | 1.000 | 1.000 |
| q003 | 事实 | 1.000 | 1.000 | 1.000 |
| q004 | 事实 | 1.000 | 1.000 | 1.000 |
| q005 | 事实 | 0.333 | 1.000 | 1.000 |
| q006 | 事实 | 1.000 | 1.000 | 1.000 |
| q007 | 事实 | 0.333 | 1.000 | 1.000 |
| q008 | 事实 | 0.500 | 1.000 | 0.500 |
| q009 | 对比 | 1.000 | 1.000 | 1.000 |
| q010 | 概念 | 0.333 | 1.000 | 1.000 |
| q011 | 对比 | 1.000 | 0.500 | 0.500 |
| q012 | 事实 | 1.000 | 1.000 | 1.000 |
| q013 | 事实 | 1.000 | 1.000 | 1.000 |
| q014 | 事实 | 1.000 | 1.000 | 1.000 |
| q015 | 对比 | 1.000 | 1.000 | 1.000 |
| q016 | 事实 | 0.333 | 1.000 | 0.500 |
| q017 | 事实 | 1.000 | 1.000 | 1.000 |
| q018 | 概念 | 1.000 | 1.000 | 1.000 |
| q019 | 概念 | 0.333 | 0.500 | 0.500 |
| q020 | 事实 | 0.500 | 1.000 | 0.500 |
| q021 | 多步推理 | 1.000 | 1.000 | 1.000 |
| **均值** | | **0.794** | **0.952** | **0.881** |

---

*本报告由 `scripts/run_benchmark.py` 自动生成，原始数据见 `data/eval_results/benchmark_comparison.json`。*
