# Evaluation Run Plan (Stage K)

## 目标

把现有评测代码跑出真实数字，写进 README，让项目从"有评测代码"变成"有评测数据"。

**简历上最有力的句式：** "Hybrid Search vs 纯向量检索，MRR@10 提升 X%，Hit@5 达到 Y%（Ragas 实测）"

## K1: Golden Test Set 准备

**文件：** `tests/fixtures/golden_test_set.json`

格式：
```json
[
  {
    "id": "q001",
    "question": "什么是 BM25 算法？",
    "golden_answer": "BM25 是基于概率检索框架的词频-逆文档频率算法...",
    "relevant_chunk_ids": ["chunk_abc123", "chunk_def456"],
    "relevant_doc_sources": ["data/docs/rag_fundamentals.pdf"]
  },
  ...
]
```

**要求：**
- 至少 20 个 QA 对
- 覆盖不同类型问题：事实型、对比型、多步推理型
- relevant_chunk_ids 需提前摄取文档后从 ChromaDB 中找到对应 ID

**生成方式：** 先摄取几个文档 → 用 `scripts/query.py` 手动确认哪些 chunk 被命中 → 手工标注

## K2: 实跑评测对比

评测对比矩阵：
```
模式A: Dense Only（关闭 BM25）
模式B: Sparse Only（BM25 only）
模式C: Hybrid（BM25 + Dense + RRF）  ← 默认，应该最好
模式D: Hybrid + Rerank               ← 最优，但慢
```

**运行脚本：**
```bash
# 运行评测（基于现有 EvalRunner）
python scripts/evaluate.py \
  --test-set tests/fixtures/golden_test_set.json \
  --modes dense,sparse,hybrid,hybrid_rerank \
  --output data/eval_results/comparison.json
```

或者直接用现有 EvalRunner：
```python
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.evaluator.eval_runner import EvalRunner

runner = EvalRunner(settings, test_set_path="tests/fixtures/golden_test_set.json")
results = runner.run()  # 返回各指标数值
```

**要收集的指标：**

| 指标 | 说明 | 目标值 |
|------|------|------|
| Hit@1 | Top1 是否命中 | Hybrid > Dense > Sparse |
| Hit@5 | Top5 是否命中 | 期望 >70% |
| MRR@10 | Mean Reciprocal Rank | Hybrid > 其他 |
| Faithfulness | Ragas：回答是否忠实于 context | >0.7 |
| Answer Relevancy | Ragas：回答相关性 | >0.7 |

## K3: 结果固化进 README

**README 评测对比表模板：**
```markdown
## Benchmark Results

评测数据集：20 个领域相关 QA 对（手工标注），基于 [文档集合名] 知识库。

| 检索模式 | Hit@1 | Hit@5 | MRR@10 | 备注 |
|---------|-------|-------|--------|------|
| Dense Only | X% | X% | X.XX | 纯向量检索基线 |
| BM25 Only | X% | X% | X.XX | 纯稀疏检索基线 |
| Hybrid (RRF) | X% | X% | X.XX | **+X% vs Dense** |
| Hybrid + Rerank | X% | X% | X.XX | 最优，耗时 +Xs |

Ragas 指标（Hybrid + Rerank 模式）：
- Faithfulness: X.XX
- Answer Relevancy: X.XX
```

## 注意事项

1. 如果 Ragas 需要 OpenAI API 才能计算，可只用自定义指标（Hit@K + MRR）代替
2. 对比数字只要方向正确（Hybrid > Dense）即有说服力，绝对值不必完美
3. 数字要真实，不要编造——面试官可能追问数据集和跑法
