# 公开黄金集来源调研与许可审计

核验日期：2026-09-02

本文件说明公开候选集从哪里取得、为什么纳入，以及哪些看似合适的数据集因许可或任务
不匹配而未纳入。所有纳入文件都固定到完整上游 commit SHA，并在
`public_golden_candidate.json` 的 `sources[].dataset_card` 与 `sources[].artifacts`
中保存下载地址、字节数和 SHA-256。生成器会先核对数据卡许可证，再核对每个下载文件；
任一内容漂移都会失败，不会继续生成候选集。

## 已纳入来源

| 来源 | 本项目用途 | 许可 | 候选数 | 官方入口 |
|---|---|---|---:|---|
| CMRC 2018 | 中文精确事实 | CC BY-SA 4.0 | 22 | [Hugging Face 数据卡](https://huggingface.co/datasets/hfl/cmrc2018) / [原项目](https://github.com/ymcui/cmrc2018) |
| XQuAD `xquad.zh` | 人工翻译的中文精确事实 | CC BY-SA 4.0 | 16 | [Hugging Face 数据卡](https://huggingface.co/datasets/google/xquad) / [原项目](https://github.com/google-deepmind/xquad) |
| SQuAD 2.0 | 英文事实与不可回答问题 | CC BY-SA 4.0 | 16 | [Hugging Face 数据卡](https://huggingface.co/datasets/rajpurkar/squad_v2) / [项目主页](https://rajpurkar.github.io/SQuAD-explorer/) |
| HotpotQA | 多文档推理 | CC BY-SA 4.0 | 14 | [Hugging Face 数据卡](https://huggingface.co/datasets/hotpotqa/hotpot_qa) / [项目主页](https://hotpotqa.github.io/) |
| MultiHop-RAG | 推断、比较、时间与无答案 | ODC-By 1.0 | 24 | [Hugging Face 数据卡](https://huggingface.co/datasets/yixuantt/MultiHopRAG) |
| RAGBench `finqa` | 表格与数值证据 | CC BY 4.0 | 8 | [Hugging Face 数据卡](https://huggingface.co/datasets/galileo-ai/ragbench) |

当前候选包共 100 条，中文/英文为 38/62，dev/final 为 50/50，关联 255 份公开
corpus 文档。它是待人工复核的候选原料，不是正式黄金集。

## 已评估但未纳入

### RGB（Retrieval-Augmented Generation Benchmark）

[官方仓库](https://github.com/chen700564/RGB)提供中文与英文的抗噪、拒答、信息整合和
反事实评测，任务形态很适合本项目。但其 README 明确说明代码和数据采用
CC BY-NC-SA 4.0，仅限非商业使用，商业使用需另行取得正式许可。因此它不进入默认
生产候选包或公开 CI 质量门禁。若以后获得单独授权，应作为新的、独立版本来源接入，
不能直接复制当前仓库的数据。

### 仅检索型语料

纯 qrels 检索基准可以补充 Recall/MRR/nDCG，但通常没有人工参考答案或可验证答案
要点，不能单独用于 Faithfulness、引用覆盖率或正确拒答率。此类数据若以后接入，应
标为 retrieval-only 套件，不能伪装成答案级黄金集。

## 在本仓库中的位置

- 待复核候选：`evaluation/candidates/public_golden_candidate.json`
- 候选公开语料：`evaluation/candidates/public_corpus/`
- 当前正式公开 seed：`evaluation/public_golden.json`（仍不可用于质量门禁）
- 私有集示例：`evaluation/private_golden.example.json`
- 私有正式集：`evaluation/private_golden.json`（只在本机生成，Git 忽略）

管理员需要在 Dashboard 的“黄金集复核”页逐条确认问题、答案、证据、类别/难度和
dev/final 划分。100 条全部批准后先导出 reviewed candidate，再通过
`scripts/promote_reviewed_golden.py` 独立提升。未经这一步，任何自动下载或转换结果都
不能称为“人工黄金集”。
