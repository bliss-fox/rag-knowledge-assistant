# Agent Layer Architecture Plan

## 设计思路

现有项目是 RAG Server（"智能搜索引擎"），Agent 层将它升级为"智能研究助理"。

**核心模式：ReAct（Reasoning + Acting）**
- Agent 在推理和工具调用之间循环，直到可以给出最终答案
- 使用现有 RAG 组件作为工具（不重写底层）
- 自反思机制：答案置信度不足时自动触发二次检索

## 目标架构

```
用户问题
   ↓
┌─────────────────────────────────┐
│         ReAct Agent Loop         │
│  Thought → Action → Observation  │◄──┐
│         (最多N轮)                 │   │
└─────────────────────────────────┘   │ 置信度不足
   ↓ 足够 or 达到最大轮次              │
Final Answer + Citations             │
   ↓                                 │
SelfChecker ──────────────────────────┘
```

## 新增模块（阶段 J）

### J1: ReAct Agent 核心框架

**文件：** `src/agent/__init__.py`, `src/agent/react_agent.py`, `src/agent/tool_registry.py`, `src/agent/agent_state.py`

**核心类：**
```python
class AgentState:
    question: str
    history: List[Turn]      # Thought/Action/Observation 轮次
    final_answer: str | None
    retrieval_context: List[RetrievalResult]

@dataclass
class Turn:
    thought: str
    action: ToolCall | None  # None 表示直接回答
    observation: str | None

class ToolCall:
    tool_name: str           # "semantic_search" / "keyword_search" / ...
    tool_input: dict

class ToolRegistry:
    def register(name: str, fn: Callable, description: str, params_schema: dict)
    def dispatch(tool_name: str, tool_input: dict) -> str  # 返回 Observation 文本
    def get_tools_prompt() -> str  # 生成工具列表的 prompt 文本

class ReActAgent:
    def __init__(settings, llm, tool_registry, hybrid_search, max_turns=5)
    def run(question: str, trace?) -> AgentResponse

@dataclass
class AgentResponse:
    answer: str
    citations: List[str]
    turns: List[Turn]
    confidence: float        # 0~1
    used_tools: List[str]
```

**ReAct Prompt 模板（存 config/prompts/react_agent.txt）：**
```
You are a research assistant with access to a knowledge base.
Answer the question step by step using the available tools.

Available tools:
{tools_description}

Format:
Thought: <reasoning about what to do>
Action: <tool_name>
Action Input: <JSON input for the tool>
Observation: <tool result - provided by system>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <comprehensive answer with source references>

Question: {question}
{history}
```

### J2: 细粒度原子工具集

**文件：** `src/agent/tools/` 下各工具文件

| 工具名 | 实现 | 面试亮点 |
|--------|------|---------|
| `semantic_search` | 调用 DenseRetriever，纯向量检索 | 演示单独走稠密检索路径 |
| `keyword_search` | 调用 SparseRetriever，纯 BM25 | 演示单独走稀疏检索路径 |
| `hybrid_search` | 调用现有 HybridSearch | 两者融合，作为默认工具 |
| `get_document_summary` | 调用现有 MCP Tool | 文档级别预览 |
| `list_documents` | 调用现有 MCP Tool | 知识库探索 |

工具接口约定：
```python
class BaseTool:
    name: str
    description: str          # 写入 prompt，让 LLM 知道何时用
    parameters_schema: dict   # JSON Schema 格式

    def run(self, **kwargs) -> str:  # 返回自然语言格式的 Observation
        ...
```

### J3: 对话记忆

**文件：** `src/agent/memory/conversation_memory.py`

```python
class ConversationMemory:
    def add_turn(role: str, content: str)
    def get_context_messages() -> List[dict]  # 返回 LLM messages 格式
    def get_recent_queries() -> List[str]     # 用于查询改写
    def summarize_if_long(llm, max_turns=10)  # 防止 context 溢出
```

**查询改写（可选）：** 多轮对话中，"这个" "它" 等指代词需结合历史改写为完整查询。

### J4: 自反思与置信度校验

**文件：** `src/agent/reflection/self_checker.py`

```python
class SelfChecker:
    def check(question: str, answer: str, context: List[RetrievalResult], llm) -> CheckResult

@dataclass
class CheckResult:
    confidence: float          # 0~1
    is_grounded: bool          # 答案是否有 context 支撑
    missing_aspects: List[str] # 还缺哪些信息
    should_retry: bool         # 是否建议重新检索
```

Prompt 思路：让 LLM 评估"答案的每个声明是否能在给定 context 中找到依据"。

### J5: Agent 入口

**CLI（`scripts/agent.py`）：**
```bash
python scripts/agent.py          # 交互式多轮对话
python scripts/agent.py --query "什么是 BM25？"   # 单次查询
```

**Dashboard 新增页面（`src/observability/dashboard/pages/agent_chat.py`）：**
- 对话输入框
- 展示每轮 Thought/Action/Observation 折叠面板（可视化 ReAct 过程）
- 最终答案 + 引用来源
- 置信度 badge

## 新增文件树（Stage J）

```
src/agent/
├── __init__.py
├── react_agent.py           # ReActAgent + AgentResponse
├── tool_registry.py         # ToolRegistry
├── agent_state.py           # AgentState + Turn + ToolCall
├── tools/
│   ├── __init__.py
│   ├── base_tool.py
│   ├── semantic_search_tool.py
│   ├── keyword_search_tool.py
│   ├── hybrid_search_tool.py
│   ├── document_summary_tool.py
│   └── list_documents_tool.py
├── memory/
│   ├── __init__.py
│   └── conversation_memory.py
└── reflection/
    ├── __init__.py
    └── self_checker.py
config/prompts/
└── react_agent.txt          # ReAct system prompt 模板
scripts/
└── agent.py                 # CLI 入口
src/observability/dashboard/pages/
└── agent_chat.py            # Dashboard 聊天页
tests/
├── unit/test_react_agent.py
├── unit/test_agent_tools.py
├── unit/test_conversation_memory.py
├── unit/test_self_checker.py
└── e2e/test_agent_e2e.py
```

## 实现顺序

```
J1 (ReAct Core + ToolRegistry)
  → J2 (Atomic Tools)
    → J3 (ConversationMemory)
      → J4 (SelfChecker)
        → J5 (CLI + Dashboard page)
```

J1+J2+J5 完成后项目已可演示；J3+J4 是进阶亮点，面试中可重点讲。
