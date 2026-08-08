"""Agent Chat Dashboard Page — ChatGPT-style multi-session conversation history."""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CONV_FILE = _REPO_ROOT / "data" / "conversations.json"


# ------------------------------------------------------------------
# Conversation persistence
# ------------------------------------------------------------------

def _load_conversations() -> List[Dict[str, Any]]:
    try:
        if _CONV_FILE.exists():
            return json.loads(_CONV_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_conversations(convs: List[Dict[str, Any]]) -> None:
    try:
        _CONV_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONV_FILE.write_text(
            json.dumps(convs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _get_conv(convs: List[Dict[str, Any]], conv_id: str) -> Optional[Dict[str, Any]]:
    return next((c for c in convs if c["id"] == conv_id), None)


def _new_conversation() -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "title": "新对话",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

def _init_session() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = _load_conversations()
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None


def _current_messages() -> List[Dict[str, Any]]:
    conv = _get_conv(st.session_state.conversations, st.session_state.current_conv_id or "")
    return conv["messages"] if conv else []


def _add_message(role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    conv = _get_conv(st.session_state.conversations, st.session_state.current_conv_id or "")
    if conv is None:
        return
    msg = {"role": role, "content": content, "meta": meta or {}}
    conv["messages"].append(msg)
    # Update title from first user message
    if role == "user" and conv["title"] == "新对话":
        conv["title"] = content[:30] + ("…" if len(content) > 30 else "")
    _save_conversations(st.session_state.conversations)


def _start_new_chat() -> None:
    conv = _new_conversation()
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_conv_id = conv["id"]
    _save_conversations(st.session_state.conversations)


def _delete_conversation(conv_id: str) -> None:
    st.session_state.conversations = [
        c for c in st.session_state.conversations if c["id"] != conv_id
    ]
    if st.session_state.current_conv_id == conv_id:
        st.session_state.current_conv_id = None
    _save_conversations(st.session_state.conversations)


# ------------------------------------------------------------------
# Agent builder (cached)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="正在初始化 Agent…")
def _build_agent(collection: str, top_k: int):
    from src.core.settings import load_settings
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.llm.llm_factory import LLMFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.agent.react_agent import ReActAgent
    from src.agent.tool_registry import ToolRegistry
    from src.agent.tools.hybrid_search_tool import HybridSearchTool
    from src.agent.tools.semantic_search_tool import SemanticSearchTool
    from src.agent.tools.keyword_search_tool import KeywordSearchTool
    from src.agent.tools.document_summary_tool import DocumentSummaryTool
    from src.agent.tools.list_documents_tool import ListDocumentsTool

    settings = load_settings()
    coll = collection or getattr(getattr(settings, "agent", None), "default_collection", "default")

    vector_store = VectorStoreFactory.create(settings, collection_name=coll)
    embedding_client = EmbeddingFactory.create(settings)
    llm = LLMFactory.create(settings)
    dense_retriever = create_dense_retriever(settings=settings, embedding_client=embedding_client, vector_store=vector_store)
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{coll}")
    sparse_retriever = create_sparse_retriever(settings=settings, bm25_indexer=bm25_indexer, vector_store=vector_store)
    sparse_retriever.default_collection = coll
    hybrid_search = create_hybrid_search(settings=settings, query_processor=QueryProcessor(), dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)

    context_sink: List[Any] = []
    registry = ToolRegistry()
    registry.register(name=HybridSearchTool.name, fn=HybridSearchTool(hybrid_search, default_top_k=top_k, default_collection=coll, context_sink=context_sink).run, description=HybridSearchTool.description, params_schema=HybridSearchTool.parameters_schema)
    registry.register(name=SemanticSearchTool.name, fn=SemanticSearchTool(dense_retriever, default_top_k=top_k, default_collection=coll).run, description=SemanticSearchTool.description, params_schema=SemanticSearchTool.parameters_schema)
    registry.register(name=KeywordSearchTool.name, fn=KeywordSearchTool(sparse_retriever, default_top_k=top_k, default_collection=coll).run, description=KeywordSearchTool.description, params_schema=KeywordSearchTool.parameters_schema)
    registry.register(name=DocumentSummaryTool.name, fn=DocumentSummaryTool(vector_store, default_collection=coll).run, description=DocumentSummaryTool.description, params_schema=DocumentSummaryTool.parameters_schema)
    registry.register(name=ListDocumentsTool.name, fn=ListDocumentsTool(vector_store, default_collection=coll).run, description=ListDocumentsTool.description, params_schema=ListDocumentsTool.parameters_schema)

    return ReActAgent(settings=settings, llm=llm, tool_registry=registry), context_sink


# ------------------------------------------------------------------
# Sidebar: conversation list
# ------------------------------------------------------------------

def _group_conversations(convs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    groups: Dict[str, List] = {"今天": [], "昨天": [], "过去 7 天": [], "更早": []}
    for conv in convs:
        try:
            d = datetime.fromisoformat(conv["created_at"]).date()
        except Exception:
            d = date.min
        if d == today:
            groups["今天"].append(conv)
        elif d == yesterday:
            groups["昨天"].append(conv)
        elif d >= week_ago:
            groups["过去 7 天"].append(conv)
        else:
            groups["更早"].append(conv)
    return groups


def _render_sidebar(collection_ref: list, top_k_ref: list) -> None:
    with st.sidebar:
        # New chat button
        if st.button("＋ 新对话", type="secondary", use_container_width=True):
            _start_new_chat()
            st.rerun()

        st.divider()

        # Agent settings
        with st.expander("⚙️ Agent 设置", expanded=False):
            collection_ref[0] = st.text_input("知识库集合", value="default", key="agent_collection")
            top_k_ref[0] = st.slider("每次检索条数", min_value=1, max_value=20, value=5, key="agent_top_k")

        st.divider()

        # Conversation history
        convs = st.session_state.conversations
        if not convs:
            st.caption("暂无历史对话")
        else:
            groups = _group_conversations(convs)
            for group_name, group_convs in groups.items():
                if not group_convs:
                    continue
                st.caption(group_name)
                for conv in group_convs:
                    is_active = conv["id"] == st.session_state.current_conv_id
                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        label = ("**" + conv["title"] + "**") if is_active else conv["title"]
                        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                            st.session_state.current_conv_id = conv["id"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{conv['id']}"):
                            _delete_conversation(conv["id"])
                            st.rerun()


# ------------------------------------------------------------------
# ReAct trace timeline renderer
# ------------------------------------------------------------------

def _render_react_trace(turns, *, from_dict: bool = False, in_progress: bool = False) -> None:
    """Render ReAct turns as a vertical timeline: Turn → Thought / Action / Observation.

    in_progress=True makes the last turn show a pulsing indicator instead of a
    green dot, for use during live streaming.
    """
    import html as _html

    rows: List[str] = [
        "<div style='border-left:2px solid #D1D5DB;margin-left:10px;padding:2px 0 4px;'>"
    ]

    for i, raw in enumerate(turns, 1):
        if from_dict:
            thought   = raw.get("thought") or ""
            act_dict  = raw.get("action") or {}
            tool_name = act_dict.get("tool_name", "")
            tool_inp  = act_dict.get("tool_input") or {}
            obs       = raw.get("observation") or ""
        else:
            thought   = raw.thought or ""
            act_obj   = raw.action
            tool_name = act_obj.tool_name  if act_obj else ""
            tool_inp  = act_obj.tool_input if act_obj else {}
            obs       = raw.observation or ""

        is_last = (i == len(turns))
        if in_progress and is_last:
            dot_color = "#F59E0B"  # amber = in-progress
            dot_extra = "animation:pulse 1s infinite;"
        else:
            dot_color = "#10B981" if is_last else "#9CA3AF"
            dot_extra = ""

        rows += [
            "<div style='position:relative;padding-left:22px;margin-bottom:18px;'>",
            f"<div style='position:absolute;left:-7px;top:5px;width:12px;height:12px;"
            f"border-radius:50%;background:{dot_color};border:2px solid white;"
            f"box-shadow:0 0 0 1px {dot_color};{dot_extra}'></div>",
            f"<div style='font-weight:700;color:#374151;font-size:12px;"
            f"letter-spacing:.6px;margin-bottom:7px;'>Turn {i}"
            + (" &nbsp;<span style='color:#F59E0B;font-size:11px;'>● 推理中…</span>" if (in_progress and is_last) else "")
            + "</div>",
        ]

        if thought:
            rows.append(
                "<div style='background:#EFF6FF;border-radius:6px;padding:8px 12px;"
                "margin-bottom:6px;border-left:3px solid #3B82F6;'>"
                "<div style='color:#3B82F6;font-weight:700;font-size:10px;"
                "letter-spacing:1px;'>💭 THOUGHT</div>"
                f"<div style='color:#1E3A8A;margin-top:4px;font-size:13px;"
                f"line-height:1.5;'>{_html.escape(thought)}</div>"
                "</div>"
            )

        if tool_name:
            inp_json = _html.escape(json.dumps(tool_inp, ensure_ascii=False, indent=2))
            action_suffix = (
                " &nbsp;<span style='color:#F59E0B;font-size:10px;'>● 工具执行中…</span>"
                if (in_progress and is_last and not obs)
                else ""
            )
            rows.append(
                "<div style='background:#F5F3FF;border-radius:6px;padding:8px 12px;"
                "margin-bottom:6px;border-left:3px solid #8B5CF6;'>"
                "<div style='color:#8B5CF6;font-weight:700;font-size:10px;"
                f"letter-spacing:1px;'>⚡ ACTION &nbsp;"
                f"<code style='background:#DDD6FE;color:#5B21B6;padding:1px 8px;"
                f"border-radius:4px;font-size:11px;font-weight:600;'>"
                f"{_html.escape(tool_name)}</code>{action_suffix}</div>"
                f"<pre style='background:#1E1B4B;color:#C4B5FD;border-radius:5px;"
                f"padding:8px 10px;margin-top:6px;font-size:11px;"
                f"overflow-x:auto;white-space:pre-wrap;word-break:break-all;"
                f"margin-bottom:0;'>{inp_json}</pre>"
                "</div>"
            )

        if obs:
            obs_text = obs[:600] + ("…" if len(obs) > 600 else "")
            rows.append(
                "<div style='background:#F0FDF4;border-radius:6px;padding:8px 12px;"
                "border-left:3px solid #10B981;'>"
                "<div style='color:#059669;font-weight:700;font-size:10px;"
                "letter-spacing:1px;'>👁 OBSERVATION</div>"
                f"<div style='color:#064E3B;margin-top:4px;font-size:12px;"
                f"line-height:1.5;white-space:pre-wrap;font-family:monospace;'>"
                f"{_html.escape(obs_text)}</div>"
                "</div>"
            )

        rows.append("</div>")  # turn block

    rows.append("</div>")  # timeline
    st.markdown("\n".join(rows), unsafe_allow_html=True)


# ------------------------------------------------------------------
# Chat rendering
# ------------------------------------------------------------------

def _render_messages(messages: List[Dict[str, Any]]) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta", {})
            turns = meta.get("turns", [])
            if turns:
                conf   = meta.get("confidence", 0)
                badge  = "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
                label  = (
                    f"{badge} ReAct 推理过程 · {len(turns)} 轮"
                    f" · 置信度 {conf:.0%}"
                    f" · {meta.get('elapsed_ms', 0):.0f} ms"
                )
                with st.expander(label, expanded=False):
                    _render_react_trace(turns, from_dict=True)
            citations = meta.get("citations", [])
            if citations:
                with st.expander("参考来源"):
                    for c in citations:
                        st.markdown(f"- `{c}`")


# ------------------------------------------------------------------
# Main render
# ------------------------------------------------------------------

def render() -> None:
    _init_session()

    collection_ref = ["default"]
    top_k_ref = [5]
    _render_sidebar(collection_ref, top_k_ref)

    collection = collection_ref[0]
    top_k = top_k_ref[0]

    # ── Empty state: no conversation selected ──────────────────────
    if st.session_state.current_conv_id is None:
        st.markdown(
            "<div style='text-align:center;margin-top:15vh;'>"
            "<h2 style='color:#6B7280;'>🤖 Agent 对话</h2>"
            "<p style='color:#9CA3AF;'>基于混合检索的 ReAct Agent</p>"
            "<p style='color:#9CA3AF;'>点击左侧「＋ 新对话」开始提问</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Active conversation ────────────────────────────────────────
    conv = _get_conv(st.session_state.conversations, st.session_state.current_conv_id)
    if conv is None:
        st.session_state.current_conv_id = None
        st.rerun()
        return

    st.title(f"🤖 {conv['title']}")

    # Build agent
    agent_error = st.empty()
    try:
        agent, _ctx = _build_agent(collection, top_k)
    except Exception as exc:
        agent_error.error(f"Agent 初始化失败: {exc}")
        st.stop()

    # Render existing messages
    _render_messages(conv["messages"])

    # Chat input
    if prompt := st.chat_input("向知识库提问…"):
        _add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Three ordered placeholders: live trace → answer → status
            # Empty placeholders take no visible space, so during streaming the
            # live trace appears at the top of the bubble.
            live_ph   = st.empty()
            answer_ph = st.empty()
            status_ph = st.empty()

            live_turns: List[Dict[str, Any]] = []
            done_event = None

            try:
                for event in agent.run_stream(prompt):
                    if event.type == "thought":
                        live_turns.append({
                            "thought": event.thought,
                            "action": {},
                            "observation": "",
                        })
                        with live_ph.container():
                            _render_react_trace(
                                live_turns, from_dict=True, in_progress=True
                            )

                    elif event.type == "action":
                        if live_turns:
                            live_turns[-1]["action"] = {
                                "tool_name": event.tool_name,
                                "tool_input": event.tool_input,
                            }
                        with live_ph.container():
                            _render_react_trace(
                                live_turns, from_dict=True, in_progress=True
                            )

                    elif event.type == "observation":
                        if live_turns:
                            live_turns[-1]["observation"] = event.observation
                        with live_ph.container():
                            _render_react_trace(
                                live_turns, from_dict=True, in_progress=False
                            )

                    elif event.type == "done":
                        done_event = event

                if done_event:
                    # Show final answer
                    answer_ph.markdown(done_event.answer)

                    conf = done_event.confidence
                    badge_colour = (
                        "green" if conf >= 0.7 else ("orange" if conf >= 0.4 else "red")
                    )
                    status_ph.markdown(
                        f":{badge_colour}[置信度 {conf:.0%}]"
                        f"  ·  {len(done_event.turns_data)} 轮推理"
                        f"  ·  {done_event.elapsed_ms:.0f} ms"
                    )

                    # Replace live trace with collapsed expander
                    live_ph.empty()
                    if done_event.turns_data:
                        badge = (
                            "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
                        )
                        label = (
                            f"{badge} ReAct 推理过程 · {len(done_event.turns_data)} 轮"
                            f" · 置信度 {conf:.0%}"
                            f" · {done_event.elapsed_ms:.0f} ms"
                        )
                        with live_ph.container():
                            with st.expander(label, expanded=False):
                                _render_react_trace(
                                    done_event.turns_data, from_dict=True
                                )

                    _add_message(
                        "assistant",
                        done_event.answer,
                        meta={
                            "turns": done_event.turns_data,
                            "citations": done_event.citations,
                            "confidence": done_event.confidence,
                            "elapsed_ms": done_event.elapsed_ms,
                            "used_tools": done_event.used_tools,
                        },
                    )
                    st.rerun()

            except Exception as exc:
                error_msg = f"Agent 错误: {exc}"
                live_ph.empty()
                answer_ph.error(error_msg)
                _add_message("assistant", error_msg)
