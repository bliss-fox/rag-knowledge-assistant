"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their
respective modules under ``pages/``.  Pages not yet implemented show
a placeholder message.
"""

from __future__ import annotations

import streamlit as st


# ── Page definitions ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


def _page_agent_chat() -> None:
    from src.observability.dashboard.pages.agent_chat import render
    render()


# ── Navigation ───────────────────────────────────────────────────────

pages = [
    st.Page(_page_overview, title="系统概览", icon="📊", default=True),
    st.Page(_page_data_browser, title="知识库浏览", icon="🔍"),
    st.Page(_page_ingestion_manager, title="文档导入", icon="📥"),
    st.Page(_page_ingestion_traces, title="导入追踪", icon="🔬"),
    st.Page(_page_query_traces, title="查询追踪", icon="🔎"),
    st.Page(_page_evaluation_panel, title="评估面板", icon="📏"),
    st.Page(_page_agent_chat, title="Agent 对话", icon="🤖"),
]


_GLOBAL_CSS = """
<style>
/* ── Base & Typography ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Hide Streamlit default branding ──────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid rgba(0,0,0,0.07);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #6B7280;
    font-size: 0.85rem;
}

/* ── Main content area ────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main {
    background-color: #F7F7F8;
}
[data-testid="block-container"] {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Cards / Info boxes ───────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] { color: #6B7280; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #111827; font-size: 1.6rem; font-weight: 600; }

/* ── Expanders ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.07) !important;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Buttons ──────────────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #7C3AED;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    transition: background 0.2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #6D28D9;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 8px;
    color: #6B7280;
}

/* ── Inputs ───────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 8px;
    color: #111827;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #7C3AED;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.15);
}

/* ── Chat messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 12px;
    margin-bottom: 0.75rem;
    padding: 0.25rem 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #7C3AED;
}

/* ── Chat input ───────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7C3AED;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.15);
}

/* ── Dividers ─────────────────────────────────────────────────── */
hr { border-color: rgba(0,0,0,0.07) !important; }

/* ── Dataframes / Tables ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Alerts ───────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 3px;
}

/* ── Subheader accent line ────────────────────────────────────── */
h2 { border-bottom: 1px solid rgba(0,0,0,0.07); padding-bottom: 0.4rem; }

/* ── 新对话 button (sidebar top) ─────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"]:first-child > button {
    border: 1.5px solid #7C3AED !important;
    color: #7C3AED !important;
    background: #FFFFFF !important;
    font-weight: 600;
    border-radius: 8px;
}
[data-testid="stSidebar"] [data-testid="stButton"]:first-child > button:hover {
    background: rgba(124,58,237,0.06) !important;
}

/* ── Navigation items ─────────────────────────────────────────── */
[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    transition: background 0.15s;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(124,58,237,0.08);
}
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: rgba(124,58,237,0.12);
    border-left: 2px solid #7C3AED;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG 知识库系统",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py`
    main()
