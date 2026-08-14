"""Production Streamlit client; all business operations go through FastAPI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import streamlit as st

from src.observability.dashboard.api_client import DashboardAPIClient, DashboardAPIError


def _client() -> DashboardAPIClient:
    return DashboardAPIClient(access_token=st.session_state.get("access_token"))


def _auth_screen() -> None:
    st.title("Modular RAG")
    st.caption("局域网知识检索、证据约束问答与可观测平台")
    try:
        ready = DashboardAPIClient().get("/health/ready", accepted_statuses={503})
    except DashboardAPIError as exc:
        st.error(str(exc))
        st.info("请先在服务器运行 rag-launcher 或 python -m src.production.api。")
        return
    if ready.get("status") != "ready":
        warmup = ready.get("warmup") or {}
        reasons: list[str] = []
        if warmup.get("reranker_configured") and not warmup.get("reranker_ready"):
            reasons.append(f"重排序器不可用：{warmup.get('reranker_error') or 'unknown'}")
        if warmup.get("evidence_calibration") != "calibrated":
            reasons.append("EvidenceGate 阈值尚未通过 dev split 校准")
        detail = "；".join(reasons)
        st.warning("服务已启动，但尚未达到生产就绪状态。" + (f" {detail}" if detail else ""))
    bootstrap = bool(ready.get("bootstrap_required"))
    st.subheader("首次创建管理员" if bootstrap else "登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if bootstrap:
        confirm = st.text_input("确认密码", type="password")
        submitted = st.button("创建管理员", type="primary")
        if submitted and password != confirm:
            st.error("两次密码不一致")
            return
        path = "/auth/bootstrap"
    else:
        submitted = st.button("登录", type="primary")
        path = "/auth/login"
    if not submitted:
        return
    try:
        tokens = DashboardAPIClient().post(path, json={"username": username, "password": password})
        st.session_state.access_token = tokens["access_token"]
        st.session_state.refresh_token = tokens["refresh_token"]
        st.session_state.user = tokens["user"]
        st.rerun()
    except DashboardAPIError as exc:
        st.error(str(exc))


def _logout() -> None:
    refresh = st.session_state.get("refresh_token")
    if refresh:
        try:
            _client().post("/auth/logout", json={"refresh_token": refresh})
        except DashboardAPIError:
            pass
    for key in ("access_token", "refresh_token", "user", "messages"):
        st.session_state.pop(key, None)


def _chat() -> None:
    st.header("证据约束问答")
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("引用"):
                    st.json(message["citations"])
    query = st.chat_input("询问知识库；证据不足时系统会拒答")
    if not query:
        return
    messages.append({"role": "user", "content": query})
    try:
        with st.spinner("检索、融合、重排并核验证据…"):
            result = _client().post("/answer", json={
                "query": query, "collection": "default", "top_k": 5,
            })
    except DashboardAPIError as exc:
        result = {"answer": str(exc), "status": "error", "citations": []}
    prefix = {"refused": "⚠️ ", "degraded": "🟡 ", "error": "❌ "}.get(result["status"], "")
    messages.append({"role": "assistant", "content": prefix + result["answer"],
                     "citations": result.get("citations"), "trace_id": result.get("trace_id")})
    st.rerun()


def _sources() -> None:
    st.header("数据源与导入")
    source_tab, upload_tab, jobs_tab = st.tabs(["只读目录 / 网页", "上传文档", "任务"])
    with source_tab:
        with st.form("new-source"):
            kind = st.selectbox("类型", ["directory", "web"])
            default = r"D:\AI-KnowledgeBase\documents" if kind == "directory" else "https://"
            location = st.text_input("目录或 URL", value=default)
            name = st.text_input("名称", value="本地知识库" if kind == "directory" else "网页数据源")
            collection = st.text_input("Collection", value="default")
            submitted = st.form_submit_button("添加数据源")
        if submitted:
            try:
                _client().post("/data-sources", json={"name": name, "kind": kind,
                    "location": location, "collection_name": collection, "config": {}})
                st.success("已添加")
            except DashboardAPIError as exc:
                st.error(str(exc))
        try:
            for source in _client().get("/data-sources"):
                with st.expander(f"{source['name']} · {source['kind']} · {source['location']}"):
                    cols = st.columns(3)
                    enabled = cols[0].toggle(
                        "启用", value=bool(source["enabled"]), key=f"enabled-{source['id']}",
                    )
                    name_value = cols[1].text_input(
                        "数据源名称", value=source["name"], key=f"source-name-{source['id']}",
                    )
                    collection_value = cols[2].text_input(
                        "目标 Collection", value=source["collection_name"],
                        key=f"source-collection-{source['id']}",
                    )
                    actions = st.columns(3)
                    if actions[0].button("保存", key=f"save-{source['id']}"):
                        _client().patch(f"/data-sources/{source['id']}", json={
                            "name": name_value, "collection_name": collection_value,
                            "enabled": enabled,
                        })
                        st.success("数据源已更新")
                    if actions[1].button("开始增量同步", key=f"sync-{source['id']}"):
                        st.json(_client().post(f"/data-sources/{source['id']}/sync"))
                    confirm = actions[2].checkbox("确认删除", key=f"confirm-delete-{source['id']}")
                    if st.button(
                        "删除数据源", key=f"delete-{source['id']}", disabled=not confirm,
                    ):
                        result = _client().delete(f"/data-sources/{source['id']}")
                        st.success(f"安全删除任务 {result['job_id']} 已排队")
                        st.rerun()
                    with st.expander("配置详情"):
                        st.json(source)
        except DashboardAPIError as exc:
            st.error(str(exc))
    with upload_tab:
        uploaded = st.file_uploader("PDF / Markdown / TXT", type=["pdf", "md", "markdown", "txt"])
        if uploaded and st.button("上传并排队"):
            try:
                result = _client().post("/documents/upload?collection=default", files={
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")})
                st.success(f"任务 {result['job_id']} 已排队")
            except DashboardAPIError as exc:
                st.error(str(exc))
    with jobs_tab:
        job_id = st.text_input("任务 ID")
        job_actions = st.columns(2)
        if job_id and job_actions[0].button("刷新任务"):
            try:
                st.json(_client().get(f"/jobs/{job_id}"))
            except DashboardAPIError as exc:
                st.error(str(exc))
        if job_id and job_actions[1].button("取消任务"):
            try:
                st.json(_client().post(f"/jobs/{job_id}/cancel"))
            except DashboardAPIError as exc:
                st.error(str(exc))


def _users() -> None:
    st.header("用户管理")
    with st.form("create-user"):
        cols = st.columns(3)
        username = cols[0].text_input("新用户名")
        password = cols[1].text_input("初始密码", type="password")
        role = cols[2].selectbox("角色", ["user", "admin"])
        create = st.form_submit_button("创建用户", type="primary")
    if create:
        try:
            _client().post("/users", json={
                "username": username, "password": password, "role": role,
            })
            st.success("用户已创建")
        except DashboardAPIError as exc:
            st.error(str(exc))
    try:
        users = _client().get("/users")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    for account in users:
        with st.expander(f"{account['username']} · {account['role']}"):
            cols = st.columns(3)
            selected_role = cols[0].selectbox(
                "账户角色", ["user", "admin"],
                index=0 if account["role"] == "user" else 1,
                key=f"user-role-{account['id']}",
            )
            active = cols[1].toggle(
                "账户启用", value=bool(account["active"]), key=f"user-active-{account['id']}",
            )
            replacement = cols[2].text_input(
                "重置密码（留空不改）", type="password", key=f"user-password-{account['id']}",
            )
            if st.button("保存账户", key=f"save-user-{account['id']}"):
                payload: dict[str, object] = {"role": selected_role, "active": active}
                if replacement:
                    payload["password"] = replacement
                try:
                    _client().patch(f"/users/{account['id']}", json=payload)
                    st.success("账户已更新；停用或改密会撤销现有刷新会话")
                except DashboardAPIError as exc:
                    st.error(str(exc))


def _traces() -> None:
    st.header("Trace 瀑布")
    with st.expander("筛选", expanded=True):
        cols = st.columns(2)
        start_date = cols[0].date_input(
            "Trace 开始日期", value=(datetime.now(timezone.utc) - timedelta(days=7)).date(),
        )
        end_date = cols[1].date_input("Trace 结束日期", value=datetime.now(timezone.utc).date())
        prompt_version = st.text_input("Prompt 版本", placeholder="例如 1.0.0")
        model_version = st.text_input("模型版本", placeholder="例如 qwen3:8b")
        dataset_version = st.text_input("数据集 / 索引版本", placeholder="例如 v1")
        config_version = st.text_input("配置版本哈希")
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    filters = {
        "limit": 200, "start": start.isoformat(), "end": end.isoformat(),
        "prompt_version": prompt_version or None, "model_version": model_version or None,
        "dataset_version": dataset_version or None, "config_version": config_version or None,
    }
    query = urlencode({key: value for key, value in filters.items() if value is not None})
    try:
        traces = _client().get(f"/traces?{query}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    options = {f"{item['started_at']} · {item['status']} · {item['id']}": item["id"] for item in traces}
    if not options:
        st.info("暂无 Trace")
        return
    selected = st.selectbox("选择请求", list(options))
    detail = _client().get(f"/traces/{options[selected]}")
    header = st.columns(4)
    header[0].metric("状态", detail["status"])
    header[1].metric("总耗时", f"{detail['total_elapsed_ms']:.1f} ms")
    header[2].metric("输入 Token", detail.get("input_tokens", 0))
    header[3].metric("输出 Token", detail.get("output_tokens", 0))
    st.caption(
        f"Prompt={detail.get('prompt_version') or 'N/A'} · "
        f"Model={detail.get('model_version') or 'N/A'} · "
        f"Dataset={detail.get('dataset_version') or 'N/A'} · "
        f"Config={detail.get('config_version') or 'N/A'}"
    )
    stages = detail["stages"]
    elapsed_rows = [{
        "阶段": f"{index + 1}. {stage['stage']}",
        "耗时(ms)": float(stage.get("elapsed_ms") or 0),
    } for index, stage in enumerate(stages)]
    st.subheader("阶段耗时瀑布")
    if any(row["耗时(ms)"] for row in elapsed_rows):
        st.bar_chart(elapsed_rows, x="阶段", y="耗时(ms)", horizontal=True)
    else:
        st.info("此 Trace 没有阶段耗时数据")

    rerank_stage = next((stage for stage in stages if stage["stage"] == "rerank"), None)
    if rerank_stage:
        rerank_data = rerank_stage.get("data") or {}
        before = _rank_rows(rerank_data.get("input_order", []))
        after = _rank_rows(rerank_data.get("output_order", []))
        st.subheader("Rerank 前后顺序")
        left, right = st.columns(2)
        left.caption("重排前")
        left.dataframe(before, width="stretch", hide_index=True)
        right.caption("重排后")
        right.dataframe(after, width="stretch", hide_index=True)
        if rerank_data.get("used_fallback"):
            st.warning(f"Rerank 已降级：{rerank_data.get('fallback_reason') or 'unknown'}")

    st.subheader("阶段详情")
    for stage in stages:
        with st.expander(f"{stage['stage']} · {stage.get('elapsed_ms') or 0:.1f} ms"):
            st.json(stage["data"])
    with st.expander("Trace 元数据"):
        st.json({key: value for key, value in detail.items() if key != "stages"})


def _rank_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not isinstance(value, list):
        return rows
    for index, item in enumerate(value, 1):
        if isinstance(item, dict):
            rows.append({
                "rank": item.get("rank", index),
                "chunk_id": item.get("chunk_id") or item.get("id") or "unknown",
                "score": item.get("score"),
            })
        else:
            rows.append({"rank": index, "chunk_id": str(item), "score": None})
    return rows


def _observability() -> None:
    st.header("指标与根因分析")
    now = datetime.now(timezone.utc)
    start_date = st.date_input("开始日期", value=(now - timedelta(days=7)).date())
    end_date = st.date_input("结束日期", value=now.date())
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    query = urlencode({"start": start.isoformat(), "end": end.isoformat()})
    try:
        metrics = _client().get(f"/metrics/summary?{query}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    cols = st.columns(5)
    cols[0].metric("请求", metrics["request_count"])
    cols[1].metric("P50", f"{metrics['latency_ms']['p50']:.0f} ms")
    cols[2].metric("P95", f"{metrics['latency_ms']['p95']:.0f} ms")
    cols[3].metric("失败率", f"{metrics['failure_rate']:.1%}")
    coverage = metrics["citation_coverage"]
    cols[4].metric("引用覆盖", "N/A" if coverage is None else f"{coverage:.1%}")
    st.subheader("阶段 P50 / P95")
    st.json(metrics["stage_latency_ms"])
    st.subheader("状态与错误")
    st.json({"status": metrics["status_counts"], "errors": metrics["error_counts"]})
    duration = end - start
    bucket = "hour" if duration <= timedelta(days=7) else "day"
    try:
        trend = _client().get(f"/metrics/timeseries?{query}&bucket={bucket}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        trend = []
    st.subheader("质量与可靠性趋势")
    if trend:
        rate_rows = [{
            "时间": item["bucket"],
            "成功率": item["success_rate"],
            "失败率": item["failure_rate"],
            "拒答率": item["refusal_rate"],
            "降级率": item["degraded_rate"],
        } for item in trend]
        st.line_chart(rate_rows, x="时间", y=["成功率", "失败率", "拒答率", "降级率"])
        latency_rows = [{
            "时间": item["bucket"], "P50(ms)": item["latency_p50_ms"],
            "P95(ms)": item["latency_p95_ms"],
        } for item in trend]
        st.line_chart(latency_rows, x="时间", y=["P50(ms)", "P95(ms)"])
        quality_rows = [{
            "时间": item["bucket"], "引用覆盖率": item.get("citation_coverage"),
            "忠实度": item.get("faithfulness"),
        } for item in trend if item.get("citation_coverage") is not None or item.get("faithfulness") is not None]
        if quality_rows:
            st.line_chart(quality_rows, x="时间", y=["引用覆盖率", "忠实度"])
    else:
        st.info("所选时间段暂无趋势数据")
    compare = urlencode({"start": start.isoformat(), "end": end.isoformat(),
        "baseline_start": (start - duration).isoformat(), "baseline_end": start.isoformat()})
    st.subheader("与前一等长窗口对比")
    comparison = _client().get(f"/metrics/compare?{compare}")
    st.json(comparison["delta"])
    st.subheader("根因候选")
    st.caption("以下是时间相关信号，不代表已证明因果；请结合单次 Trace 与部署记录确认。")
    candidates = comparison.get("root_cause_candidates", [])
    if candidates:
        st.dataframe(candidates, width="stretch", hide_index=True)
    else:
        st.info("未发现明显的指标退化、错误增长、版本切换或窗口内部署事件")
    st.subheader("关联版本与部署事件")
    st.json({"versions": comparison.get("versions", {}),
             "deployment_events": comparison.get("deployment_events", [])})


def _evaluations() -> None:
    st.header("离线评测")
    st.caption("管理员从版本化黄金集启动后台消融评测；所有检索和报告均通过 FastAPI/Worker。")
    with st.form("evaluation-run"):
        dataset_path = st.text_input("黄金集", value="evaluation/public_golden.json")
        split = st.selectbox("数据划分", ["dev", "final"], index=0)
        collection = st.text_input("Collection", value="evaluation")
        top_k = st.number_input("Top-K", min_value=1, max_value=50, value=5)
        repeats = st.number_input("重复次数", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("启动评测", type="primary")
    if submitted:
        try:
            created = _client().post("/evaluations", json={
                "dataset_path": dataset_path,
                "split": split,
                "collection": collection,
                "top_k": int(top_k),
                "repeats": int(repeats),
            })
            st.session_state.last_evaluation_id = created["evaluation_id"]
            st.session_state.last_evaluation_job_id = created["job_id"]
            st.success(f"评测任务已排队：{created['job_id']}")
        except DashboardAPIError as exc:
            st.error(str(exc))

    try:
        history = _client().get("/evaluations?limit=50")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    if not history:
        st.info("暂无评测记录")
        return
    labels = {
        f"{item['created_at']} · {item['status']} · {item['dataset_name']} · {item['split']}": item["id"]
        for item in history
    }
    selected = st.selectbox("评测历史", list(labels))
    if st.button("刷新评测状态"):
        st.rerun()
    try:
        detail = _client().get(f"/evaluations/{labels[selected]}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    cols = st.columns(4)
    cols[0].metric("状态", detail["status"])
    cols[1].metric("数据集版本", detail["dataset_version"])
    cols[2].metric("划分", detail["split"])
    cols[3].metric("查询数", (detail.get("report") or {}).get("dataset", {}).get("query_count", "N/A"))
    metrics = detail.get("metrics") or {}
    if metrics:
        st.subheader("消融指标")
        rows = [{"variant": name, **values} for name, values in metrics.items()]
        st.dataframe(rows, use_container_width=True)
    report = detail.get("report") or {}
    if report.get("unavailable_variants"):
        st.warning(f"未运行方案：{report['unavailable_variants']}")
    answer_quality = report.get("answer_quality") or {}
    if answer_quality.get("unavailable_metrics"):
        st.warning(f"未产生答案指标：{answer_quality['unavailable_metrics']}")
    if answer_quality.get("judge_errors"):
        with st.expander("答案 Judge 错误"):
            st.json(answer_quality["judge_errors"])
    if report.get("variants"):
        with st.expander("逐查询失败分析"):
            for variant in report["variants"]:
                st.markdown(f"**{variant['variant']}**")
                st.json(variant.get("failures", []))


def _golden_review() -> None:
    st.header("黄金集候选复核")
    st.caption("候选必须逐条核对问题、答案要点和支持文档；自动生成内容不能直接进入质量门禁。")
    try:
        summary = _client().get("/golden-candidates/summary")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    counts = summary["counts"]
    cols = st.columns(5)
    cols[0].metric("总数", summary["total"])
    cols[1].metric("未复核", counts["unreviewed"])
    cols[2].metric("已批准", counts["approved"])
    cols[3].metric("已退回", counts["rejected"])
    cols[4].metric("待定", counts["pending"])
    st.progress(counts["approved"] / summary["total"] if summary["total"] else 0.0)
    st.caption(f"候选版本 {summary['version']} · SHA-256 {summary['dataset_sha256']}")
    if summary["all_approved"]:
        st.success("全部候选均已批准；可导出已复核产物，再通过受审查的独立变更提升为正式黄金集。")
        if st.button("导出已复核产物", type="primary", key="export-reviewed-golden"):
            try:
                exported = _client().post("/golden-candidates/export-reviewed", json={
                    "dataset_sha256": summary["dataset_sha256"],
                })
                st.success(f"已导出：{exported['path']}（仍不可用于质量门禁）")
            except DashboardAPIError as exc:
                st.error(str(exc))
    else:
        st.warning("当前候选不可用于质量门禁。批准进度不会自动修改正式黄金集。")

    status = st.selectbox("复核状态", ["unreviewed", "pending", "rejected", "approved"])
    try:
        cases = _client().get(f"/golden-candidates/cases?status={status}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    if not cases:
        st.info("该状态下没有候选")
        return
    labels = {
        f"{case['query_id']} · {case['language']} · {case['category']} · {case['split']}": case["query_id"]
        for case in cases
    }
    selected = st.selectbox("候选问题", list(labels))
    try:
        detail = _client().get(f"/golden-candidates/cases/{labels[selected]}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    case = detail["case"]
    st.subheader(case["query"])
    st.json({
        "query_id": case["query_id"], "answerable": case["answerable"],
        "category": case["category"], "difficulty": case["difficulty"],
        "language": case["language"], "split": case["split"],
        "answer_key_points": case["answer_key_points"],
        "expected_document_ids": case["expected_document_ids"],
        "upstream": case.get("candidate_metadata", {}),
    })
    st.subheader("支持证据")
    if detail["evidence"]:
        for evidence in detail["evidence"]:
            with st.expander(f"{evidence['document_id']} · {evidence.get('title', '')}", expanded=True):
                st.code(evidence["content"], language="markdown")
    else:
        st.info("无答案候选没有期望支持文档；请确认问题确实无法由候选 corpus 回答。")
    prior = detail.get("review") or {}
    notes = st.text_area("复核备注", value=prior.get("notes", ""), key=f"review-notes-{case['query_id']}")
    actions = st.columns(3)
    selected_status = None
    if actions[0].button("批准", type="primary", key=f"approve-{case['query_id']}"):
        selected_status = "approved"
    if actions[1].button("退回", key=f"reject-{case['query_id']}"):
        selected_status = "rejected"
    if actions[2].button("标记待定", key=f"pending-{case['query_id']}"):
        selected_status = "pending"
    if selected_status:
        try:
            _client().put(f"/golden-candidates/cases/{case['query_id']}/review", json={
                "dataset_sha256": summary["dataset_sha256"],
                "status": selected_status, "notes": notes,
            })
            st.success("复核状态已保存")
            st.rerun()
        except DashboardAPIError as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="Modular RAG", page_icon="🔎", layout="wide")
    if not st.session_state.get("access_token"):
        _auth_screen()
        return
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{user['username']}** · `{user['role']}`")
        pages = ["问答", "我的 Trace"]
        if user["role"] == "admin":
            pages.extend(["数据源", "用户管理", "黄金集复核", "离线评测", "可观测性"])
        page = st.radio("导航", pages)
        if st.button("退出"):
            _logout()
            st.rerun()
    {"问答": _chat, "我的 Trace": _traces, "数据源": _sources, "用户管理": _users,
     "黄金集复核": _golden_review,
     "离线评测": _evaluations, "可观测性": _observability}[page]()


main()
