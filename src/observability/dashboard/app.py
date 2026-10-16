"""Production Streamlit client; all business operations go through FastAPI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
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
    username = st.text_input(
        "用户名",
        help="3–64 个字符；仅允许字母、数字、点、下划线和连字符。",
    )
    password = st.text_input(
        "密码", type="password",
        help="新建或重置密码时必须为 12–256 个字符。",
    )
    if bootstrap:
        confirm = st.text_input("确认密码", type="password")
        submitted = st.button("创建管理员", type="primary")
        if submitted and password != confirm:
            st.error("两次密码不一致")
            return
        if submitted and not 12 <= len(password) <= 256:
            st.error("密码必须为 12–256 个字符")
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
        password = cols[1].text_input(
            "初始密码", type="password", help="12–256 个字符",
        )
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
                "重置密码（留空不改）", type="password", help="12–256 个字符",
                key=f"user-password-{account['id']}",
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


def _review_attestation_inputs(prefix: str, prior: dict[str, Any]) -> dict[str, bool]:
    st.markdown("**结构化人工核对**")
    st.caption("五项必须全部确认后才能批准；待定或退回可以保存未完成的核对进度。")
    labels = (
        ("query_verified", "问题表述、可回答性与业务意图正确"),
        ("answer_verified", "答案要点正确；无答案题已确认应拒答"),
        ("evidence_verified", "预期文档逐条支持答案，且没有伪造引用"),
        ("category_difficulty_verified", "类别、语言和难度标注正确"),
        ("split_verified", "dev/final 分配正确，final 未参与调参"),
    )
    result: dict[str, bool] = {}
    for field, label in labels:
        result[field] = st.checkbox(
            label, value=bool(prior.get(field, False)), key=f"{prefix}-{field}",
        )
    return result


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
    if case.get("candidate_metadata", {}).get("answer_key_origin") == (
        "model_response_requires_human_review"
    ):
        st.warning("该答案要点来自模型输出，必须对照表格和原始证据人工重算后才能批准。")
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
    attestation = _review_attestation_inputs(f"public-review-{case['query_id']}", prior)
    actions = st.columns(3)
    selected_status = None
    if actions[0].button(
        "批准", type="primary", disabled=not all(attestation.values()),
        key=f"approve-{case['query_id']}",
    ):
        selected_status = "approved"
    if actions[1].button("退回", key=f"reject-{case['query_id']}"):
        selected_status = "rejected"
    if actions[2].button("标记待定", key=f"pending-{case['query_id']}"):
        selected_status = "pending"
    if selected_status:
        try:
            _client().put(f"/golden-candidates/cases/{case['query_id']}/review", json={
                "dataset_sha256": summary["dataset_sha256"],
                "status": selected_status, "notes": notes, "attestation": attestation,
            })
            st.success("复核状态已保存")
            st.rerun()
        except DashboardAPIError as exc:
            st.error(str(exc))


def _private_golden() -> None:
    st.header("私有黄金集")
    st.caption(
        "草稿和审批记录只保存在本机 SQLite；导出文件不会进入 Git。每次修改样例都会撤销原审批。"
    )
    st.warning("final split 只用于冻结后的最终验收，不得用于日常调参或 EvidenceGate 阈值校准。")

    with st.expander("创建私有数据集"):
        new_name = st.text_input("数据集名称", key="private-golden-new-name")
        new_version = st.text_input(
            "初始版本", value="1.0.0", key="private-golden-new-version",
        )
        if st.button("创建数据集", type="primary", key="private-golden-create-set"):
            try:
                _client().post("/private-golden-sets", json={
                    "name": new_name, "version": new_version,
                })
                st.success("私有黄金集草稿已创建")
                st.rerun()
            except DashboardAPIError as exc:
                st.error(str(exc))

    try:
        datasets = _client().get("/private-golden-sets")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    if not datasets:
        st.info("尚未创建私有黄金集。请先创建一个 50–200 条的本地草稿。")
        return

    dataset_labels = {
        f"{item['name']} · v{item['version']} · {item['case_count']} 条": item["id"]
        for item in datasets
    }
    selected_dataset = st.selectbox("选择私有黄金集", list(dataset_labels))
    set_id = dataset_labels[selected_dataset]
    try:
        dataset = _client().get(f"/private-golden-sets/{set_id}")
    except DashboardAPIError as exc:
        st.error(str(exc))
        return

    review_counts = dataset["review_counts"]
    split_counts = dataset["split_counts"]
    answerability = dataset["answerability_counts"]
    metrics = st.columns(7)
    metrics[0].metric("样例", dataset["case_count"])
    metrics[1].metric("已批准", review_counts["approved"])
    metrics[2].metric("待审批", review_counts["pending"])
    metrics[3].metric("已退回", review_counts["rejected"])
    metrics[4].metric("dev / final", f"{split_counts['dev']} / {split_counts['final']}")
    metrics[5].metric("可回答", answerability["answerable"])
    metrics[6].metric("不可回答", answerability["unanswerable"])

    with st.expander("数据集版本和名称"):
        edited_name = st.text_input(
            "名称", value=dataset["name"], key=f"private-set-name-{set_id}",
        )
        edited_version = st.text_input(
            "版本", value=dataset["version"], key=f"private-set-version-{set_id}",
        )
        st.caption("名称、样例或其他发布内容变化后，覆盖既有导出必须提升语义版本。")
        if st.button("保存数据集信息", key=f"private-set-save-{set_id}"):
            payload = {}
            if edited_name != dataset["name"]:
                payload["name"] = edited_name
            if edited_version != dataset["version"]:
                payload["version"] = edited_version
            if not payload:
                st.info("数据集信息没有变化")
            else:
                try:
                    _client().patch(f"/private-golden-sets/{set_id}", json=payload)
                    st.success("数据集信息已保存")
                    st.rerun()
                except DashboardAPIError as exc:
                    st.error(str(exc))

    if dataset["export_ready"]:
        st.success("50–200 条样例均已逐条批准，dev/final 和可回答性覆盖满足导出要求。")
        if st.button("导出本机私有黄金集", type="primary", key=f"private-export-{set_id}"):
            try:
                exported = _client().post(f"/private-golden-sets/{set_id}/export")
                st.success(
                    f"已导出 {exported['path']} · {exported['case_count']} 条 · "
                    f"SHA-256 {exported['sha256']}"
                )
            except DashboardAPIError as exc:
                st.error(str(exc))
    else:
        st.info("当前草稿尚不可导出：")
        for error in dataset["export_validation_errors"]:
            st.markdown(f"- {error}")

    status_filter = st.selectbox(
        "样例状态筛选", ["全部", "pending", "approved", "rejected"],
        key=f"private-case-filter-{set_id}",
    )
    case_path = f"/private-golden-sets/{set_id}/cases"
    if status_filter != "全部":
        case_path += f"?status={status_filter}"
    try:
        cases = _client().get(case_path)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return

    modes = ["新建样例"] + (["编辑已有样例"] if cases else [])
    mode = st.radio("样例操作", modes, horizontal=True, key=f"private-case-mode-{set_id}")
    selected_case: dict[str, object] = {}
    if mode == "编辑已有样例":
        case_labels = {
            f"{item['query_id']} · {item['split']} · {item['category']} · {item['review_status']}":
                item["query_id"]
            for item in cases
        }
        selected_label = st.selectbox(
            "选择样例", list(case_labels), key=f"private-case-select-{set_id}",
        )
        query_id = case_labels[selected_label]
        try:
            selected_case = _client().get(
                f"/private-golden-sets/{set_id}/cases/{query_id}",
            )
        except DashboardAPIError as exc:
            st.error(str(exc))
            return
    else:
        query_id = ""

    categories = [
        "精确事实", "同义词与缩写", "多文档组合", "长尾实体", "否定或条件限制",
        "时间敏感", "表格/列表信息", "模糊查询", "无答案", "冲突文档",
    ]
    difficulties = ["easy", "medium", "hard"]
    selected_category = str(selected_case.get("category", "精确事实"))
    selected_difficulty = str(selected_case.get("difficulty", "medium"))
    st.subheader("编辑样例" if selected_case else "新增样例")
    edited_query_id = st.text_input(
        "Query ID", value=str(selected_case.get("query_id", query_id)),
        disabled=bool(selected_case), key=f"private-query-id-{set_id}-{query_id or 'new'}",
    )
    query = st.text_area(
        "问题", value=str(selected_case.get("query", "")),
        key=f"private-query-{set_id}-{query_id or 'new'}",
    )
    form_columns = st.columns(4)
    category = form_columns[0].selectbox(
        "类别", categories, index=categories.index(selected_category),
        key=f"private-category-{set_id}-{query_id or 'new'}",
    )
    split = form_columns[1].selectbox(
        "Split", ["dev", "final"], index=0 if selected_case.get("split", "dev") == "dev" else 1,
        key=f"private-split-{set_id}-{query_id or 'new'}",
    )
    language = form_columns[2].text_input(
        "语言", value=str(selected_case.get("language", "zh")),
        key=f"private-language-{set_id}-{query_id or 'new'}",
    )
    difficulty = form_columns[3].selectbox(
        "难度", difficulties, index=difficulties.index(selected_difficulty),
        key=f"private-difficulty-{set_id}-{query_id or 'new'}",
    )
    answerable = st.checkbox(
        "可由知识库证据回答", value=bool(selected_case.get("answerable", True)),
        key=f"private-answerable-{set_id}-{query_id or 'new'}",
    )
    source = st.text_input(
        "来源标识", value=str(selected_case.get("source", "local-only")),
        key=f"private-source-{set_id}-{query_id or 'new'}",
    )
    expected_documents = st.text_area(
        "期望文档 ID（每行一个）",
        value="\n".join(selected_case.get("expected_document_ids", [])),
        key=f"private-docs-{set_id}-{query_id or 'new'}",
    )
    key_points = st.text_area(
        "答案要点（每行一个）",
        value="\n".join(selected_case.get("answer_key_points", [])),
        key=f"private-key-points-{set_id}-{query_id or 'new'}",
    )
    if split == "final":
        st.error("这是 final 样例：保存后不得将它用于日常调参。")
    if selected_case.get("review_status") == "approved":
        st.warning("该样例已批准；保存任何修改都会自动撤销审批并回到 pending。")

    if st.button("保存样例", type="primary", key=f"private-case-save-{set_id}-{query_id or 'new'}"):
        payload = {
            "query": query,
            "category": category,
            "expected_document_ids": [
                item.strip() for item in expected_documents.splitlines() if item.strip()
            ],
            "answer_key_points": [
                item.strip() for item in key_points.splitlines() if item.strip()
            ],
            "answerable": answerable,
            "source": source,
            "split": split,
            "language": language,
            "difficulty": difficulty,
        }
        try:
            _client().put(
                f"/private-golden-sets/{set_id}/cases/{edited_query_id}", json=payload,
            )
            st.success("样例已保存；当前状态为 pending")
            st.rerun()
        except DashboardAPIError as exc:
            st.error(str(exc))

    if selected_case:
        st.subheader("人工审批")
        st.caption(
            f"当前状态：{selected_case['review_status']} · "
            f"审批人：{selected_case.get('reviewer_id') or '无'}"
        )
        review_notes = st.text_area(
            "审批备注", value=str(selected_case.get("review_notes", "")),
            key=f"private-review-notes-{set_id}-{query_id}",
        )
        attestation = _review_attestation_inputs(
            f"private-review-{set_id}-{query_id}", selected_case,
        )
        review_columns = st.columns(3)
        review_status = None
        if review_columns[0].button(
            "批准样例", type="primary", disabled=not all(attestation.values()),
            key=f"private-approve-{set_id}-{query_id}",
        ):
            review_status = "approved"
        if review_columns[1].button("退回样例", key=f"private-reject-{set_id}-{query_id}"):
            review_status = "rejected"
        if review_columns[2].button("标记待定", key=f"private-pending-{set_id}-{query_id}"):
            review_status = "pending"
        if review_status:
            try:
                _client().put(
                    f"/private-golden-sets/{set_id}/cases/{query_id}/review",
                    json={
                        "status": review_status,
                        "notes": review_notes,
                        "attestation": attestation,
                    },
                )
                st.success("审批状态已保存")
                st.rerun()
            except DashboardAPIError as exc:
                st.error(str(exc))

        delete_confirmed = st.checkbox(
            f"确认删除样例 {query_id}", key=f"private-delete-confirm-{set_id}-{query_id}",
        )
        if st.button(
            "删除样例", disabled=not delete_confirmed,
            key=f"private-delete-{set_id}-{query_id}",
        ):
            try:
                _client().delete(f"/private-golden-sets/{set_id}/cases/{query_id}")
                st.success("样例已删除")
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
            pages.extend([
                "数据源", "用户管理", "黄金集复核", "私有黄金集", "离线评测", "可观测性",
            ])
        page = st.radio("导航", pages)
        if st.button("退出"):
            _logout()
            st.rerun()
    {"问答": _chat, "我的 Trace": _traces, "数据源": _sources, "用户管理": _users,
     "黄金集复核": _golden_review,
     "私有黄金集": _private_golden,
     "离线评测": _evaluations, "可观测性": _observability}[page]()


main()
