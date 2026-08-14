"""Authenticated FastAPI boundary for the production RAG application."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.settings import resolve_path
from src.production.auth import Principal
from src.production.golden_review import GoldenReviewService
from src.production.runtime import Runtime


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: Literal["admin", "user"] = "user"


class UserUpdate(BaseModel):
    role: Literal["admin", "user"] | None = None
    active: bool | None = None
    password: str | None = None


class SourceCreate(BaseModel):
    name: str
    kind: Literal["directory", "web", "upload"]
    location: str
    collection_name: str = "default"
    config: dict[str, Any] = Field(default_factory=dict)


class SourceUpdate(BaseModel):
    name: str | None = None
    collection_name: str | None = None
    enabled: bool | None = None
    config: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=50)


class EvaluationRequest(BaseModel):
    dataset_path: str = "evaluation/public_golden.json"
    split: Literal["dev", "final"] = "final"
    collection: str = "evaluation"
    top_k: int = Field(default=5, ge=1, le=50)
    repeats: int = Field(default=5, ge=1, le=20)


class GoldenReviewRequest(BaseModel):
    dataset_sha256: str
    status: Literal["pending", "approved", "rejected"]
    notes: str = Field(default="", max_length=4000)


class GoldenExportRequest(BaseModel):
    dataset_sha256: str


class _RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        values = self._requests[key]
        while values and now - values[0] >= 60:
            values.popleft()
        if len(values) >= self.per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        values.append(now)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    owned_runtime = runtime is None
    runtime = runtime or Runtime.create()
    limiter = _RateLimiter(runtime.settings.server.rate_limit_per_minute)
    golden_reviews = GoldenReviewService(runtime.database, resolve_path(".").resolve())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        warmup_finished = threading.Event()

        def warmup_dependencies() -> None:
            try:
                result = runtime.rag.warmup()
                app.state.warmup = {"status": "completed", **result}
            except Exception as exc:
                app.state.warmup = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "evidence_calibration": runtime.settings.evidence.calibration_status,
                }
            finally:
                warmup_finished.set()

        app.state.warmup = {
            "status": "starting",
            "evidence_calibration": runtime.settings.evidence.calibration_status,
        }
        warmup_thread = None
        if hasattr(runtime.rag, "warmup"):
            warmup_thread = threading.Thread(
                target=warmup_dependencies, name="rag-dependency-warmup", daemon=True,
            )
            warmup_thread.start()
        else:
            app.state.warmup = {
                "status": "completed",
                "evidence_calibration": runtime.settings.evidence.calibration_status,
            }
            warmup_finished.set()
        yield
        if warmup_thread is not None:
            warmup_thread.join(timeout=1)
        if owned_runtime and warmup_finished.is_set():
            runtime.close()

    app = FastAPI(title="Modular RAG Production API", version="1.0.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware, allow_origins=[], allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type"],
    )

    def principal(authorization: str = Header(default="")) -> Principal:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        try:
            value = runtime.auth.verify_access(authorization[7:])
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        limiter.check(value.user_id)
        return value

    def admin(value: Principal = Depends(principal)) -> Principal:
        if value.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator role required")
        return value

    def dependencies_ready() -> bool:
        warmup = getattr(app.state, "warmup", {})
        return bool(
            warmup.get("status") == "completed"
            and not warmup.get("error")
            and not (
                warmup.get("reranker_configured")
                and not warmup.get("reranker_ready")
            )
            and warmup.get("evidence_calibration", "pending") == "calibrated"
        )

    def require_retrieval_ready() -> None:
        warmup = getattr(app.state, "warmup", {})
        if warmup.get("status") != "completed" or warmup.get("error"):
            raise HTTPException(
                status_code=503,
                detail="RAG dependencies are still warming up",
            )

    def require_answer_ready() -> None:
        require_retrieval_ready()
        warmup = getattr(app.state, "warmup", {})
        if warmup.get("evidence_calibration", "pending") != "calibrated":
            raise HTTPException(
                status_code=503,
                detail="EvidenceGate thresholds have not been calibrated",
            )

    @app.get("/health/live")
    def live() -> dict[str, Any]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready(response: Response) -> dict[str, Any]:
        health = runtime.database.health()
        warmup = getattr(app.state, "warmup", None)
        dependency_ok = isinstance(warmup, dict) and dependencies_ready()
        ready_now = bool(health["ok"] and dependency_ok)
        if not ready_now:
            response.status_code = 503
        health.update({"status": "ready" if ready_now else "not_ready",
                       "bootstrap_required": not runtime.auth.has_users(), "warmup": warmup})
        return health

    @app.post("/auth/bootstrap", status_code=201)
    def bootstrap(body: LoginRequest, request: Request) -> dict[str, Any]:
        try:
            user = runtime.auth.bootstrap_admin(body.username, body.password)
            runtime.audit.record("auth.bootstrap", user.user_id, "user", user.user_id, ip_address=request.client.host if request.client else None)
            return runtime.auth.authenticate(body.username, body.password)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request) -> dict[str, Any]:
        try:
            result = runtime.auth.authenticate(body.username, body.password)
            runtime.audit.record("auth.login", result["user"]["id"], "session", detail={}, ip_address=request.client.host if request.client else None)
            return result
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/auth/refresh")
    def refresh(body: RefreshRequest) -> dict[str, Any]:
        try:
            return runtime.auth.refresh(body.refresh_token)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/auth/logout", status_code=204)
    def logout(body: RefreshRequest, value: Principal = Depends(principal)) -> None:
        runtime.auth.logout(body.refresh_token)
        runtime.audit.record("auth.logout", value.user_id, "session")

    @app.get("/users")
    def users(value: Principal = Depends(admin)) -> list[dict[str, Any]]:
        return runtime.auth.list_users(value)

    @app.post("/users", status_code=201)
    def create_user(body: UserCreate, value: Principal = Depends(admin)) -> dict[str, Any]:
        try:
            user = runtime.auth.create_user(body.username, body.password, body.role, value)
            runtime.audit.record("user.create", value.user_id, "user", user.user_id, {"role": user.role})
            return {"id": user.user_id, "username": user.username, "role": user.role}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/users/{user_id}")
    def update_user(user_id: str, body: UserUpdate, value: Principal = Depends(admin)) -> dict[str, Any]:
        changes = body.model_dump(exclude_none=True)
        try:
            result = runtime.auth.update_user(user_id, changes, value)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="User not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.audit.record("user.update", value.user_id, "user", user_id, {k: v for k, v in changes.items() if k != "password"})
        return result

    @app.get("/data-sources")
    def list_sources(_: Principal = Depends(admin)) -> list[dict[str, Any]]:
        return runtime.sources.list()

    @app.post("/data-sources", status_code=201)
    def create_source(body: SourceCreate, value: Principal = Depends(admin)) -> dict[str, Any]:
        try:
            result = runtime.sources.create(body.name, body.kind, body.location, body.collection_name, value.user_id, body.config)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.audit.record("source.create", value.user_id, "data_source", result["id"], {"kind": body.kind})
        return result

    @app.patch("/data-sources/{source_id}")
    def update_source(source_id: str, body: SourceUpdate, value: Principal = Depends(admin)) -> dict[str, Any]:
        try:
            result = runtime.sources.update(source_id, body.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Data source not found") from exc
        runtime.audit.record("source.update", value.user_id, "data_source", source_id)
        return result

    @app.delete("/data-sources/{source_id}", status_code=202)
    def delete_source(source_id: str, value: Principal = Depends(admin)) -> dict[str, str]:
        try:
            runtime.sources.get(source_id)
            job_id = runtime.jobs.enqueue(
                "delete_source", {"source_id": source_id}, value.user_id,
                f"source:{source_id}",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Data source not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.audit.record(
            "source.delete_requested", value.user_id, "data_source", source_id,
            {"job_id": job_id},
        )
        return {"job_id": job_id}

    @app.post("/data-sources/{source_id}/sync", status_code=202)
    def sync_source(source_id: str, value: Principal = Depends(admin)) -> dict[str, str]:
        try:
            runtime.sources.get(source_id)
            job_id = runtime.jobs.enqueue("sync_source", {"source_id": source_id}, value.user_id, f"source:{source_id}")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Data source not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.audit.record("source.sync", value.user_id, "data_source", source_id, {"job_id": job_id})
        return {"job_id": job_id}

    @app.post("/documents/upload", status_code=202)
    async def upload_document(file: UploadFile = File(...), collection: str = "default", value: Principal = Depends(admin)) -> dict[str, str]:
        suffix = Path(file.filename or "").suffix.lower()
        allowed = set(runtime.settings.ingestion.allowed_extensions if runtime.settings.ingestion else (".pdf", ".md", ".txt"))
        if suffix not in allowed:
            raise HTTPException(status_code=415, detail="Unsupported document type")
        upload_id = str(uuid.uuid4())
        target_dir = resolve_path("data/uploads") / upload_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / f"document{suffix}"
        max_bytes = runtime.settings.server.upload_max_mb * 1024 * 1024
        written = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    handle.close()
                    shutil.rmtree(target_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="Upload exceeds configured limit")
                handle.write(chunk)
        job_id = runtime.jobs.enqueue("ingest_file", {"path": str(target), "collection": collection}, value.user_id, f"upload:{upload_id}")
        runtime.audit.record("document.upload", value.user_id, "upload", upload_id, {"job_id": job_id, "size": written})
        return {"job_id": job_id, "upload_id": upload_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, value: Principal = Depends(principal)) -> dict[str, Any]:
        try:
            job = runtime.jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if value.role != "admin" and job["created_by"] != value.user_id:
            raise HTTPException(status_code=403, detail="Job is not visible to this user")
        return job

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, value: Principal = Depends(principal)) -> dict[str, Any]:
        job = get_job(job_id, value)
        del job
        return runtime.jobs.cancel(job_id)

    @app.post("/search")
    def search(body: QueryRequest, value: Principal = Depends(principal)) -> dict[str, Any]:
        require_retrieval_ready()
        try:
            return runtime.rag.search(body.query, body.collection, body.top_k, value.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/answer")
    def answer(body: QueryRequest, value: Principal = Depends(principal)) -> dict[str, Any]:
        require_answer_ready()
        return runtime.rag.answer(body.query, body.collection, body.top_k, value.user_id)

    @app.get("/traces")
    def traces(
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
        prompt_version: str | None = None,
        model_version: str | None = None,
        dataset_version: str | None = None,
        config_version: str | None = None,
        value: Principal = Depends(principal),
    ) -> list[dict[str, Any]]:
        return runtime.traces.list(
            value.user_id, value.role == "admin", limit,
            start=start, end=end, prompt_version=prompt_version,
            model_version=model_version, dataset_version=dataset_version,
            config_version=config_version,
        )

    @app.get("/traces/{trace_id}")
    def trace(trace_id: str, value: Principal = Depends(principal)) -> dict[str, Any]:
        result = runtime.traces.get(trace_id, value.user_id, value.role == "admin")
        if result is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return result

    @app.get("/metrics/summary")
    def metrics(start: str | None = None, end: str | None = None, value: Principal = Depends(admin)) -> dict[str, Any]:
        del value
        return runtime.traces.metrics(start, end)

    @app.get("/metrics/timeseries")
    def metric_time_series(
        start: str, end: str, bucket: Literal["hour", "day"] = "hour",
        value: Principal = Depends(admin),
    ) -> list[dict[str, Any]]:
        del value
        try:
            return runtime.traces.time_series(start, end, bucket)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/metrics/compare")
    def compare(start: str, end: str, baseline_start: str, baseline_end: str, value: Principal = Depends(admin)) -> dict[str, Any]:
        del value
        return runtime.traces.compare_windows(start, end, baseline_start, baseline_end)

    @app.post("/evaluations", status_code=202)
    def evaluate(body: EvaluationRequest, value: Principal = Depends(admin)) -> dict[str, str]:
        evaluation_root = resolve_path("evaluation").resolve()
        path = Path(body.dataset_path)
        path = path.resolve() if path.is_absolute() else resolve_path(body.dataset_path).resolve()
        if evaluation_root != path and evaluation_root not in path.parents:
            raise HTTPException(status_code=400, detail="Evaluation path is outside evaluation directory")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        evaluation_id = str(uuid.uuid4())
        payload = body.model_dump()
        payload["dataset_path"] = str(path)
        payload["evaluation_id"] = evaluation_id
        payload["created_by"] = value.user_id
        with runtime.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO evaluations(id,dataset_name,dataset_version,split,status,config_json,created_by,created_at)
                   VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                (evaluation_id, Path(body.dataset_path).name, "pending-validation", body.split,
                 "queued", json.dumps(payload, ensure_ascii=False), value.user_id),
            )
        job_id = runtime.jobs.enqueue("evaluation", payload, value.user_id, "evaluation")
        runtime.audit.record("evaluation.create", value.user_id, "job", job_id)
        return {"job_id": job_id, "evaluation_id": evaluation_id}

    @app.get("/evaluations")
    def evaluations(limit: int = 50, value: Principal = Depends(admin)) -> list[dict[str, Any]]:
        del value
        limit = min(max(limit, 1), 200)
        with runtime.database.connect() as connection:
            rows = connection.execute(
                """SELECT id,dataset_name,dataset_version,split,status,metrics_json,
                          report_path,created_at,finished_at
                   FROM evaluations ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
            results.append(item)
        return results

    @app.get("/evaluations/{evaluation_id}")
    def evaluation(evaluation_id: str, value: Principal = Depends(admin)) -> dict[str, Any]:
        del value
        with runtime.database.connect() as connection:
            row = connection.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json")) if result["metrics_json"] else None
        result["config"] = json.loads(result.pop("config_json"))
        result["report"] = None
        if result["report_path"]:
            report_root = resolve_path("outputs/evaluations").resolve()
            report_path = Path(result["report_path"])
            report_path = report_path.resolve() if report_path.is_absolute() else resolve_path(report_path).resolve()
            if report_root == report_path.parent and report_path.is_file():
                try:
                    result["report"] = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    result["report"] = None
        return result

    @app.get("/golden-candidates/summary")
    def golden_candidate_summary(value: Principal = Depends(admin)) -> dict[str, Any]:
        del value
        return golden_reviews.summary()

    @app.get("/golden-candidates/cases")
    def golden_candidate_cases(
        status: Literal["unreviewed", "pending", "approved", "rejected"] | None = None,
        value: Principal = Depends(admin),
    ) -> list[dict[str, Any]]:
        del value
        return golden_reviews.list_cases(status=status)

    @app.get("/golden-candidates/cases/{query_id}")
    def golden_candidate_case(query_id: str, value: Principal = Depends(admin)) -> dict[str, Any]:
        del value
        try:
            return golden_reviews.get_case(query_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Candidate query not found") from exc

    @app.put("/golden-candidates/cases/{query_id}/review")
    def review_golden_candidate(
        query_id: str, body: GoldenReviewRequest, value: Principal = Depends(admin),
    ) -> dict[str, Any]:
        try:
            result = golden_reviews.review(
                query_id, body.status, body.notes, value.user_id, body.dataset_sha256,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Candidate query not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.audit.record(
            "golden_candidate.review", value.user_id, "golden_candidate", query_id,
            {"status": body.status, "dataset_sha256": body.dataset_sha256},
        )
        return result

    @app.post("/golden-candidates/export-reviewed")
    def export_reviewed_golden_candidate(
        body: GoldenExportRequest, value: Principal = Depends(admin),
    ) -> dict[str, Any]:
        try:
            result = golden_reviews.export_reviewed(body.dataset_sha256, value.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime.audit.record(
            "golden_candidate.export_reviewed", value.user_id, "golden_candidate",
            body.dataset_sha256, result,
        )
        return result

    return app


def main() -> None:
    import uvicorn
    runtime = Runtime.create()
    uvicorn.run(create_app(runtime), host=runtime.settings.server.host, port=runtime.settings.server.port)


def app_factory() -> FastAPI:
    """Uvicorn factory that avoids creating files and models on import."""
    return create_app()


# Backward-compatible opt-in for deployments that require ``module:app``.
app = create_app() if os.environ.get("RAG_EAGER_APP", "0") == "1" else None
