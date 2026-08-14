"""Human review workflow for reproducible public golden-set candidates."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.production.database import ProductionDatabase


class GoldenReviewService:
    def __init__(self, database: ProductionDatabase, repository_root: Path) -> None:
        self.database = database
        self.repository_root = repository_root.resolve()
        self.evaluation_root = (self.repository_root / "evaluation").resolve()

    def summary(self, candidate_path: str = "evaluation/candidates/public_golden_candidate.json") -> dict[str, Any]:
        path, raw, payload = self._load(candidate_path)
        digest = hashlib.sha256(raw).hexdigest()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT status,COUNT(*) AS count FROM golden_reviews WHERE dataset_sha256=? GROUP BY status",
                (digest,),
            ).fetchall()
        counts = {"pending": 0, "approved": 0, "rejected": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        reviewed = counts["approved"] + counts["rejected"] + counts["pending"]
        total = len(payload["cases"])
        counts["unreviewed"] = max(total - reviewed, 0)
        return {
            "candidate_path": path.relative_to(self.repository_root).as_posix(),
            "dataset_sha256": digest, "name": payload["name"],
            "version": payload["version"], "total": total, "counts": counts,
            "all_approved": counts["approved"] == total,
            "eligible_for_quality_gate": bool(payload.get("eligible_for_quality_gate", False)),
        }

    def list_cases(
        self, candidate_path: str = "evaluation/candidates/public_golden_candidate.json",
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        _, raw, payload = self._load(candidate_path)
        digest = hashlib.sha256(raw).hexdigest()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT query_id,status,notes,reviewer_id,reviewed_at FROM golden_reviews WHERE dataset_sha256=?",
                (digest,),
            ).fetchall()
        reviews = {str(row["query_id"]): dict(row) for row in rows}
        result = []
        for case in payload["cases"]:
            review = reviews.get(case["query_id"])
            review_status = str(review["status"]) if review else "unreviewed"
            if status and review_status != status:
                continue
            result.append({
                "query_id": case["query_id"], "query": case["query"],
                "category": case["category"], "language": case["language"],
                "split": case["split"], "answerable": case["answerable"],
                "review_status": review_status, "review": review,
            })
        return result

    def get_case(
        self, query_id: str,
        candidate_path: str = "evaluation/candidates/public_golden_candidate.json",
    ) -> dict[str, Any]:
        _, raw, payload = self._load(candidate_path)
        digest = hashlib.sha256(raw).hexdigest()
        case = next((item for item in payload["cases"] if item["query_id"] == query_id), None)
        if case is None:
            raise KeyError(query_id)
        corpus = {item["document_id"]: item for item in payload["corpus"]}
        evidence = []
        for document_id in case["expected_document_ids"]:
            document = corpus.get(document_id)
            if document is None:
                raise ValueError(f"{query_id}: candidate references missing corpus item {document_id}")
            path = self._repository_file(document["path"])
            evidence.append({**document, "content": path.read_text(encoding="utf-8")})
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT status,notes,reviewer_id,reviewed_at FROM golden_reviews WHERE dataset_sha256=? AND query_id=?",
                (digest, query_id),
            ).fetchone()
        return {"dataset_sha256": digest, "case": case, "evidence": evidence,
                "review": dict(row) if row else None}

    def review(
        self, query_id: str, status: str, notes: str, reviewer_id: str,
        dataset_sha256: str,
        candidate_path: str = "evaluation/candidates/public_golden_candidate.json",
    ) -> dict[str, Any]:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid review status")
        detail = self.get_case(query_id, candidate_path)
        if detail["dataset_sha256"] != dataset_sha256:
            raise ValueError("candidate file changed; reload it before recording review")
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO golden_reviews(dataset_sha256,query_id,status,notes,reviewer_id,reviewed_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(dataset_sha256,query_id) DO UPDATE SET
                   status=excluded.status,notes=excluded.notes,
                   reviewer_id=excluded.reviewer_id,reviewed_at=excluded.reviewed_at""",
                (dataset_sha256, query_id, status, notes.strip(), reviewer_id, reviewed_at),
            )
        return {"query_id": query_id, "status": status, "notes": notes.strip(),
                "reviewer_id": reviewer_id, "reviewed_at": reviewed_at}

    def export_reviewed(
        self, dataset_sha256: str, exported_by: str,
        candidate_path: str = "evaluation/candidates/public_golden_candidate.json",
    ) -> dict[str, Any]:
        """Export an immutable, reviewed candidate without making it CI eligible."""
        path, raw, payload = self._load(candidate_path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != dataset_sha256:
            raise ValueError("candidate file changed; reload it before exporting")
        query_ids = {str(case["query_id"]) for case in payload["cases"]}
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT query_id,status,notes,reviewer_id,reviewed_at
                   FROM golden_reviews WHERE dataset_sha256=?""",
                (digest,),
            ).fetchall()
        reviews = {str(row["query_id"]): dict(row) for row in rows}
        missing = sorted(query_id for query_id in query_ids
                         if reviews.get(query_id, {}).get("status") != "approved")
        if missing:
            raise ValueError(
                f"all candidate cases must be approved before export; {len(missing)} remain"
            )

        exported_at = datetime.now(timezone.utc).isoformat()
        reviewed_cases = []
        for case in payload["cases"]:
            metadata = {**case.get("candidate_metadata", {}), "human_reviewed": True}
            reviewed_cases.append({**case, "candidate_metadata": metadata})
        reviewed_payload = {
            **payload,
            "artifact_type": "golden_set_reviewed_candidate",
            "review_status": "all_cases_approved",
            # Promotion to the canonical dataset remains a separate reviewed code change.
            "eligible_for_quality_gate": False,
            "source_candidate": {
                "path": path.relative_to(self.repository_root).as_posix(),
                "sha256": digest,
            },
            "human_review": {
                "exported_at": exported_at,
                "exported_by": exported_by,
                "approved_case_count": len(query_ids),
                "records": [reviews[query_id] for query_id in sorted(query_ids)],
            },
            "cases": reviewed_cases,
        }
        output_directory = (self.evaluation_root / "reviewed").resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / f"public_golden_reviewed-{digest[:12]}.json"
        encoded = (json.dumps(reviewed_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("source_candidate", {}).get("sha256") != digest:
                raise FileExistsError(f"reviewed artifact path already exists: {output}")
        else:
            temporary = output.with_suffix(f"{output.suffix}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(encoded)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "path": output.relative_to(self.repository_root).as_posix(),
            "source_candidate_sha256": digest,
            "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "case_count": len(query_ids),
            "eligible_for_quality_gate": False,
        }

    def _load(self, candidate_path: str) -> tuple[Path, bytes, dict[str, Any]]:
        path = self._repository_file(candidate_path)
        if self.evaluation_root != path and self.evaluation_root not in path.parents:
            raise PermissionError("candidate path is outside evaluation directory")
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("artifact_type") != "golden_set_candidate":
            raise ValueError("file is not a golden-set candidate artifact")
        if payload.get("eligible_for_quality_gate") is not False:
            raise ValueError("candidate artifact must remain quality-gate ineligible")
        return path, raw, payload

    def _repository_file(self, value: str) -> Path:
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (self.repository_root / path).resolve()
        if self.repository_root != path and self.repository_root not in path.parents:
            raise PermissionError("path is outside repository")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
