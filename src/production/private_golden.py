"""Local-only authoring and review workflow for the private golden dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.production.database import ProductionDatabase
from src.production.evaluation import load_golden_dataset, require_quality_gate_eligible
from src.production.review_attestation import (
    REVIEW_ATTESTATION_FIELDS,
    attestation_complete,
    normalize_attestation,
    require_complete_attestation,
)

_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_CATEGORIES = {
    "精确事实", "同义词与缩写", "多文档组合", "长尾实体", "否定或条件限制",
    "时间敏感", "表格/列表信息", "模糊查询", "无答案", "冲突文档",
}


class PrivateGoldenService:
    """Persist private cases in SQLite and export only fully reviewed datasets."""

    def __init__(self, database: ProductionDatabase, repository_root: Path) -> None:
        self.database = database
        self.repository_root = repository_root.resolve()
        self.evaluation_root = (self.repository_root / "evaluation").resolve()
        self.output_path = self.evaluation_root / "private_golden.json"

    def create_set(self, name: str, version: str, created_by: str) -> dict[str, Any]:
        name = self._required_text(name, "name", 120)
        version = self._version(version)
        set_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO private_golden_sets(
                   id,name,version,status,created_by,created_at,updated_at
                   ) VALUES (?,?,?,'draft',?,?,?)""",
                (set_id, name, version, created_by, now, now),
            )
        return self.get_set(set_id)

    def update_set(
        self, set_id: str, *, name: str | None = None, version: str | None = None,
    ) -> dict[str, Any]:
        current = self._set_row(set_id)
        new_name = current["name"] if name is None else self._required_text(name, "name", 120)
        new_version = current["version"] if version is None else self._version(version)
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE private_golden_sets
                   SET name=?,version=?,status='draft',updated_at=? WHERE id=?""",
                (new_name, new_version, now, set_id),
            )
        return self.get_set(set_id)

    def list_sets(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM private_golden_sets ORDER BY updated_at DESC",
            ).fetchall()
        return [self.get_set(str(row["id"])) for row in rows]

    def get_set(self, set_id: str) -> dict[str, Any]:
        row = self._set_row(set_id)
        cases = self.list_cases(set_id, include_content=False)
        counts = {"pending": 0, "approved": 0, "rejected": 0}
        splits = {"dev": 0, "final": 0}
        answerability = {"answerable": 0, "unanswerable": 0}
        for case in cases:
            counts[case["review_status"]] += 1
            splits[case["split"]] += 1
            answerability["answerable" if case["answerable"] else "unanswerable"] += 1
        errors = self._export_validation_errors(cases)
        return {
            **row,
            "case_count": len(cases),
            "review_counts": counts,
            "split_counts": splits,
            "answerability_counts": answerability,
            "export_ready": not errors,
            "export_validation_errors": errors,
            "output_path": self.output_path.relative_to(self.repository_root).as_posix(),
        }

    def put_case(
        self, set_id: str, query_id: str, payload: dict[str, Any], created_by: str,
    ) -> dict[str, Any]:
        self._set_row(set_id)
        normalized = self._normalize_case(query_id, payload)
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT created_by,created_at FROM private_golden_cases WHERE set_id=? AND query_id=?",
                (set_id, query_id),
            ).fetchone()
            connection.execute(
                """INSERT INTO private_golden_cases(
                   set_id,query_id,query_text,category,expected_document_ids_json,
                   answer_key_points_json,answerable,source,split,language,difficulty,
                   review_status,review_notes,reviewer_id,reviewed_at,created_by,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending','',NULL,NULL,?,?,?)
                   ON CONFLICT(set_id,query_id) DO UPDATE SET
                   query_text=excluded.query_text,category=excluded.category,
                   expected_document_ids_json=excluded.expected_document_ids_json,
                   answer_key_points_json=excluded.answer_key_points_json,
                   answerable=excluded.answerable,source=excluded.source,split=excluded.split,
                   language=excluded.language,difficulty=excluded.difficulty,
                   review_status='pending',review_notes='',reviewer_id=NULL,reviewed_at=NULL,
                   query_verified=0,answer_verified=0,evidence_verified=0,
                   category_difficulty_verified=0,split_verified=0,
                   updated_at=excluded.updated_at""",
                (
                    set_id, query_id, normalized["query"], normalized["category"],
                    json.dumps(normalized["expected_document_ids"], ensure_ascii=False),
                    json.dumps(normalized["answer_key_points"], ensure_ascii=False),
                    int(normalized["answerable"]), normalized["source"], normalized["split"],
                    normalized["language"], normalized["difficulty"],
                    str(existing["created_by"]) if existing else created_by,
                    str(existing["created_at"]) if existing else now, now,
                ),
            )
            connection.execute(
                "UPDATE private_golden_sets SET status='draft',updated_at=? WHERE id=?",
                (now, set_id),
            )
        return self.get_case(set_id, query_id)

    def list_cases(
        self, set_id: str, *, review_status: str | None = None,
        include_content: bool = True,
    ) -> list[dict[str, Any]]:
        self._set_row(set_id)
        sql = "SELECT * FROM private_golden_cases WHERE set_id=?"
        params: list[Any] = [set_id]
        if review_status is not None:
            if review_status not in {"pending", "approved", "rejected"}:
                raise ValueError("invalid review status")
            sql += " AND review_status=?"
            params.append(review_status)
        sql += " ORDER BY query_id"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._case_dict(row, include_content=include_content) for row in rows]

    def get_case(self, set_id: str, query_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM private_golden_cases WHERE set_id=? AND query_id=?",
                (set_id, query_id),
            ).fetchone()
        if row is None:
            raise KeyError(query_id)
        return self._case_dict(row, include_content=True)

    def review_case(
        self, set_id: str, query_id: str, status: str, notes: str, reviewer_id: str,
        *, attestation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid review status")
        normalized = normalize_attestation(attestation)
        require_complete_attestation(status, normalized)
        self.get_case(set_id, query_id)
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE private_golden_cases SET review_status=?,review_notes=?,
                   reviewer_id=?,reviewed_at=?,query_verified=?,answer_verified=?,
                   evidence_verified=?,category_difficulty_verified=?,split_verified=?,
                   updated_at=? WHERE set_id=? AND query_id=?""",
                (
                    status, notes.strip()[:4000], reviewer_id, now,
                    *(int(normalized[field]) for field in REVIEW_ATTESTATION_FIELDS),
                    now, set_id, query_id,
                ),
            )
            connection.execute(
                "UPDATE private_golden_sets SET status='draft',updated_at=? WHERE id=?",
                (now, set_id),
            )
        return self.get_case(set_id, query_id)

    def delete_case(self, set_id: str, query_id: str) -> None:
        self.get_case(set_id, query_id)
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM private_golden_cases WHERE set_id=? AND query_id=?",
                (set_id, query_id),
            )
            connection.execute(
                "UPDATE private_golden_sets SET status='draft',updated_at=? WHERE id=?",
                (now, set_id),
            )

    def export(self, set_id: str, exported_by: str) -> dict[str, Any]:
        dataset = self._set_row(set_id)
        cases = self.list_cases(set_id, include_content=True)
        errors = self._export_validation_errors(cases)
        if errors:
            raise ValueError("private golden set is not exportable: " + "; ".join(errors))
        exported_cases = [self._export_case(case) for case in cases]
        payload = {
            "artifact_type": "golden_set",
            "name": dataset["name"],
            "version": dataset["version"],
            "visibility": "private",
            "review_status": "human_reviewed",
            "eligible_for_quality_gate": True,
            "local_only": True,
            "source_draft": {"set_id": set_id},
            "human_review": {
                "attestation_version": 1,
                "exported_by": exported_by,
                "approved_case_count": len(exported_cases),
                "complete_attestation_count": len(exported_cases),
                "attestation_fields": list(REVIEW_ATTESTATION_FIELDS),
            },
            "cases": exported_cases,
        }
        self.evaluation_root.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            existing = json.loads(self.output_path.read_text(encoding="utf-8"))
            same_release = (
                existing.get("source_draft", {}).get("set_id") == set_id
                and existing.get("version") == dataset["version"]
            )
            same_content = self._release_content(existing) == self._release_content(payload)
            if same_release and not same_content:
                raise ValueError(
                    "private dataset content or metadata changed; increment its version "
                    "before replacing export"
                )
            if same_release and same_content:
                load_golden_dataset(self.output_path)
                require_quality_gate_eligible(self.output_path)
                self._mark_reviewed(set_id)
                return self._export_result(self.output_path, len(exported_cases), dataset["version"])
            existing_version = existing.get("version")
            try:
                replacing_newer_release = self._version_tuple(dataset["version"]) > self._version_tuple(
                    existing_version,
                )
            except ValueError as exc:
                raise ValueError(
                    "existing private golden export has an invalid version and cannot be "
                    "safely replaced"
                ) from exc
            if not replacing_newer_release:
                raise ValueError(
                    "private golden export already exists; increment the new dataset version "
                    "before replacing it"
                )
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        temporary = self.output_path.with_suffix(f"{self.output_path.suffix}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(encoded)
            load_golden_dataset(temporary)
            require_quality_gate_eligible(temporary)
            temporary.replace(self.output_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._mark_reviewed(set_id)
        return self._export_result(self.output_path, len(exported_cases), dataset["version"])

    def _mark_reviewed(self, set_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE private_golden_sets SET status='reviewed',updated_at=? WHERE id=?",
                (now, set_id),
            )

    def _set_row(self, set_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM private_golden_sets WHERE id=?", (set_id,),
            ).fetchone()
        if row is None:
            raise KeyError(set_id)
        return dict(row)

    @staticmethod
    def _case_dict(row: Any, *, include_content: bool) -> dict[str, Any]:
        result = {key: row[key] for key in row.keys()}
        result["answerable"] = bool(result["answerable"])
        normalized = normalize_attestation(result)
        result.update(normalized)
        result["attestation_complete"] = attestation_complete(normalized)
        result["expected_document_ids"] = json.loads(result.pop("expected_document_ids_json"))
        result["answer_key_points"] = json.loads(result.pop("answer_key_points_json"))
        if not include_content:
            result.pop("query_text", None)
            result.pop("expected_document_ids", None)
            result.pop("answer_key_points", None)
            result.pop("source", None)
            result.pop("review_notes", None)
        else:
            result["query"] = result.pop("query_text")
        return result

    def _normalize_case(self, query_id: str, value: dict[str, Any]) -> dict[str, Any]:
        query_id = self._required_text(query_id, "query_id", 160)
        query = self._required_text(value.get("query"), "query", 2000)
        category = self._required_text(value.get("category"), "category", 80)
        if category not in _CATEGORIES:
            raise ValueError("category is not in the supported golden-set taxonomy")
        expected = self._string_list(value.get("expected_document_ids"), "expected_document_ids")
        key_points = self._string_list(value.get("answer_key_points"), "answer_key_points")
        answerable = value.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("answerable must be a boolean")
        if answerable and (not expected or not key_points):
            raise ValueError("answerable cases need document IDs and answer key points")
        if not answerable and (expected or key_points):
            raise ValueError("unanswerable cases must not contain expected documents or key points")
        split = value.get("split")
        if split not in {"dev", "final"}:
            raise ValueError("split must be dev or final")
        difficulty = value.get("difficulty")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium or hard")
        return {
            "query_id": query_id,
            "query": query,
            "category": category,
            "expected_document_ids": expected,
            "answer_key_points": key_points,
            "answerable": answerable,
            "source": self._required_text(value.get("source"), "source", 500),
            "split": split,
            "language": self._required_text(value.get("language"), "language", 20),
            "difficulty": difficulty,
        }

    @staticmethod
    def _export_validation_errors(cases: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        if not 50 <= len(cases) <= 200:
            errors.append("case count must be between 50 and 200")
        if any(case["review_status"] != "approved" for case in cases):
            errors.append("every case must be approved")
        if any(not case.get("attestation_complete", False) for case in cases):
            errors.append("every case must include all review attestations")
        if not any(case["split"] == "dev" for case in cases):
            errors.append("dev split is empty")
        if not any(case["split"] == "final" for case in cases):
            errors.append("final split is empty")
        dev = [case for case in cases if case["split"] == "dev"]
        if not any(case["answerable"] for case in dev):
            errors.append("dev split needs an answerable case")
        if not any(not case["answerable"] for case in dev):
            errors.append("dev split needs an unanswerable case")
        return errors

    @staticmethod
    def _export_case(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_id": case["query_id"], "query": case["query"],
            "category": case["category"],
            "expected_document_ids": case["expected_document_ids"],
            "answer_key_points": case["answer_key_points"],
            "answerable": case["answerable"], "source": case["source"],
            "split": case["split"], "language": case["language"],
            "difficulty": case["difficulty"],
        }

    def _export_result(self, path: Path, case_count: int, version: str) -> dict[str, Any]:
        return {
            "path": path.resolve().relative_to(self.repository_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "case_count": case_count,
            "version": version,
            "eligible_for_quality_gate": True,
        }

    @staticmethod
    def _release_content(payload: dict[str, Any]) -> dict[str, Any]:
        """Return the version-governed fields, excluding the exporting operator."""
        keys = (
            "artifact_type", "name", "version", "visibility", "review_status",
            "eligible_for_quality_gate", "local_only", "source_draft", "cases",
        )
        return {key: payload.get(key) for key in keys}

    @staticmethod
    def _version_tuple(value: Any) -> tuple[int, int, int]:
        normalized = PrivateGoldenService._version(value)
        return tuple(int(part) for part in normalized.split("."))  # type: ignore[return-value]

    @staticmethod
    def _required_text(value: Any, field: str, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        result = value.strip()
        if len(result) > maximum:
            raise ValueError(f"{field} exceeds maximum length {maximum}")
        return result

    @staticmethod
    def _string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list) or len(value) > 20:
            raise ValueError(f"{field} must be a list with at most 20 values")
        result = [str(item).strip() for item in value]
        if any(not item for item in result) or len(result) != len(set(result)):
            raise ValueError(f"{field} values must be non-empty and unique")
        return result

    @staticmethod
    def _version(value: Any) -> str:
        if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value.strip()):
            raise ValueError("version must use semantic form MAJOR.MINOR.PATCH")
        return value.strip()
