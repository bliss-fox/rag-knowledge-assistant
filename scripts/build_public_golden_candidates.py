#!/usr/bin/env python
"""Build a reproducible, license-audited public golden-set candidate pack.

The output is deliberately marked ineligible for the quality gate.  It becomes
a formal golden set only after a human checks every question, evidence mapping,
answer key point and dev/final assignment and promotes it through a separate
reviewed change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LICENSE_ALLOWLIST = {"apache-2.0", "cc-by-sa-4.0"}
SOURCES = {
    "cmrc2018": {
        "dataset": "hfl/cmrc2018",
        "revision": "137f2c45a24275fb68f6961c4d357f46288886aa",
        "license": "cc-by-sa-4.0",
        "url": (
            "https://huggingface.co/datasets/hfl/cmrc2018/resolve/"
            "137f2c45a24275fb68f6961c4d357f46288886aa/"
            "data/validation-00000-of-00001.parquet"
        ),
        "homepage": "https://huggingface.co/datasets/hfl/cmrc2018",
    },
    "squad_v2": {
        "dataset": "rajpurkar/squad_v2",
        "revision": "3ffb306f725f7d2ce8394bc1873b24868140c412",
        "license": "cc-by-sa-4.0",
        "url": (
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/"
            "3ffb306f725f7d2ce8394bc1873b24868140c412/"
            "squad_v2/validation-00000-of-00001.parquet"
        ),
        "homepage": "https://huggingface.co/datasets/rajpurkar/squad_v2",
    },
    "hotpotqa": {
        "dataset": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "license": "cc-by-sa-4.0",
        "url": (
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
            "1908d6afbbead072334abe2965f91bd2709910ab/"
            "distractor/validation-00000-of-00001.parquet"
        ),
        "homepage": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
    },
}


@dataclass(frozen=True)
class Document:
    document_id: str
    path: str
    source_dataset: str
    source_record_id: str
    title: str
    content: str


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stable_rows(frame: pd.DataFrame, count: int, *, salt: str, predicate=None) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    if predicate is not None:
        records = [row for row in records if predicate(row)]
    records.sort(key=lambda row: hashlib.sha256(
        f"{salt}:{row.get('id', '')}".encode("utf-8")
    ).hexdigest())
    if len(records) < count:
        raise ValueError(f"{salt}: requested {count} rows but only {len(records)} are eligible")
    return records[:count]


class CandidateBuilder:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.corpus_root = output_root / "public_corpus"
        self.documents: dict[str, Document] = {}
        self.cases: list[dict[str, Any]] = []

    def add_document(
        self, source_dataset: str, record_id: str, title: str, text: str,
    ) -> Document:
        content = f"# {title.strip() or record_id}\n\n{text.strip()}\n"
        encoded = content.encode("utf-8")
        document_id = f"doc_{hashlib.sha256(encoded).hexdigest()[:16]}"
        existing = self.documents.get(document_id)
        if existing is not None:
            return existing
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", record_id).strip("-")[:60] or "record"
        path = self.corpus_root / source_dataset / f"{safe}-{document_id[4:12]}.md"
        document = Document(
            document_id=document_id,
            path=path.relative_to(self.output_root.parent.parent).as_posix(),
            source_dataset=source_dataset,
            source_record_id=record_id,
            title=title,
            content=content,
        )
        self.documents[document_id] = document
        return document

    def add_cmrc(self, frame: pd.DataFrame, count: int = 45) -> None:
        for index, row in enumerate(_stable_rows(frame, count, salt="cmrc2018")):
            answers = [str(value) for value in _sequence(row["answers"]["text"]) if str(value)]
            document = self.add_document("cmrc2018", str(row["id"]), "CMRC 2018 passage", str(row["context"]))
            self.cases.append(self._case(
                source="cmrc2018", index=index, raw_id=str(row["id"]),
                query=str(row["question"]), category="精确事实",
                expected=[document.document_id], key_points=answers[:1], language="zh",
                difficulty="easy", source_path=document.path,
            ))

    def add_squad(self, frame: pd.DataFrame, answerable: int = 15, unanswerable: int = 15) -> None:
        has_answer = lambda row: bool(_sequence(row["answers"]["text"]))
        selected = [
            *[(row, True) for row in _stable_rows(frame, answerable, salt="squad-v2-answerable", predicate=has_answer)],
            *[(row, False) for row in _stable_rows(frame, unanswerable, salt="squad-v2-unanswerable", predicate=lambda row: not has_answer(row))],
        ]
        for index, (row, can_answer) in enumerate(selected):
            title = str(row.get("title") or "SQuAD 2.0 passage")
            document = self.add_document("squad_v2", str(row["id"]), title, str(row["context"]))
            answers = [str(value) for value in _sequence(row["answers"]["text"]) if str(value)]
            self.cases.append(self._case(
                source="squad_v2", index=index, raw_id=str(row["id"]),
                query=str(row["question"]), category="精确事实" if can_answer else "无答案",
                expected=[document.document_id] if can_answer else [],
                key_points=answers[:1] if can_answer else [], language="en",
                difficulty="medium" if can_answer else "hard", source_path=document.path,
                answerable=can_answer,
            ))

    def add_hotpot(self, frame: pd.DataFrame, count: int = 25) -> None:
        for index, row in enumerate(_stable_rows(frame, count, salt="hotpotqa")):
            context = row["context"]
            titles = [str(value) for value in _sequence(context["title"])]
            sentences = _sequence(context["sentences"])
            support_titles = {str(value) for value in _sequence(row["supporting_facts"]["title"])}
            by_title: dict[str, Document] = {}
            for title, parts in zip(titles, sentences):
                text = "".join(str(value) for value in _sequence(parts))
                by_title[title] = self.add_document(
                    "hotpotqa", f"{row['id']}-{title}", title, text,
                )
            missing = support_titles - by_title.keys()
            if missing:
                raise ValueError(f"{row['id']}: supporting titles missing from context: {sorted(missing)}")
            expected = [by_title[title].document_id for title in sorted(support_titles)]
            primary = by_title[sorted(support_titles)[0]]
            self.cases.append(self._case(
                source="hotpotqa", index=index, raw_id=str(row["id"]),
                query=str(row["question"]), category="多文档组合", expected=expected,
                key_points=[str(row["answer"])], language="en",
                difficulty=str(row.get("level") or "hard"), source_path=primary.path,
            ))

    @staticmethod
    def _case(
        *, source: str, index: int, raw_id: str, query: str, category: str,
        expected: list[str], key_points: list[str], language: str, difficulty: str,
        source_path: str, answerable: bool = True,
    ) -> dict[str, Any]:
        split = "dev" if index % 2 == 0 else "final"
        return {
            "query_id": f"candidate-{source}-{raw_id}", "query": query,
            "category": category, "expected_document_ids": expected,
            "answer_key_points": key_points, "answerable": answerable,
            "source": source_path, "split": split, "language": language,
            "difficulty": difficulty,
            "candidate_metadata": {
                "upstream_dataset": SOURCES[source]["dataset"],
                "upstream_record_id": raw_id,
                "human_reviewed": False,
            },
        }

    def write(self) -> Path:
        self._validate()
        for document in self.documents.values():
            target = self.output_root.parent.parent / document.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(document.content.encode("utf-8"))
        sources = [{key: value for key, value in spec.items() if key != "url"}
                   for spec in SOURCES.values()]
        manifest = {
            "artifact_type": "golden_set_candidate",
            "name": "modular-rag-public-golden-candidate",
            "version": "2026.08.13.1",
            "visibility": "public",
            "review_status": "requires_human_review",
            "eligible_for_quality_gate": False,
            "promotion_requirements": [
                "review every query and answer key point",
                "verify every expected document supports the query",
                "review category, difficulty and dev/final assignment",
                "record reviewer identity and review date",
            ],
            "sources": sources,
            "corpus": [{
                "document_id": doc.document_id, "path": doc.path,
                "source_dataset": doc.source_dataset,
                "source_record_id": doc.source_record_id,
                "title": doc.title,
            } for doc in sorted(self.documents.values(), key=lambda value: value.path)],
            "cases": sorted(self.cases, key=lambda value: value["query_id"]),
        }
        target = self.output_root / "public_golden_candidate.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        return target

    def _validate(self) -> None:
        if len(self.cases) != 100:
            raise ValueError(f"candidate pack must contain 100 cases, got {len(self.cases)}")
        query_ids = [case["query_id"] for case in self.cases]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("candidate query IDs are not unique")
        for name, spec in SOURCES.items():
            if spec["license"] not in LICENSE_ALLOWLIST:
                raise ValueError(f"{name}: license is not allowlisted: {spec['license']}")
        known = set(self.documents)
        for case in self.cases:
            if case["answerable"] and not case["expected_document_ids"]:
                raise ValueError(f"{case['query_id']}: answerable candidate has no evidence")
            if not set(case["expected_document_ids"]) <= known:
                raise ValueError(f"{case['query_id']}: references an unknown document")
            if case["candidate_metadata"]["human_reviewed"] is not False:
                raise ValueError("generator must never mark candidates as human reviewed")


def build(output_root: Path) -> Path:
    builder = CandidateBuilder(output_root)
    frames = {name: pd.read_parquet(spec["url"]) for name, spec in SOURCES.items()}
    builder.add_cmrc(frames["cmrc2018"])
    builder.add_squad(frames["squad_v2"])
    builder.add_hotpot(frames["hotpotqa"])
    return builder.write()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation/candidates"))
    args = parser.parse_args()
    target = build(args.output.resolve())
    print(f"Wrote public candidate pack: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
