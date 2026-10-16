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
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LICENSE_ALLOWLIST = {
    "apache-2.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by-1.0",
}
SOURCES = {
    "cmrc2018": {
        "dataset": "hfl/cmrc2018",
        "revision": "137f2c45a24275fb68f6961c4d357f46288886aa",
        "license": "cc-by-sa-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Chinese Machine Reading Comprehension 2018",
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/hfl/cmrc2018/raw/"
                "137f2c45a24275fb68f6961c4d357f46288886aa/README.md"
            ),
            "sha256": "977508e5d90c3e590dbffe7b8a6d2e9748eb2f61dc791a4dbc8b2fa7cf0d9688",
            "size_bytes": 7851,
        },
        "artifacts": [{
            "role": "validation",
            "url": (
                "https://huggingface.co/datasets/hfl/cmrc2018/resolve/"
                "137f2c45a24275fb68f6961c4d357f46288886aa/"
                "data/validation-00000-of-00001.parquet"
            ),
            "sha256": "c3d3116e5845b8d3c44a863c1e2f973817c9f10a52e503b079ac49d4366f78b2",
            "size_bytes": 1136115,
        }],
        "homepage": "https://huggingface.co/datasets/hfl/cmrc2018",
    },
    "xquad_zh": {
        "dataset": "google/xquad",
        "config": "xquad.zh",
        "revision": "51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
        "license": "cc-by-sa-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "XQuAD: Cross-lingual Question Answering Dataset",
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/google/xquad/raw/"
                "51adfef1c1287aab1d2d91b5bead9bcfb9c68583/README.md"
            ),
            "sha256": "9bb7215530f157e3761585cfbfe11161cda1951546dbfa5b65fd797972c85946",
            "size_bytes": 15630,
        },
        "artifacts": [{
            "role": "validation",
            "url": (
                "https://huggingface.co/datasets/google/xquad/resolve/"
                "51adfef1c1287aab1d2d91b5bead9bcfb9c68583/"
                "xquad.zh/validation-00000-of-00001.parquet"
            ),
            "sha256": "6d9033e82b2a1dbf67b66025e84fab9a44e13dc91401e4914a9529db784146f6",
            "size_bytes": 205768,
        }],
        "homepage": "https://huggingface.co/datasets/google/xquad",
    },
    "squad_v2": {
        "dataset": "rajpurkar/squad_v2",
        "revision": "3ffb306f725f7d2ce8394bc1873b24868140c412",
        "license": "cc-by-sa-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "SQuAD 2.0",
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/rajpurkar/squad_v2/raw/"
                "3ffb306f725f7d2ce8394bc1873b24868140c412/README.md"
            ),
            "sha256": "7a6726352072e8517318959fe0438d4f45880affc041ced6da3bf2e4dec94c68",
            "size_bytes": 8916,
        },
        "artifacts": [{
            "role": "validation",
            "url": (
                "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/"
                "3ffb306f725f7d2ce8394bc1873b24868140c412/"
                "squad_v2/validation-00000-of-00001.parquet"
            ),
            "sha256": "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb",
            "size_bytes": 1350511,
        }],
        "homepage": "https://huggingface.co/datasets/rajpurkar/squad_v2",
    },
    "hotpotqa": {
        "dataset": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "license": "cc-by-sa-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "HotpotQA",
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/hotpotqa/hotpot_qa/raw/"
                "1908d6afbbead072334abe2965f91bd2709910ab/README.md"
            ),
            "sha256": "3cfab003a856275d3198b031c6b2ac46c63178fb462a4123705f652b71b22813",
            "size_bytes": 9522,
        },
        "artifacts": [{
            "role": "validation",
            "url": (
                "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
                "1908d6afbbead072334abe2965f91bd2709910ab/"
                "distractor/validation-00000-of-00001.parquet"
            ),
            "sha256": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
            "size_bytes": 27452575,
        }],
        "homepage": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
    },
    "multihop_rag": {
        "dataset": "yixuantt/MultiHopRAG",
        "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
        "license": "odc-by-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "attribution": (
            "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for "
            "Multi-Hop Queries, Yixuan Tang and Yi Yang (2024)"
        ),
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/yixuantt/MultiHopRAG/raw/"
                "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/README.md"
            ),
            "sha256": "f671db44cf1802f359451e466d82c68c5059d58451b8976ce8329b74d8b89001",
            "size_bytes": 1312,
            "declared_license": "odc-by",
        },
        "artifacts": [
            {
                "role": "questions",
                "url": (
                    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/"
                    "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/MultiHopRAG.json"
                ),
                "sha256": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
                "size_bytes": 5171312,
            },
            {
                "role": "corpus",
                "url": (
                    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/"
                    "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/corpus.json"
                ),
                "sha256": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
                "size_bytes": 6785567,
            },
        ],
        "homepage": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
    },
    "ragbench_finqa": {
        "dataset": "galileo-ai/ragbench",
        "config": "finqa",
        "revision": "97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
        "license": "cc-by-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "RAGBench, Galileo Technologies Inc. (2024)",
        "dataset_card": {
            "url": (
                "https://huggingface.co/datasets/galileo-ai/ragbench/raw/"
                "97808f3e5fd16ede40bbff6c2949af8139b2eb7b/README.md"
            ),
            "sha256": "1afeca59575e9508432990054fd75b18b851a384376e2dd7e83852655f33cbe5",
            "size_bytes": 24681,
        },
        "artifacts": [{
            "role": "test",
            "url": (
                "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/"
                "97808f3e5fd16ede40bbff6c2949af8139b2eb7b/"
                "finqa/test-00000-of-00001.parquet"
            ),
            "sha256": "b55bd373741fbe65933191690267421183f898c8253f5db2a4a4703360ecf94d",
            "size_bytes": 8936570,
        }],
        "homepage": "https://huggingface.co/datasets/galileo-ai/ragbench",
    },
}

_MULTIHOP_TYPE_SPECS = {
    "inference_query": {"count": 6, "category": "多文档组合", "answerable": True},
    "comparison_query": {"count": 6, "category": "跨文档比较", "answerable": True},
    "temporal_query": {"count": 6, "category": "时间条件", "answerable": True},
    "null_query": {"count": 6, "category": "无答案", "answerable": False},
}


def _artifact(source_name: str, role: str) -> dict[str, Any]:
    matches = [
        artifact for artifact in SOURCES[source_name]["artifacts"]
        if artifact["role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(f"{source_name}: expected exactly one {role!r} artifact")
    return matches[0]


def _download_verified(artifact: dict[str, Any], *, timeout: float = 120.0) -> bytes:
    """Download one pinned artifact and fail closed on transport or content drift."""
    url = str(artifact["url"])
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError(f"untrusted candidate artifact URL: {url}")
    expected_size = int(artifact["size_bytes"])
    if expected_size <= 0:
        raise ValueError(f"invalid expected artifact size for {url}: {expected_size}")
    request = urllib.request.Request(url, headers={"User-Agent": "modular-rag-eval/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(expected_size + 1)
    if len(payload) != expected_size:
        raise ValueError(
            f"artifact size mismatch for {url}: expected {expected_size}, got {len(payload)}"
        )
    expected_hash = str(artifact["sha256"]).lower()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"artifact SHA-256 mismatch for {url}: expected {expected_hash}, got {actual_hash}"
        )
    return payload


def _verify_dataset_card(source_name: str) -> None:
    source = SOURCES[source_name]
    card = source["dataset_card"]
    payload = _download_verified(card)
    text = payload.decode("utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{source_name}: dataset card has no YAML front matter")
    metadata = yaml.safe_load(parts[1]) or {}
    declared = metadata.get("license")
    if isinstance(declared, list):
        declared = declared[0] if len(declared) == 1 else declared
    expected = card.get("declared_license", source["license"])
    if declared != expected:
        raise ValueError(
            f"{source_name}: dataset card license drifted: expected {expected!r}, "
            f"got {declared!r}"
        )


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


def _stable_row_identity(row: dict[str, Any]) -> str:
    """Return a source-stable identity even when an upstream dataset has no ID."""
    for key in ("id", "query_id", "query", "question"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("candidate source row has no stable identity field")


def _stable_rows(
    frame: pd.DataFrame, count: int, *, salt: str, predicate=None,
) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    if predicate is not None:
        records = [row for row in records if predicate(row)]
    records.sort(key=lambda row: hashlib.sha256(
        f"{salt}:{_stable_row_identity(row)}".encode()
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

    def add_cmrc(self, frame: pd.DataFrame, count: int = 22) -> None:
        for index, row in enumerate(_stable_rows(frame, count, salt="cmrc2018")):
            answers = [str(value) for value in _sequence(row["answers"]["text"]) if str(value)]
            document = self.add_document("cmrc2018", str(row["id"]), "CMRC 2018 passage", str(row["context"]))
            self.cases.append(self._case(
                source="cmrc2018", index=index, raw_id=str(row["id"]),
                query=str(row["question"]), category="精确事实",
                expected=[document.document_id], key_points=answers[:1], language="zh",
                difficulty="easy", source_path=document.path,
            ))

    def add_xquad_zh(self, frame: pd.DataFrame, count: int = 16) -> None:
        """Add human-translated Chinese XQuAD passages and answer spans."""
        for index, row in enumerate(_stable_rows(frame, count, salt="xquad-zh")):
            answers = [
                str(value) for value in _sequence(row["answers"]["text"])
                if str(value)
            ]
            if not answers:
                raise ValueError(f"XQuAD row {row['id']} unexpectedly has no answer")
            document = self.add_document(
                "xquad_zh", str(row["id"]), "XQuAD Chinese passage",
                str(row["context"]),
            )
            self.cases.append(self._case(
                source="xquad_zh", index=index, raw_id=str(row["id"]),
                query=str(row["question"]), category="精确事实",
                expected=[document.document_id], key_points=answers[:1], language="zh",
                difficulty="medium", source_path=document.path,
            ))

    def add_squad(
        self, frame: pd.DataFrame, answerable: int = 8, unanswerable: int = 8,
    ) -> None:
        def has_answer(row: dict[str, Any]) -> bool:
            return bool(_sequence(row["answers"]["text"]))

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

    def add_hotpot(self, frame: pd.DataFrame, count: int = 14) -> None:
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

    def add_multihop(self, questions: pd.DataFrame, corpus: pd.DataFrame) -> None:
        """Add MultiHop-RAG candidates using its official type and evidence labels."""
        corpus_by_url: dict[str, dict[str, Any]] = {}
        for row in corpus.to_dict(orient="records"):
            url = str(row.get("url") or "").strip()
            if not url:
                raise ValueError("MultiHop-RAG corpus contains a document without a URL")
            if url in corpus_by_url:
                raise ValueError(f"MultiHop-RAG corpus contains duplicate URL: {url}")
            corpus_by_url[url] = row

        for question_type, type_spec in _MULTIHOP_TYPE_SPECS.items():
            selected = _stable_rows(
                questions,
                type_spec["count"],
                salt=f"multihop-rag-{question_type}",
                predicate=lambda row, expected=question_type: (
                    str(row.get("question_type") or "") == expected
                ),
            )
            for index, row in enumerate(selected):
                query = str(row["query"]).strip()
                raw_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
                documents: list[Document] = []
                seen_urls: set[str] = set()
                for evidence in _sequence(row.get("evidence_list")):
                    if not isinstance(evidence, dict):
                        raise ValueError(f"{raw_id}: evidence entry is not an object")
                    url = str(evidence.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    corpus_row = corpus_by_url.get(url)
                    if corpus_row is None:
                        raise ValueError(f"{raw_id}: evidence URL is absent from corpus: {url}")
                    title = str(corpus_row.get("title") or url)
                    metadata_lines = [
                        f"Source: {str(corpus_row.get('source') or '').strip()}",
                        f"Published at: {str(corpus_row.get('published_at') or '').strip()}",
                        f"URL: {url}",
                    ]
                    author = str(corpus_row.get("author") or "").strip()
                    if author:
                        metadata_lines.insert(1, f"Author: {author}")
                    documents.append(self.add_document(
                        "multihop_rag",
                        url,
                        title,
                        "\n".join(metadata_lines) + "\n\n" + str(corpus_row.get("body") or ""),
                    ))

                answerable = bool(type_spec["answerable"])
                if answerable and not documents:
                    raise ValueError(f"{raw_id}: answerable MultiHop-RAG case has no evidence")
                if not answerable and documents:
                    raise ValueError(f"{raw_id}: null MultiHop-RAG case unexpectedly has evidence")
                # MultiHop-RAG null queries intentionally have no supporting
                # evidence.  ``manual-negative`` is the schema sentinel for
                # that condition; a dataset homepage is provenance, not a
                # corpus document and must never be treated as ingestible input.
                source_path = documents[0].path if documents else "manual-negative"
                case = self._case(
                    source="multihop_rag",
                    index=index,
                    raw_id=raw_id,
                    query=query,
                    category=str(type_spec["category"]),
                    expected=[document.document_id for document in documents],
                    key_points=[str(row["answer"]).strip()] if answerable else [],
                    language="en",
                    difficulty="hard",
                    source_path=source_path,
                    answerable=answerable,
                )
                case["candidate_metadata"].update({
                    "upstream_question_type": question_type,
                    "category_assignment": "upstream_question_type",
                })
                self.cases.append(case)

    def add_ragbench_finqa(self, frame: pd.DataFrame, count: int = 8) -> None:
        """Add table/numerical candidates with RAGBench sentence support labels."""
        def eligible(row: dict[str, Any]) -> bool:
            return (
                str(row.get("generation_model_name") or "") == "gpt-3.5-turbo-0125"
                and bool(row.get("adherence_score"))
                and not _sequence(row.get("unsupported_response_sentence_keys"))
                and bool(_sequence(row.get("all_relevant_sentence_keys")))
                and float(row.get("completeness_score") or 0.0) == 1.0
                and bool(_sequence(row.get("documents")))
            )

        for index, row in enumerate(_stable_rows(
            frame, count, salt="ragbench-finqa", predicate=eligible,
        )):
            raw_id = str(row["id"])
            documents = [
                self.add_document(
                    "ragbench_finqa",
                    f"{raw_id}-document-{document_index}",
                    f"RAGBench FinQA {raw_id} document {document_index}",
                    str(text),
                )
                for document_index, text in enumerate(_sequence(row["documents"]))
            ]
            relevant_indices = {
                int(match.group(1))
                for key in _sequence(row["all_relevant_sentence_keys"])
                if (match := re.match(r"^(\d+)", str(key))) is not None
            }
            if not relevant_indices:
                raise ValueError(f"{raw_id}: RAGBench FinQA has no mappable evidence key")
            if max(relevant_indices) >= len(documents):
                raise ValueError(f"{raw_id}: RAGBench FinQA evidence index is out of range")
            expected = [documents[position].document_id for position in sorted(relevant_indices)]
            case = self._case(
                source="ragbench_finqa",
                index=index,
                raw_id=raw_id,
                query=str(row["question"]),
                category="表格",
                expected=expected,
                key_points=[str(row["response"])],
                language="en",
                difficulty="hard",
                source_path=documents[min(relevant_indices)].path,
            )
            case["candidate_metadata"].update({
                "upstream_config": "finqa",
                "category_assignment": "upstream_dataset_config",
                "answer_key_origin": "model_response_requires_human_review",
                "generation_model_name": str(row["generation_model_name"]),
            })
            self.cases.append(case)

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
        expected_targets: set[Path] = set()
        for document in self.documents.values():
            target = self.output_root.parent.parent / document.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(document.content.encode("utf-8"))
            expected_targets.add(target.resolve())
        self._prune_stale_corpus(expected_targets)
        sources = [{"id": name, **spec} for name, spec in SOURCES.items()]
        manifest = {
            "artifact_type": "golden_set_candidate",
            "name": "modular-rag-public-golden-candidate",
            "version": "2026.09.02.3",
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

    def _prune_stale_corpus(self, expected_targets: set[Path]) -> None:
        """Remove only stale Markdown files owned by this generated corpus."""
        if not self.corpus_root.exists():
            return
        corpus_root = self.corpus_root.resolve()
        for candidate in self.corpus_root.rglob("*.md"):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(corpus_root):
                raise ValueError(f"generated corpus path escaped its root: {candidate}")
            if resolved not in expected_targets:
                candidate.unlink()
        directories = sorted(
            (path for path in self.corpus_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()

    def _validate(self) -> None:
        if len(self.cases) != 100:
            raise ValueError(f"candidate pack must contain 100 cases, got {len(self.cases)}")
        query_ids = [case["query_id"] for case in self.cases]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("candidate query IDs are not unique")
        for name, spec in SOURCES.items():
            if spec["license"] not in LICENSE_ALLOWLIST:
                raise ValueError(f"{name}: license is not allowlisted: {spec['license']}")
            if not spec.get("attribution") or not spec.get("license_url"):
                raise ValueError(f"{name}: attribution and license URL are required")
            revision = str(spec["revision"])
            if len(revision) != 40 or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"{name}: revision must be a full commit SHA")
            artifacts = [spec["dataset_card"], *spec["artifacts"]]
            for artifact in artifacts:
                url = str(artifact["url"])
                if revision not in url:
                    raise ValueError(f"{name}: artifact URL is not revision pinned: {url}")
                if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
                    raise ValueError(f"{name}: invalid artifact SHA-256")
                if int(artifact["size_bytes"]) <= 0:
                    raise ValueError(f"{name}: invalid artifact size")
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
    for source_name in SOURCES:
        _verify_dataset_card(source_name)
    frames = {
        "cmrc2018": pd.read_parquet(BytesIO(_download_verified(
            _artifact("cmrc2018", "validation")
        ))),
        "xquad_zh": pd.read_parquet(BytesIO(_download_verified(
            _artifact("xquad_zh", "validation")
        ))),
        "squad_v2": pd.read_parquet(BytesIO(_download_verified(
            _artifact("squad_v2", "validation")
        ))),
        "hotpotqa": pd.read_parquet(BytesIO(_download_verified(
            _artifact("hotpotqa", "validation")
        ))),
        "ragbench_finqa": pd.read_parquet(BytesIO(_download_verified(
            _artifact("ragbench_finqa", "test")
        ))),
    }
    builder.add_cmrc(frames["cmrc2018"])
    builder.add_xquad_zh(frames["xquad_zh"])
    builder.add_squad(frames["squad_v2"])
    builder.add_hotpot(frames["hotpotqa"])
    builder.add_multihop(
        pd.read_json(BytesIO(_download_verified(
            _artifact("multihop_rag", "questions")
        ))),
        pd.read_json(BytesIO(_download_verified(
            _artifact("multihop_rag", "corpus")
        ))),
    )
    builder.add_ragbench_finqa(frames["ragbench_finqa"])
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
