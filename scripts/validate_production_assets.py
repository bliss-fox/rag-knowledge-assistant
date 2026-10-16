#!/usr/bin/env python
"""Validate versioned prompts and public/private golden-set schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.core.settings import resolve_path
from src.production.evaluation import load_golden_dataset
from src.production.prompts import PromptRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[
            "evaluation/public_golden.json",
            "evaluation/candidates/public_golden_candidate.json",
        ],
    )
    args = parser.parse_args()
    registry = PromptRegistry(resolve_path("config/prompts"))
    print(f"Validated {len(registry.manifest())} versioned prompts")
    for value in args.datasets:
        dataset_path = resolve_path(value)
        dataset = load_golden_dataset(dataset_path)
        print(f"Validated {dataset.name} {dataset.version}: {len(dataset.cases)} cases ({dataset.sha256})")
        if dataset.visibility == "public":
            raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            raw_corpus = raw.get("corpus", [])
            if not isinstance(raw_corpus, list):
                raise ValueError("public corpus manifest must be a list")
            corpus: dict[str, dict[str, object]] = {}
            declared_paths: set[Path] = set()
            repository_root = resolve_path(".")
            for index, item in enumerate(raw_corpus):
                if not isinstance(item, dict):
                    raise ValueError(f"public corpus item[{index}] must be an object")
                document_id = str(item.get("document_id", "")).strip()
                path_value = str(item.get("path", "")).strip()
                if not document_id or not path_value:
                    raise ValueError(
                        f"public corpus item[{index}] needs document_id and path"
                    )
                if document_id in corpus:
                    raise ValueError(f"duplicate public corpus document_id: {document_id}")
                source = resolve_path(path_value)
                if source in declared_paths:
                    raise ValueError(f"duplicate public corpus path: {path_value}")
                if repository_root != source and repository_root not in source.parents:
                    raise ValueError(f"public corpus path is outside repository: {source}")
                if not source.is_file():
                    raise ValueError(f"public corpus file does not exist: {source}")
                expected_id = f"doc_{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}"
                if expected_id != document_id:
                    raise ValueError(
                        f"public corpus document hash mismatch: {document_id}"
                    )
                corpus[document_id] = item
                declared_paths.add(source)
            for case in raw["cases"]:
                if corpus:
                    for document_id in case["expected_document_ids"]:
                        if document_id not in corpus:
                            raise ValueError(
                                f"{case['query_id']}: missing corpus item {document_id}"
                            )
                    if case["source"] != "manual-negative":
                        if resolve_path(case["source"]) not in declared_paths:
                            raise ValueError(
                                f"{case['query_id']}: source is not declared in public corpus"
                            )
                    continue
                if not case["answerable"] or case["source"] == "manual-negative":
                    continue
                source = resolve_path(case["source"])
                if not source.is_file():
                    raise ValueError(f"{case['query_id']}: public source does not exist: {source}")
                expected_id = f"doc_{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}"
                if expected_id not in case["expected_document_ids"]:
                    raise ValueError(
                        f"{case['query_id']}: expected_document_ids must include {expected_id}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
