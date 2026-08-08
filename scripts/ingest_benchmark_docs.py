"""Ingest sample documents for benchmark evaluation (Stage K).

Ingests the 4 sample PDFs into the 'benchmark' collection with LLM enrichment
disabled for speed.  Run once before scripts/run_benchmark.py --skip-ingest.

Usage:
    python scripts/ingest_benchmark_docs.py
    python scripts/ingest_benchmark_docs.py --collection my_col
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="benchmark")
    p.add_argument("--fast", action="store_true", default=True,
                   help="Disable LLM enrichment (default: True)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.core.settings import load_settings
    settings = load_settings()

    # Optionally disable LLM-based enrichment for speed
    if args.fast:
        if hasattr(settings, "ingestion") and settings.ingestion:
            if isinstance(settings.ingestion.chunk_refiner, dict):
                settings.ingestion.chunk_refiner["use_llm"] = False
            if isinstance(settings.ingestion.metadata_enricher, dict):
                settings.ingestion.metadata_enricher["use_llm"] = False
        print("⚡ Fast mode: LLM enrichment disabled")

    from src.ingestion.pipeline import IngestionPipeline

    collection = args.collection
    doc_dir = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents"

    target_docs = [
        "complex_technical_doc.pdf",
        "chinese_technical_doc.pdf",
        "chinese_table_chart_doc.pdf",
        "chinese_long_doc.pdf",
    ]

    print(f"\n📥 Ingesting {len(target_docs)} documents into collection '{collection}' ...")
    pipeline = IngestionPipeline(settings=settings, collection=collection, force=True)

    success, failed = 0, 0
    for name in target_docs:
        fp = doc_dir / name
        if not fp.exists():
            print(f"  ⚠️  {name} not found — skipping")
            failed += 1
            continue
        try:
            print(f"  📄 Ingesting {name} ...", end=" ", flush=True)
            result = pipeline.run(str(fp))
            chunks = getattr(result, "chunk_count", 0)
            ok = getattr(result, "success", True)
            if ok:
                print(f"✅ {chunks} chunks")
                success += 1
            else:
                msg = getattr(result, "error_message", "unknown error")
                print(f"⚠️  {msg} (chunks={chunks})")
                failed += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  ✅ Success: {success}   ❌ Failed: {failed}")
    print(f"  Run: python scripts/run_benchmark.py --skip-ingest")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
