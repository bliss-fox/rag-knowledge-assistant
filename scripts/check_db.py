"""Quick diagnostic: check what's in ChromaDB collections."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import load_settings
import chromadb

settings = load_settings()
persist_dir = settings.vector_store.persist_directory
print(f"ChromaDB persist dir: {persist_dir}")

client = chromadb.PersistentClient(path=persist_dir)
collections = client.list_collections()
print(f"Found {len(collections)} collection(s):")
for col in collections:
    c = client.get_collection(col.name)
    n = c.count()
    print(f"  - {col.name}: {n} chunks")
    if n > 0:
        sample = c.peek(limit=2)
        for i, (cid, doc, meta) in enumerate(zip(
            sample.get("ids", []),
            sample.get("documents", []),
            sample.get("metadatas", []),
        )):
            src = meta.get("source_path", meta.get("source", "?")) if meta else "?"
            print(f"    [{i}] id={cid[:20]}... source={Path(src).name if src else '?'}")
