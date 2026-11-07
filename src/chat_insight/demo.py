"""Offline end-to-end demo on a bundled synthetic chat: chunk → local embed → Chroma → retrieve.

Needs no API key and no WhatsApp account; everything is written to a temp directory.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from importlib import resources
from pathlib import Path

from chat_insight import config
from chat_insight.chunking import load_csv, make_chunks
from chat_insight.embeddings import embed_local
from chat_insight.query import match_label, retrieve
from chat_insight.store import get_or_create_collection, upsert_chunks

SAMPLE_QUERIES = [
    "What are the concerns about AI replacing clinical judgement?",
    "Which AI tools are being used for clinical documentation?",
    "What is slowing down AI adoption?",
]


def sample_csv_path() -> Path:
    return Path(str(resources.files("chat_insight") / "data" / "synthetic_chat.csv"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chat-insight demo", description=__doc__)
    p.add_argument("--local-model", default=config.LOCAL_EMBEDDING_MODEL, metavar="MODEL")
    args = p.parse_args(argv)

    df = load_csv(str(sample_csv_path()))
    chunks = make_chunks(df, chunk_size=8, stride=4)
    print(f"✓  {len(df)} synthetic messages → {len(chunks)} overlapping chunks (size 8, stride 4)")

    print(f"→  Embedding locally with {args.local_model} (first run downloads ~90 MB)…")
    vectors = embed_local([c["text"] for c in chunks], args.local_model)

    tmp = tempfile.mkdtemp(prefix="chat_insight_demo_")
    try:
        collection = get_or_create_collection(tmp, "demo")
        upsert_chunks(collection, chunks, vectors)
        print(f"✓  Indexed {collection.count()} chunks in a temporary Chroma store\n")
        for query in SAMPLE_QUERIES:
            q_vec = embed_local([query], args.local_model)[0]
            print(f"  Q: {query}")
            for rank, hit in enumerate(retrieve(collection, q_vec, 3), 1):
                first = hit["document"].splitlines()[0]
                print(f"    {rank}. [{hit['id']}] {hit['distance']:.2f} {match_label(hit['distance'])}  {first[:90]}")
            print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("✓  Demo complete — the local retrieval pipeline works. Add OPENAI_API_KEY for cited answers and scans.")
    return 0
