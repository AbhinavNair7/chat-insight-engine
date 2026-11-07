"""Thin wrapper over a persistent local Chroma collection (cosine distance)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)


def _client(persist_dir: str) -> chromadb.ClientAPI:
    # Anonymous usage telemetry is on by default in chromadb; this tool keeps everything local.
    return chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(persist_dir: str, name: str) -> chromadb.Collection:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = _client(persist_dir)
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    log.info("Chroma collection '%s' → %s (%d docs)", name, Path(persist_dir).resolve(), collection.count())
    return collection


def open_collection(persist_dir: str, name: str) -> chromadb.Collection:
    """Open an existing, populated collection or raise RuntimeError with the fix."""
    client = _client(persist_dir)
    try:
        collection = client.get_collection(name=name)
    except Exception as exc:
        raise RuntimeError(f"Collection '{name}' not found in '{persist_dir}'.\n  → Run:  chat-insight embed") from exc
    if collection.count() == 0:
        raise RuntimeError(f"Collection '{name}' is empty.\n  → Run:  chat-insight embed")
    return collection


def delete_collection(persist_dir: str, name: str) -> None:
    """Drop the collection if it exists (used when the embedding model changes)."""
    client = _client(persist_dir)
    if name in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(name)


def flatten_meta(meta: dict) -> dict:
    """Chroma metadata values must be scalars; lists (e.g. participants) become JSON strings."""
    return {k: v if isinstance(v, (str, int, float)) else json.dumps(v, ensure_ascii=False) for k, v in meta.items()}


def upsert_chunks(collection: chromadb.Collection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[flatten_meta(c["meta"]) for c in chunks],
        embeddings=embeddings,  # type: ignore[arg-type]
    )
    log.info("Upserted %d chunks into Chroma.", len(chunks))


def all_chunks_chronological(collection: chromadb.Collection) -> list[tuple[str, str, dict]]:
    """Every (id, document, metadata) in the collection, ordered by start timestamp."""
    data = collection.get(include=["documents", "metadatas"])  # type: ignore[list-item]
    triples = zip(data["ids"], data["documents"] or [], data["metadatas"] or [])
    return sorted(
        ((cid, doc, dict(meta)) for cid, doc, meta in triples),
        key=lambda t: str(t[2].get("start_ts", "")),
    )
