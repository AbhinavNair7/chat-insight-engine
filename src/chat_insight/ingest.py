"""Stage 3 — embed new or changed chunks and upsert them into Chroma.

Idempotent: manifest.json records a SHA-256 of each embedded chunk's text plus
the embedding model, so re-runs only pay for chunks whose text or model changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from chat_insight import config
from chat_insight.embeddings import embed_texts
from chat_insight.store import delete_collection, get_or_create_collection, upsert_chunks

DEFAULT_BATCH_SIZE = 64

log = logging.getLogger(__name__)


def chunk_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Manifest ──────────────────────────────────────────────────────────────────


def load_manifest(path: str) -> dict:
    """``{"embedded_chunks": {chunk_id: sha256}, "embedding_model": str | None}``."""
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {"embedded_chunks": {}, "embedding_model": None}


def save_manifest(manifest: dict, path: str) -> None:
    """Write atomically (temp file then rename) so a crash never leaves a half-written manifest."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


# ── Chunks ────────────────────────────────────────────────────────────────────


def load_chunks(path: str, sample: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {p.resolve()}\n  → Run:  chat-insight chunk")
    with p.open(encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    for key in ("id", "text", "meta"):
        if chunks and key not in chunks[0]:
            raise ValueError(f"chunk items must have key '{key}'")
    if sample is not None:
        chunks = chunks[:sample]
    return chunks


def filter_new_chunks(chunks: list[dict], manifest: dict, embedding_model: str) -> tuple[list[dict], list[dict]]:
    """Split into (to_embed, to_skip). A change of text or embedding model forces a re-embed."""
    embedded = manifest.get("embedded_chunks", {})
    previous_model = manifest.get("embedding_model")
    model_changed = bool(previous_model) and previous_model != embedding_model

    to_embed: list[dict] = []
    to_skip: list[dict] = []
    for chunk in chunks:
        unchanged = embedded.get(chunk["id"]) == chunk_checksum(chunk["text"])
        (to_skip if unchanged and not model_changed else to_embed).append(chunk)
    return to_embed, to_skip


def embed_in_batches(
    chunks: list[dict], batch_size: int, use_local: bool, openai_model: str, local_model: str
) -> list[list[float]]:
    label = local_model if use_local else openai_model
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(chunks), batch_size), desc=f"Embedding ({label})", unit="batch"):
        texts = [c["text"] for c in chunks[start : start + batch_size]]
        vectors.extend(embed_texts(texts, use_local=use_local, openai_model=openai_model, local_model=local_model))
    return vectors


# ── Commands ──────────────────────────────────────────────────────────────────


def run(
    chunks_path: str,
    manifest_path: str,
    persist_dir: str,
    collection_name: str,
    batch_size: int,
    use_local: bool,
    openai_model: str,
    local_model: str,
    sample: int | None,
) -> int:
    try:
        chunks = load_chunks(chunks_path, sample)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    log.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    manifest = load_manifest(manifest_path)
    embedding_model = local_model if use_local else openai_model
    previous_model = manifest.get("embedding_model")
    if previous_model and previous_model != embedding_model:
        # Vectors from different models are not comparable (and may differ in size): rebuild from scratch.
        log.warning("Embedding model changed (%s → %s): rebuilding the index.", previous_model, embedding_model)
        delete_collection(persist_dir, collection_name)
        manifest = {"embedded_chunks": {}, "embedding_model": None}
        save_manifest(manifest, manifest_path)

    if sample is None:
        _prune_orphans(chunks, manifest, manifest_path, persist_dir, collection_name)

    to_embed, to_skip = filter_new_chunks(chunks, manifest, embedding_model)
    if to_skip:
        log.info("Skipping %d already-embedded chunk(s).", len(to_skip))
    if not to_embed:
        print("✓  Nothing new to embed — all chunks are up to date.")
        return 0

    log.info("Embedding %d chunk(s) with '%s'.", len(to_embed), embedding_model)
    try:
        vectors = embed_in_batches(to_embed, batch_size, use_local, openai_model, local_model)
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        return 1

    try:
        upsert_chunks(get_or_create_collection(persist_dir, collection_name), to_embed, vectors)
    except Exception as exc:
        log.error("Chroma upsert failed: %s", exc)
        return 1

    embedded = manifest.setdefault("embedded_chunks", {})
    for chunk in to_embed:
        embedded[chunk["id"]] = chunk_checksum(chunk["text"])
    manifest["embedding_model"] = embedding_model
    save_manifest(manifest, manifest_path)

    print(f"✓  Embedded {len(to_embed)} new chunk(s) ({len(embedded)} total) → {Path(persist_dir).resolve()}")
    return 0


def _prune_orphans(
    chunks: list[dict], manifest: dict, manifest_path: str, persist_dir: str, collection_name: str
) -> None:
    """Remove indexed chunks that no longer exist in the chunks file (e.g. after re-chunking)."""
    current = {c["id"] for c in chunks}
    embedded = manifest.get("embedded_chunks", {})
    orphans = sorted(set(embedded) - current)
    if not orphans:
        return
    get_or_create_collection(persist_dir, collection_name).delete(ids=orphans)
    for cid in orphans:
        del embedded[cid]
    save_manifest(manifest, manifest_path)
    log.info("Removed %d chunk(s) no longer in the chunks file.", len(orphans))


def print_status(chunks_path: str, manifest_path: str) -> int:
    """Report embedded / missing / stale chunks. Exit 1 if anything needs embedding."""
    try:
        chunks = load_chunks(chunks_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    manifest = load_manifest(manifest_path)
    embedded = manifest.get("embedded_chunks", {})
    missing = [c["id"] for c in chunks if c["id"] not in embedded]
    stale = [c["id"] for c in chunks if c["id"] in embedded and embedded[c["id"]] != chunk_checksum(c["text"])]
    ok = len(chunks) - len(missing) - len(stale)

    print("\n── Index status ──")
    print(f"  chunks          : {len(chunks)}")
    print(f"  embedding model : {manifest.get('embedding_model') or 'none yet'}")
    print(f"  ✓ embedded      : {ok}")
    print(f"  ✗ missing       : {len(missing)}")
    print(f"  ⚠ stale         : {len(stale)} (text changed since last embed)")
    for label, ids in (("Missing", missing), ("Stale", stale)):
        if ids:
            print(f"\n  {label}: {', '.join(ids)}")
    if missing or stale:
        print(f"\n  → Run:  chat-insight embed  to index {len(missing) + len(stale)} chunk(s).\n")
        return 1
    print("\n  All chunks are up to date.\n")
    return 0


SMOKE_CHUNKS = [
    "[2025-01-01T09:00:00] member_a: Has anyone deployed AI triage tools yet?",
    "[2025-01-01T09:05:00] member_b: We trialled an ambient scribe for documentation — consent issues came up.",
    "[2025-01-01T09:10:00] member_a: What does the trust say about data governance for AI?",
    "[2025-01-01T09:15:00] member_b: There's real worry about AI replacing clinical judgement.",
]


def smoke_test(chunks_path: str, use_local: bool, openai_model: str, local_model: str) -> int:
    """Embed 4 chunks into a throwaway Chroma dir and run one similarity query."""
    print("\n── Smoke test ──")
    try:
        sample = load_chunks(chunks_path, sample=4)
    except FileNotFoundError:
        log.warning("No chunks file — using synthetic sample chunks.")
        sample = [
            {
                "id": f"smoke_{i:04d}",
                "text": text,
                "meta": {"chunk_id": i, "start_ts": "2025-01-01T09:00:00+00:00", "n_messages": 1},
            }
            for i, text in enumerate(SMOKE_CHUNKS)
        ]

    tmp_dir = tempfile.mkdtemp(prefix="chat_insight_smoke_")
    try:
        vectors = embed_in_batches(sample, len(sample), use_local, openai_model, local_model)
        collection = get_or_create_collection(tmp_dir, "smoke_test")
        upsert_chunks(collection, sample, vectors)
        query = "What are people saying about AI and patient data?"
        q_vec = embed_texts([query], use_local=use_local, openai_model=openai_model, local_model=local_model)
        results = collection.query(query_embeddings=q_vec, n_results=min(3, len(sample)))  # type: ignore[arg-type]
        print(f"  Query: {query!r}")
        for rank, (cid, doc) in enumerate(zip(results["ids"][0], (results["documents"] or [[]])[0]), 1):
            print(f"    {rank}. [{cid}]  {doc[:100].replace(chr(10), ' ')}…")
        print("\n✓  Smoke test passed.\n")
        return 0
    except Exception as exc:
        log.error("Smoke test failed: %s", exc)
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-insight embed",
        description="Embed chunks and upsert them into the local Chroma collection (idempotent).",
    )
    p.add_argument("--chunks", "-c", default=config.CHUNKS_PATH, metavar="PATH")
    p.add_argument("--manifest", "-m", default=config.MANIFEST_PATH, metavar="PATH")
    p.add_argument("--chroma-dir", default=config.CHROMA_DIR, metavar="PATH")
    p.add_argument("--collection", default=config.COLLECTION, metavar="NAME")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, metavar="N")
    p.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Use sentence-transformers instead of OpenAI (no API key needed)",
    )
    p.add_argument("--openai-model", default=config.EMBEDDING_MODEL, metavar="MODEL")
    p.add_argument("--local-model", default=config.LOCAL_EMBEDDING_MODEL, metavar="MODEL")
    p.add_argument("--sample", type=int, default=None, metavar="N", help="Only embed the first N chunks")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Embed 4 chunks into a temporary DB and run a query; touches nothing real",
    )
    p.add_argument("--status", action="store_true", help="Show embedded / missing / stale chunks")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s  %(message)s",
    )
    if args.status:
        return print_status(args.chunks, args.manifest)
    if args.smoke_test:
        return smoke_test(args.chunks, args.local_embeddings, args.openai_model, args.local_model)
    return run(
        chunks_path=args.chunks,
        manifest_path=args.manifest,
        persist_dir=args.chroma_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        use_local=args.local_embeddings,
        openai_model=args.openai_model,
        local_model=args.local_model,
        sample=args.sample,
    )
