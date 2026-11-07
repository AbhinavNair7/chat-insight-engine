"""Stage 2 — slice the exported CSV into rolling conversation windows.

Group chat meaning lives in exchanges, not single messages, so each chunk is a
window of consecutive messages. Each chunk looks like:

    {
        "id":   "chunk_0042",
        "text": "[2026-01-01T09:00:00+00:00] member_a: Hello\\n...",
        "meta": {"chunk_id": 42, "chat_jid": "...", "start_ts": "...", "end_ts": "...",
                 "participants": [...], "n_messages": 30}
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from chat_insight import config

REQUIRED_COLUMNS = {"timestamp", "chat_jid", "sender", "message", "message_id"}

log = logging.getLogger(__name__)


def load_csv(path: str, limit: int | None = None) -> pd.DataFrame:
    """Load the export CSV as strings; *limit* keeps only the last N rows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Export CSV not found: {p.resolve()}\n  → Run:  chat-insight export")
    df = pd.read_csv(p, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def make_chunks(df: pd.DataFrame, chunk_size: int, stride: int) -> list[dict]:
    """Rolling windows of *chunk_size* rows stepping by *stride*.

    Every row lands in at least one chunk; the final window may be shorter.
    """
    if chunk_size < 1 or stride < 1:
        raise ValueError("chunk_size and stride must be ≥ 1")

    chunks: list[dict] = []
    for chunk_idx, start in enumerate(range(0, len(df), stride)):
        block = df.iloc[start : start + chunk_size]
        text = "\n".join(f"[{row.timestamp}] {row.sender}: {row.message}" for row in block.itertuples(index=False))
        chunks.append(
            {
                "id": f"chunk_{chunk_idx:04d}",
                "text": text,
                "meta": {
                    "chunk_id": chunk_idx,
                    "chat_jid": block["chat_jid"].iloc[0],
                    "start_ts": block["timestamp"].iloc[0],
                    "end_ts": block["timestamp"].iloc[-1],
                    "participants": sorted(block["sender"].unique().tolist()),
                    "n_messages": len(block),
                },
            }
        )
    return chunks


def write_chunks(chunks: list[dict], output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d chunks → %s", len(chunks), out.resolve())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-insight chunk",
        description="Chunk the exported CSV into rolling conversation windows.",
    )
    p.add_argument("--input", "-i", default=config.EXPORT_CSV, help="Input CSV (default: %(default)s)")
    p.add_argument("--out", "-o", default=config.CHUNKS_PATH, help="Output JSON (default: %(default)s)")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_SIZE,
        metavar="N",
        help="Messages per chunk (default: %(default)s)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=config.STRIDE,
        metavar="N",
        help="Step between windows; < chunk-size gives overlap (default: %(default)s)",
    )
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Use only the last N rows")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s  %(message)s",
    )
    if args.chunk_size < 1:
        log.error("--chunk-size must be ≥ 1")
        return 1
    if args.stride < 1:
        log.error("--stride must be ≥ 1")
        return 1

    try:
        df = load_csv(args.input, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    log.info("Loaded %d messages from %s", len(df), args.input)

    chunks = make_chunks(df, args.chunk_size, args.stride)
    if not chunks:
        log.warning("No chunks produced — is the CSV empty?")
        return 1
    log.info("Created %d chunks (chunk_size=%d, stride=%d)", len(chunks), args.chunk_size, args.stride)

    first = chunks[0]
    meta = first["meta"]
    people = meta["participants"]
    print(f"\n── First chunk ({first['id']}) ──")
    print(f"  chat_jid : {meta['chat_jid']}")
    print(f"  start_ts : {meta['start_ts']}")
    print(f"  end_ts   : {meta['end_ts']}")
    print(f"  messages : {meta['n_messages']}")
    print(f"  participants ({len(people)}): {', '.join(people[:5])}" + (" …" if len(people) > 5 else ""))
    print()
    for line in first["text"].splitlines()[:5]:
        print(f"  {line[:100]}")
    if meta["n_messages"] > 5:
        print(f"  … ({meta['n_messages'] - 5} more lines)")
    print()

    write_chunks(chunks, args.out)
    print(f"✓  {len(chunks)} chunks → {args.out}")
    return 0
