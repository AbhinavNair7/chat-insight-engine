"""Stage 4 — ask questions of the indexed chat.

Three strategies:
    rag   — embed the question, retrieve the top-N chunks, answer with mandatory [chunk_id] citations
    scan  — map-reduce over every chunk (see scan.py) for corpus-wide questions
    auto  — an LLM router picks rag or scan; a rag answer whose best match is weak
            escalates to scan instead of answering from irrelevant context
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import dataclass

import chromadb

from chat_insight import config, prompts
from chat_insight.embeddings import embed_texts
from chat_insight.llm import ChatFn, chat
from chat_insight.scan import scan_corpus
from chat_insight.store import open_collection

POOR_MATCH_DISTANCE = 0.65  # cosine distance above which retrieved context is unlikely to be relevant
FIND_CONTEXT_WINDOW = 15

log = logging.getLogger(__name__)


# ── Retrieval and grounded answers ────────────────────────────────────────────


def retrieve(collection: chromadb.Collection, query_vec: list[float], n_results: int) -> list[dict]:
    """Top-N chunks as ``{id, document, metadata, distance}``, best first."""
    res = collection.query(
        query_embeddings=[query_vec],  # type: ignore[arg-type]
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
    )
    return [
        {"id": cid, "document": doc, "metadata": dict(meta), "distance": dist}
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            (res["documents"] or [[]])[0],
            (res["metadatas"] or [[]])[0],
            (res["distances"] or [[]])[0],
        )
    ]


def build_user_message(query: str, chunks: list[dict]) -> str:
    """Retrieved chunks, each labelled with its id so the model can cite it, then the question."""
    lines = ["## Retrieved chunks\n"]
    for chunk in chunks:
        meta = chunk["metadata"]
        lines.append(
            f"### [{chunk['id']}]  ({meta.get('start_ts', '')} → {meta.get('end_ts', '')}, "
            f"{meta.get('n_messages', '?')} messages)"
        )
        lines.append(chunk["document"])
        lines.append("")
    lines += ["---", f"## Query\n{query}"]
    return "\n".join(lines)


def grounded_answer(
    query: str, chunks: list[dict], model: str, *, corpus: str = config.CORPUS_DESCRIPTION, llm: ChatFn = chat
) -> str:
    return llm(
        [
            {"role": "system", "content": prompts.GROUNDED_ANSWER.format(corpus=corpus)},
            {"role": "user", "content": build_user_message(query, chunks)},
        ],
        model,
    )


def route_question(query: str, model: str, *, corpus: str = config.CORPUS_DESCRIPTION, llm: ChatFn = chat) -> str:
    """Return "scan" for corpus-wide questions, otherwise "rag"."""
    verdict = llm(
        [
            {"role": "system", "content": prompts.ROUTER.format(corpus=corpus)},
            {"role": "user", "content": query},
        ],
        model,
    )
    return "scan" if "BROAD" in verdict.upper() else "rag"


def match_label(dist: float) -> str:
    if dist < 0.35:
        return "(strong match)"
    if dist < 0.50:
        return "(good match)"
    if dist < POOR_MATCH_DISTANCE:
        return "(weak match)"
    return "(poor match — retrieval may be irrelevant)"


# ── Orchestration ─────────────────────────────────────────────────────────────


@dataclass
class QueryOptions:
    n_results: int = config.TOP_N
    chat_model: str = config.CHAT_MODEL
    router_model: str = config.MAP_MODEL
    embedding_model: str = config.EMBEDDING_MODEL
    local_model: str = config.LOCAL_EMBEDDING_MODEL
    use_local: bool = False
    mode: str = "rag"  # rag | scan | auto
    dry_run: bool = False
    show_chunks: bool = False


def handle_query(query: str, collection: chromadb.Collection, opts: QueryOptions, *, llm: ChatFn = chat) -> None:
    query = query.strip()
    if not query:
        return

    mode = opts.mode
    if mode == "auto" and not opts.dry_run:
        mode = route_question(query, opts.router_model, llm=llm)
        print(f"\n  Router → {'full-corpus scan' if mode == 'scan' else 'targeted retrieval'}")

    if mode == "scan" and not opts.dry_run:
        _run_scan(query, collection, opts, llm)
        return

    q_vec = embed_texts(
        [query], use_local=opts.use_local, openai_model=opts.embedding_model, local_model=opts.local_model
    )[0]
    chunks = retrieve(collection, q_vec, opts.n_results)
    if not chunks:
        print("  No results found.")
        return

    best = chunks[0]["distance"]
    if best > POOR_MATCH_DISTANCE:
        if opts.mode == "auto" and not opts.dry_run:
            print(f"  Best match distance {best:.2f} is poor → escalating to full-corpus scan")
            _run_scan(query, collection, opts, llm)
            return
        print(
            f"\n  ⚠  Best match distance = {best:.4f}; retrieved chunks may be irrelevant.\n"
            "     Try more specific terms, or --scan / --auto for broad questions.\n"
        )

    if opts.dry_run or opts.show_chunks:
        print_retrieved_chunks(chunks)
    if opts.dry_run:
        return

    print_answer(query, grounded_answer(query, chunks, opts.chat_model, llm=llm), chunks)


def _run_scan(query: str, collection: chromadb.Collection, opts: QueryOptions, llm: ChatFn) -> None:
    result = scan_corpus(collection, query, chat_model=opts.chat_model, llm=llm)
    if result is None:
        print("\n  No relevant content found anywhere in the corpus.\n")
        return
    print(f"{'═' * 60}\n  SCAN RESULT: {query}\n{'═' * 60}\n")
    print(result)
    print(f"\n{'─' * 60}\n")


# ── Output ────────────────────────────────────────────────────────────────────


def print_retrieved_chunks(chunks: list[dict]) -> None:
    print(f"\n{'─' * 60}\n  Retrieved {len(chunks)} chunk(s)\n{'─' * 60}")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        dist = chunk["distance"]
        print(
            f"\n  {i}. [{chunk['id']}]  dist={dist:.4f} {match_label(dist)}  "
            f"{str(meta.get('start_ts', ''))[:19]} → {str(meta.get('end_ts', ''))[:19]}\n"
        )
        for line in chunk["document"].splitlines():
            print(f"    {line}")
    print()


def print_answer(query: str, answer: str, chunks: list[dict]) -> None:
    print(f"\n{'═' * 60}\n  Query : {query}\n  Chunks: {'  '.join(c['id'] for c in chunks)}\n{'═' * 60}\n")
    print(answer)
    print(f"\n{'─' * 60}\n")


# ── Quote finder (no LLM, no index) ───────────────────────────────────────────


def find_quote_context(quote: str, csv_path: str, chunks_path: str, window: int = FIND_CONTEXT_WINDOW) -> int:
    """Print every message containing *quote* with surrounding conversation and its covering chunk id."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except FileNotFoundError:
        print(f"ERROR: CSV not found at '{csv_path}'. Run:  chat-insight export", file=sys.stderr)
        return 1

    hits = df.index[df["message"].str.contains(quote, case=False, regex=False)].tolist()
    if not hits:
        print(f"\n  No messages contain {quote!r}. Try a shorter fragment.\n")
        return 0

    spans: list[tuple[str, str, str]] = []
    try:
        with open(chunks_path, encoding="utf-8") as f:
            spans = [(c["id"], c["meta"]["start_ts"], c["meta"]["end_ts"]) for c in json.load(f)]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass  # chunk attribution is a nicety; the context still prints

    print(f"\n  Found {len(hits)} match(es) for {quote!r}\n")
    for n, idx in enumerate(hits, 1):
        ts = df.at[idx, "timestamp"]
        chunk_id = next((cid for cid, start, end in spans if start <= ts <= end), "(unknown)")
        print(f"{'═' * 60}\n  Match {n}/{len(hits)} · {ts[:19]} · {chunk_id}\n{'═' * 60}\n")
        for row_idx in range(max(0, idx - window), min(len(df), idx + window + 1)):
            row = df.iloc[row_idx]
            marker = "  ▶ " if row_idx == idx else "    "
            print(f"{marker}[{row['timestamp'][:19]}] {row['sender'][:20]}")
            for line in textwrap.wrap(row["message"], width=78):
                print(f"        {line}")
            print()
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-insight ask",
        description="Ask grounded, cited questions of the indexed chat (REPL if --query is omitted).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              chat-insight ask -q "Which ambient scribe tools have people trialled?"
              chat-insight ask --scan -q "What are the top recurring frustrations?"
              chat-insight ask --auto            # router picks retrieval or full scan
              chat-insight ask --dry-run -q "EPR integration"   # inspect retrieval only
              chat-insight ask --find-quote "printing and re-scanning"
        """),
    )
    p.add_argument("--query", "-q", default=None, metavar="TEXT")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--scan", action="store_true", help="Map-reduce over every chunk")
    mode.add_argument("--auto", action="store_true", help="LLM router chooses retrieval or scan")
    p.add_argument("--n-results", "-n", type=int, default=config.TOP_N, metavar="N")
    p.add_argument("--model", default=config.CHAT_MODEL, metavar="MODEL", help="Answer/synthesis model")
    p.add_argument("--embedding-model", default=config.EMBEDDING_MODEL, metavar="MODEL")
    p.add_argument("--local-model", default=config.LOCAL_EMBEDDING_MODEL, metavar="MODEL")
    p.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Embed queries locally (must match the model used to build the index)",
    )
    p.add_argument("--chroma-dir", default=config.CHROMA_DIR, metavar="PATH")
    p.add_argument("--collection", default=config.COLLECTION, metavar="NAME")
    p.add_argument("--dry-run", action="store_true", help="Show retrieved chunks and distances; no LLM call")
    p.add_argument("--show-chunks", action="store_true", help="Show retrieved chunks before the answer")
    p.add_argument(
        "--find-quote",
        metavar="TEXT",
        default=None,
        help="Find a phrase in the raw CSV and print the surrounding conversation",
    )
    p.add_argument("--context-window", type=int, default=FIND_CONTEXT_WINDOW, metavar="N")
    p.add_argument("--csv", default=config.EXPORT_CSV, metavar="PATH")
    p.add_argument("--chunks-file", default=config.CHUNKS_PATH, metavar="PATH")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s  %(message)s",
    )

    if args.find_quote:
        return find_quote_context(args.find_quote, args.csv, args.chunks_file, args.context_window)

    try:
        collection = open_collection(args.chroma_dir, args.collection)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    opts = QueryOptions(
        n_results=args.n_results,
        chat_model=args.model,
        embedding_model=args.embedding_model,
        local_model=args.local_model,
        use_local=args.local_embeddings,
        mode="scan" if args.scan else "auto" if args.auto else "rag",
        dry_run=args.dry_run,
        show_chunks=args.show_chunks,
    )

    def safe_handle(q: str) -> bool:
        try:
            handle_query(q, collection, opts)
            return True
        except Exception as exc:
            print(f"ERROR: query failed: {exc}", file=sys.stderr)
            return False

    if args.query:
        return 0 if safe_handle(args.query) else 1

    print(f"\nchat-insight · {collection.count()} chunks · mode={opts.mode} · model={opts.chat_model}")
    print("Type a question and press Enter. Empty line or Ctrl-C to exit.\n")
    try:
        while query := input("Query> ").strip():
            safe_handle(query)
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
    return 0
