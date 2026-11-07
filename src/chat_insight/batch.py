"""Batch research reports: run every question in a markdown file as a full-corpus scan.

Each question becomes its own markdown report. Existing reports are skipped, so
adding a question and re-running only pays for the new one.

Question file format:
    Query 1 — Short title
    The full question, which may span
    several lines.

    Query 2 — Next title
    ...
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from chat_insight import config
from chat_insight.scan import DEFAULT_BATCH_SIZE, scan_corpus
from chat_insight.store import open_collection

TITLE_RE = re.compile(r"^Query\s+\d+", re.IGNORECASE)


def parse_queries(text: str) -> list[dict]:
    """``[{"title": str, "query": str}, ...]``; titles without a body are dropped."""
    sections: list[dict] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title is not None and body:
            sections.append({"title": title, "query": " ".join(body)})

    for raw in text.splitlines():
        line = raw.strip()
        if TITLE_RE.match(line):
            flush()
            title, body = line, []
        elif line and title is not None:
            body.append(line)
    flush()
    return sections


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", title).strip().rstrip(".")[:200]


def result_path(results_dir: str, title: str) -> Path:
    return Path(results_dir) / f"{safe_filename(title)}.md"


def write_result(path: Path, title: str, query: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(
        f"# {title}\n\n**Query:** {query}\n\n**Generated:** {generated}\n\n---\n\n{answer}\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-insight batch",
        description="Run each question in a markdown file as a full-corpus scan and save one report per question.",
    )
    p.add_argument("--queries", default="examples/questions.md", metavar="PATH")
    p.add_argument("--results-dir", default=f"{config.DATA_DIR}/reports", metavar="PATH")
    p.add_argument("--rerun", action="store_true", help="Regenerate reports that already exist")
    p.add_argument("--model", default=config.CHAT_MODEL, metavar="MODEL", help="Synthesis model")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, metavar="N")
    p.add_argument("--chroma-dir", default=config.CHROMA_DIR, metavar="PATH")
    p.add_argument("--collection", default=config.COLLECTION, metavar="NAME")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s  %(message)s",
    )

    try:
        queries = parse_queries(Path(args.queries).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: questions file not found: '{args.queries}'", file=sys.stderr)
        return 1
    if not queries:
        print(f"ERROR: no questions found in '{args.queries}'.", file=sys.stderr)
        return 1

    pending = [q for q in queries if args.rerun or not result_path(args.results_dir, q["title"]).exists()]
    print(f"\n  {len(queries)} question(s) · {len(queries) - len(pending)} already done · {len(pending)} to run\n")
    if not pending:
        print("  Nothing to do. Use --rerun to regenerate.\n")
        return 0

    try:
        collection = open_collection(args.chroma_dir, args.collection)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failed = 0
    for i, q in enumerate(pending, 1):
        out = result_path(args.results_dir, q["title"])
        print(f"[{i}/{len(pending)}] {q['title']}\n  → {out}\n")
        try:
            answer = scan_corpus(collection, q["query"], chat_model=args.model, batch_size=args.batch_size)
        except Exception as exc:
            # No report on failure, so the next run retries this question instead of skipping it.
            print(f"\n  ✗ Scan failed, no report written: {exc}\n", file=sys.stderr)
            failed += 1
            continue
        write_result(out, q["title"], q["query"], answer or "_No relevant content found in the corpus._")
        print(f"\n  ✓ Saved {out}\n")

    print(f"  Done. {len(pending) - failed} report(s) in {args.results_dir}/, {failed} failed\n")
    return 1 if failed else 0
