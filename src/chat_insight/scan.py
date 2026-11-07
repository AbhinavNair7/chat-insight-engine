"""Full-corpus map-reduce scan.

Top-k retrieval sees a few chunks, which is the wrong tool for questions like
"what are the top frustrations?". Scan mode reads every chunk instead:

    map    — a cheap, high-throughput model extracts quoted, cited evidence from
             each chronological batch of chunks (or says NONE)
    reduce — a stronger model synthesises all batch findings into ranked themes
"""

from __future__ import annotations

from collections.abc import Callable

import chromadb

from chat_insight import config, prompts
from chat_insight.llm import ChatFn, chat
from chat_insight.store import all_chunks_chronological

DEFAULT_BATCH_SIZE = 8  # 8 chunks × 30 messages ≈ 14k tokens: under typical per-minute caps


def _count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith(("-", "•", "*")))


def scan_corpus(
    collection: chromadb.Collection,
    query: str,
    *,
    chat_model: str = config.CHAT_MODEL,
    map_model: str = config.MAP_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    corpus: str = config.CORPUS_DESCRIPTION,
    llm: ChatFn = chat,
    echo: Callable[..., None] = print,
) -> str | None:
    """Map-reduce *query* over every chunk. Returns the synthesis, or None if nothing was relevant."""
    triples = all_chunks_chronological(collection)
    n_batches = (len(triples) + batch_size - 1) // batch_size
    echo(f"  Scanning {len(triples)} chunks in {n_batches} batches · map={map_model} · reduce={chat_model}\n")

    map_prompt = prompts.MAP_EXTRACT.format(corpus=corpus, query=query)
    findings: list[str] = []
    failures: list[Exception] = []
    for i in range(n_batches):
        batch = triples[i * batch_size : (i + 1) * batch_size]
        start_ts = str(batch[0][2].get("start_ts", ""))[:10]
        end_ts = str(batch[-1][2].get("end_ts", ""))[:10]
        echo(f"  Batch {i + 1:>2}/{n_batches} ({start_ts} → {end_ts})…", end=" ", flush=True)

        context = "\n".join(f"### [{cid}]\n{doc}\n" for cid, doc, _ in batch)
        try:
            finding = llm(
                [
                    {"role": "system", "content": map_prompt},
                    {"role": "user", "content": f"## Conversation chunks\n\n{context}"},
                ],
                map_model,
            ).strip()
        except Exception as exc:  # one failed batch should not sink a long scan
            echo(f"ERROR: {exc}")
            failures.append(exc)
            continue

        if not finding or finding.upper() == "NONE":
            echo("(nothing relevant)")
            continue
        echo(f"{max(_count_bullets(finding), 1)} item(s)")
        findings.append(f"### Batch {i + 1} ({start_ts} → {end_ts})\n{finding}")

    if n_batches and len(failures) == n_batches:
        raise RuntimeError(f"All {n_batches} map batches failed; last error: {failures[-1]}") from failures[-1]
    if failures:
        echo(f"\n  ⚠  {len(failures)}/{n_batches} batches failed; the result covers the rest only.")
    if not findings:
        return None

    echo(f"\n  Synthesising {len(findings)} batches with {chat_model}…\n")
    combined = "\n\n".join(findings)
    return llm(
        [
            {"role": "system", "content": prompts.REDUCE_SYNTHESISE.format(corpus=corpus, query=query)},
            {
                "role": "user",
                "content": f"## Research question\n{query}\n\n## Extracted findings from all batches\n\n{combined}",
            },
        ],
        chat_model,
    )
