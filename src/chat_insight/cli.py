"""``chat-insight <command>`` — one entry point for every pipeline stage."""

from __future__ import annotations

import sys
from importlib import import_module

COMMANDS = {
    "export": ("chat_insight.export", "WhatsApp (via MCP bridge) → CSV"),
    "chunk": ("chat_insight.chunking", "CSV → rolling conversation chunks"),
    "embed": ("chat_insight.ingest", "chunks → embeddings → Chroma (idempotent)"),
    "ask": ("chat_insight.query", "grounded Q&A: retrieval, full scan, or auto-routed"),
    "batch": ("chat_insight.batch", "one research report per question in a file"),
    "keywords": ("chat_insight.keywords", "deterministic term/bigram frequencies"),
    "demo": ("chat_insight.demo", "offline demo on synthetic data, no API key"),
}


def usage() -> str:
    lines = ["usage: chat-insight <command> [options]", "", "commands:"]
    lines += [f"  {name:<9} {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run `chat-insight <command> --help` for a command's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print(usage())
        return 0
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{usage()}", file=sys.stderr)
        return 2
    module = import_module(COMMANDS[command][0])
    return int(module.main(rest))


if __name__ == "__main__":
    sys.exit(main())
