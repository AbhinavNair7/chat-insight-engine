"""Stage 1 — export one WhatsApp group from the MCP bridge's SQLite store to CSV.

The whatsapp-mcp bridge writes two files:
    store/whatsapp.db  — session/device keys (never read here)
    store/messages.db  — chats + messages (read-only here)

Output columns: timestamp (ISO-8601 UTC), chat_jid, sender, message, message_id
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from chat_insight import config

SAMPLE_ROWS = 5
CSV_FIELDS = ["timestamp", "chat_jid", "sender", "message", "message_id"]

_REQUIRED_CHATS_COLS = {"jid", "name"}
_REQUIRED_MSG_COLS = {"id", "chat_jid", "sender", "content", "timestamp"}

log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────


def open_db(db_path: str) -> sqlite3.Connection:
    """Open *db_path* read-only; raises FileNotFoundError if absent."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path.resolve()}\n"
            "  → Start the bridge first:  cd whatsapp-mcp/whatsapp-bridge && go run main.go"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_schema(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return ``{table_name: [col, ...]}`` for every table in the DB."""
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    schema: dict[str, list[str]] = {}
    for (tbl,) in tables.fetchall():
        info = conn.execute(f"PRAGMA table_info({tbl})")
        schema[tbl] = [row["name"] for row in info.fetchall()]
    return schema


def validate_schema(schema: dict[str, list[str]]) -> list[str]:
    """Return a list of problems; an empty list means the schema is usable."""
    missing = {"chats", "messages"} - schema.keys()
    if missing:
        return [f"Missing tables: {sorted(missing)}"]
    problems = [
        f"chats table is missing column: {col!r}" for col in sorted(_REQUIRED_CHATS_COLS) if col not in schema["chats"]
    ]
    problems += [
        f"messages table is missing column: {col!r}"
        for col in sorted(_REQUIRED_MSG_COLS)
        if col not in schema["messages"]
    ]
    return problems


# ── Group discovery ───────────────────────────────────────────────────────────


def find_group_jid(conn: sqlite3.Connection, group_name: str) -> str | None:
    """JID of the single group whose name contains *group_name* (case-insensitive), else None."""
    rows = conn.execute(
        "SELECT jid, name FROM chats WHERE LOWER(name) LIKE LOWER(?) AND jid LIKE '%@g.us'",
        (f"%{group_name}%",),
    ).fetchall()
    if len(rows) == 1:
        log.info("Auto-detected group %r → %s", rows[0]["name"], rows[0]["jid"])
        return rows[0]["jid"]
    if len(rows) > 1:
        log.warning("Multiple groups matched %r — pass --group-jid to disambiguate:", group_name)
        for r in rows:
            log.warning("  %s  %s", r["jid"], r["name"])
    return None


def list_group_candidates(conn: sqlite3.Connection, sample: int = 3) -> list[dict]:
    """Every group chat with its name and up to *sample* recent messages."""
    groups = conn.execute(
        "SELECT jid, name FROM chats WHERE jid LIKE '%@g.us' ORDER BY last_message_time DESC"
    ).fetchall()
    results: list[dict] = []
    for g in groups:
        msgs = conn.execute(
            "SELECT content, timestamp FROM messages "
            "WHERE chat_jid = ? AND content IS NOT NULL AND content != '' "
            "ORDER BY timestamp DESC LIMIT ?",
            (g["jid"], sample),
        ).fetchall()
        results.append({"jid": g["jid"], "name": g["name"], "sample_messages": [dict(r) for r in msgs]})
    return results


# ── Export ────────────────────────────────────────────────────────────────────


def to_iso_utc(raw: object) -> str:
    """Normalise the bridge's timestamp (``YYYY-MM-DD HH:MM:SS+00:00`` or epoch) to ISO-8601 UTC."""
    if raw is None:
        return ""
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc).isoformat()  # type: ignore[call-overload]
    except (ValueError, TypeError):
        pass
    try:
        dt = datetime.fromisoformat(str(raw).replace(" ", "T"))
    except ValueError:
        return str(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def export_messages(
    conn: sqlite3.Connection,
    group_jid: str,
    limit: int | None,
    output: str,
) -> int:
    """Write the group's text messages to *output* as CSV; returns rows written.

    Media-only messages (NULL/empty content) are skipped. ``limit`` keeps the most recent N.
    """
    query = (
        "SELECT timestamp, chat_jid, sender, content AS message, id AS message_id "
        "FROM messages WHERE chat_jid = ? AND content IS NOT NULL AND content != '' "
        "ORDER BY timestamp ASC"
    )
    rows = conn.execute(query, (group_jid,)).fetchall()
    if limit:
        rows = rows[-limit:]
    if not rows:
        log.warning("No messages found for JID %s", group_jid)
        return 0

    n_sample = min(SAMPLE_ROWS, len(rows))
    print(f"\n── Sample (last {n_sample} of {len(rows)} rows) ──")
    for row in rows[-n_sample:]:
        msg = (row["message"] or "").replace("\n", " ")[:80]
        print(f"  [{to_iso_utc(row['timestamp'])}]  {row['sender']}:  {msg}")
    print()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": to_iso_utc(row["timestamp"]),
                    "chat_jid": row["chat_jid"],
                    "sender": row["sender"],
                    "message": row["message"],
                    "message_id": row["message_id"],
                }
            )
    log.info("Wrote %d rows → %s", len(rows), out_path.resolve())
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-insight export",
        description="Export one WhatsApp group from the whatsapp-mcp bridge's messages.db to CSV.",
    )
    p.add_argument("--db", default=config.WHATSAPP_DB, help="Path to messages.db (default: %(default)s)")
    p.add_argument(
        "--group-name",
        default=config.GROUP_NAME,
        metavar="NAME",
        help="Group name substring, case-insensitive (env: GROUP_NAME)",
    )
    p.add_argument("--group-jid", default=None, metavar="JID", help="Exact group JID (skips the name search)")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Export only the most recent N messages")
    p.add_argument("--output", default=config.EXPORT_CSV, help="Output CSV (default: %(default)s)")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s  %(message)s",
    )

    if args.limit is not None and args.limit < 1:
        log.error("--limit must be ≥ 1")
        return 1

    try:
        conn = open_db(args.db)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    schema = inspect_schema(conn)
    problems = validate_schema(schema)
    if problems:
        log.error("Unexpected database schema — cannot continue.")
        for prob in problems:
            log.error("  • %s", prob)
        for tbl, cols in schema.items():
            log.info("  %-42s %s", tbl, cols)
        return 1

    group_jid: str | None = args.group_jid
    if not group_jid and args.group_name:
        group_jid = find_group_jid(conn, args.group_name)

    if not group_jid:
        log.warning("No group selected. Group chats in this database:\n")
        for c in list_group_candidates(conn):
            print(f"  JID  : {c['jid']}")
            print(f"  Name : {c['name']}")
            for m in c["sample_messages"]:
                snippet = (m["content"] or "").replace("\n", " ")[:100]
                print(f"    [{m['timestamp']}]  {snippet}")
            print()
        log.info("Re-run with --group-name <NAME> or --group-jid <JID>")
        return 1

    n = export_messages(conn, group_jid, args.limit, args.output)
    if n == 0:
        return 1
    print(f"✓  Exported {n} messages → {args.output}")
    return 0
