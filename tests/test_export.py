"""
tests/test_export.py

Unit tests for chat_insight.export using an in-memory SQLite fixture.
No real database or network access required.

Run with:
    uv run pytest tests/test_export.py -v
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from chat_insight.export import (
    export_messages,
    find_group_jid,
    inspect_schema,
    list_group_candidates,
    open_db,
    to_iso_utc,
    validate_schema,
)

# ── In-memory fixture ─────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE chats (
    jid               TEXT PRIMARY KEY,
    name              TEXT,
    last_message_time TIMESTAMP
);
CREATE TABLE messages (
    id              TEXT,
    chat_jid        TEXT,
    sender          TEXT,
    content         TEXT,
    timestamp       TIMESTAMP,
    is_from_me      BOOLEAN,
    media_type      TEXT,
    filename        TEXT,
    url             TEXT,
    media_key       BLOB,
    file_sha256     BLOB,
    file_enc_sha256 BLOB,
    file_length     INTEGER,
    PRIMARY KEY (id, chat_jid),
    FOREIGN KEY (chat_jid) REFERENCES chats(jid)
);
"""

_CHATS = [
    ("100000000000000001@g.us", "AI in Healthcare", "2026-03-08 12:31:55"),
    ("100000000000000002@g.us", "Running Club", "2026-03-07 10:00:00"),
    ("100000000000000003@g.us", "Book Club", "2026-03-06 09:00:00"),
]

_MESSAGES = [
    # (id, chat_jid, sender, content, timestamp, is_from_me)
    (
        "MSG001",
        "100000000000000001@g.us",
        "alice@lid",
        "Anyone using ambient AI scribes?",
        "2026-01-01 09:00:00+00:00",
        0,
    ),
    ("MSG002", "100000000000000001@g.us", "bob@lid", "We trialled Nabla last month.", "2026-01-01 09:07:00+00:00", 0),
    ("MSG003", "100000000000000001@g.us", "alice@lid", "Consent is the main concern.", "2026-01-01 09:14:00+00:00", 0),
    (
        "MSG004",
        "100000000000000001@g.us",
        "carol@lid",
        "CCIO approved a six-month pilot.",
        "2026-01-01 09:21:00+00:00",
        0,
    ),
    (
        "MSG005",
        "100000000000000001@g.us",
        "bob@lid",
        "Data quality is the real bottleneck.",
        "2026-01-01 09:28:00+00:00",
        0,
    ),
    ("OTHER01", "100000000000000002@g.us", "dave@lid", "Good morning team!", "2026-03-07 08:00:00+00:00", 0),
]

TARGET_JID = "100000000000000001@g.us"


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    conn.executemany("INSERT INTO chats VALUES (?, ?, ?)", _CHATS)
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me) VALUES (?, ?, ?, ?, ?, ?)",
        _MESSAGES,
    )
    conn.commit()
    return conn


# ── inspect_schema / validate_schema ─────────────────────────────────────────


class TestInspectSchema:
    def test_returns_all_tables(self):
        schema = inspect_schema(_make_conn())
        assert "chats" in schema
        assert "messages" in schema

    def test_chats_columns_present(self):
        schema = inspect_schema(_make_conn())
        assert "jid" in schema["chats"]
        assert "name" in schema["chats"]

    def test_messages_columns_present(self):
        schema = inspect_schema(_make_conn())
        for col in ("id", "chat_jid", "sender", "content", "timestamp"):
            assert col in schema["messages"]

    def test_empty_db_returns_empty_dict(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        assert inspect_schema(conn) == {}


class TestValidateSchema:
    def test_clean_schema_no_problems(self):
        assert validate_schema(inspect_schema(_make_conn())) == []

    def test_flags_missing_tables(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        problems = validate_schema(inspect_schema(conn))
        assert any("Missing tables" in p for p in problems)

    def test_flags_missing_chats_column(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # 'name' is absent intentionally
        conn.execute("CREATE TABLE chats (jid TEXT, last_message_time TIMESTAMP)")
        conn.execute("CREATE TABLE messages (id TEXT, chat_jid TEXT, sender TEXT, content TEXT, timestamp TIMESTAMP)")
        problems = validate_schema(inspect_schema(conn))
        assert any("name" in p for p in problems)

    def test_flags_missing_messages_column(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE chats (jid TEXT, name TEXT)")
        # 'content' is absent intentionally
        conn.execute("CREATE TABLE messages (id TEXT, chat_jid TEXT, sender TEXT, timestamp TIMESTAMP)")
        problems = validate_schema(inspect_schema(conn))
        assert any("content" in p for p in problems)


# ── find_group_jid ────────────────────────────────────────────────────────────


class TestFindGroupJid:
    def test_exact_name_match(self):
        assert find_group_jid(_make_conn(), "AI in Healthcare") == TARGET_JID

    def test_case_insensitive(self):
        assert find_group_jid(_make_conn(), "ai in healthcare") == TARGET_JID

    def test_partial_name_match(self):
        assert find_group_jid(_make_conn(), "AI in") == TARGET_JID

    def test_no_match_returns_none(self):
        assert find_group_jid(_make_conn(), "Nonexistent Group XYZ") is None

    def test_multiple_matches_returns_none(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO chats VALUES (?, ?, ?)",
            ("999@g.us", "AI in Healthcare duplicate", "2026-01-01"),
        )
        conn.commit()
        assert find_group_jid(conn, "AI in Healthcare") is None

    def test_ignores_non_group_jids(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO chats VALUES (?, ?, ?)",
            ("100000000000@s.whatsapp.net", "AI in Healthcare DM", "2026-01-01"),
        )
        conn.commit()
        # The DM should not match because it's not @g.us
        assert find_group_jid(conn, "AI in Healthcare") == TARGET_JID


# ── list_group_candidates ─────────────────────────────────────────────────────


class TestListGroupCandidates:
    def test_only_returns_group_jids(self):
        for c in list_group_candidates(_make_conn()):
            assert c["jid"].endswith("@g.us")

    def test_all_group_chats_present(self):
        jids = {c["jid"] for c in list_group_candidates(_make_conn())}
        assert TARGET_JID in jids
        assert "100000000000000002@g.us" in jids

    def test_includes_sample_messages(self):
        candidates = list_group_candidates(_make_conn())
        ai = next(c for c in candidates if "AI in Healthcare" in (c["name"] or ""))
        assert len(ai["sample_messages"]) > 0

    def test_sample_size_capped(self):
        for c in list_group_candidates(_make_conn(), sample=2):
            assert len(c["sample_messages"]) <= 2

    def test_sample_keys_present(self):
        candidates = list_group_candidates(_make_conn())
        ai = next(c for c in candidates if "AI in Healthcare" in (c["name"] or ""))
        for msg in ai["sample_messages"]:
            assert "content" in msg
            assert "timestamp" in msg


# ── export_messages ───────────────────────────────────────────────────────────


class TestExportMessages:
    def test_csv_fieldnames(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["timestamp", "chat_jid", "sender", "message", "message_id"]

    def test_row_count(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        n = export_messages(_make_conn(), TARGET_JID, None, out)
        assert n == 5

    def test_limit_flag_row_count(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        n = export_messages(_make_conn(), TARGET_JID, 3, out)
        assert n == 3

    def test_limit_flag_csv_rows(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, 3, out)
        with open(out) as f:
            assert len(list(csv.DictReader(f))) == 3

    def test_limit_keeps_most_recent(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, 2, out)
        with open(out) as f:
            ids = [row["message_id"] for row in csv.DictReader(f)]
        assert ids == ["MSG004", "MSG005"]

    def test_only_target_group_exported(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        with open(out) as f:
            for row in csv.DictReader(f):
                assert row["chat_jid"] == TARGET_JID

    def test_timestamps_are_iso_utc(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        with open(out) as f:
            for row in csv.DictReader(f):
                dt = datetime.fromisoformat(row["timestamp"])
                assert dt.tzinfo is not None, f"No timezone in {row['timestamp']}"

    def test_message_id_preserved(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        with open(out) as f:
            ids = {row["message_id"] for row in csv.DictReader(f)}
        assert "MSG001" in ids

    def test_unknown_jid_returns_zero(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        n = export_messages(_make_conn(), "unknown@g.us", None, out)
        assert n == 0

    def test_blank_messages_excluded(self, tmp_path: Path):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me) VALUES (?,?,?,?,?,?)",
            ("MSG_BLANK", TARGET_JID, "alice@lid", "", "2026-01-01 09:30:00+00:00", 0),
        )
        conn.execute(
            "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me) VALUES (?,?,?,?,?,?)",
            ("MSG_NULL", TARGET_JID, "alice@lid", None, "2026-01-01 09:31:00+00:00", 0),
        )
        conn.commit()
        out = str(tmp_path / "out.csv")
        n = export_messages(conn, TARGET_JID, None, out)
        assert n == 5  # blanks/nulls must be excluded

    def test_messages_ordered_by_timestamp(self, tmp_path: Path):
        out = str(tmp_path / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        with open(out) as f:
            timestamps = [row["timestamp"] for row in csv.DictReader(f)]
        assert timestamps == sorted(timestamps)

    def test_creates_parent_directory(self, tmp_path: Path):
        out = str(tmp_path / "subdir" / "out.csv")
        export_messages(_make_conn(), TARGET_JID, None, out)
        assert Path(out).exists()


# ── to_iso_utc ───────────────────────────────────────────────────────────────


class TestToIsoUtc:
    def test_bridge_format(self):
        result = to_iso_utc("2026-03-08 12:31:55+00:00")
        assert result.startswith("2026-03-08T12:31:55")

    def test_adds_utc_when_no_tz(self):
        result = to_iso_utc("2026-01-01 09:00:00")
        dt = datetime.fromisoformat(result)
        assert dt.tzinfo is not None

    def test_unix_epoch_zero(self):
        assert "1970" in to_iso_utc("0")

    def test_integer_input(self):
        result = to_iso_utc(0)
        assert "1970" in result

    def test_none_returns_empty_string(self):
        assert to_iso_utc(None) == ""

    def test_unparseable_passthrough(self):
        result = to_iso_utc("not-a-date")
        assert result == "not-a-date"


# ── open_db ───────────────────────────────────────────────────────────────────


class TestOpenDb:
    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            open_db("/nonexistent/path/messages.db")

    def test_opens_valid_file(self, tmp_path: Path):
        db = tmp_path / "test.db"
        sqlite3.connect(str(db)).close()
        conn = open_db(str(db))
        assert conn is not None
        conn.close()

    def test_row_factory_set(self, tmp_path: Path):
        db = tmp_path / "test.db"
        c = sqlite3.connect(str(db))
        c.execute("CREATE TABLE t (x INTEGER)")
        c.execute("INSERT INTO t VALUES (42)")
        c.commit()
        c.close()
        conn = open_db(str(db))
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row["x"] == 42
        conn.close()
