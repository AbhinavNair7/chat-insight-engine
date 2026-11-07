"""
tests/test_chunker.py

Unit tests for chat_insight.chunking.

Validates:
  - No message loss    (every message_id appears in at least one chunk)
  - No duplication     (within a single chunk, no message_id appears twice)
  - Chunk shape        (id, text, meta keys; meta field types)
  - Rolling overlap    (stride < chunk_size produces expected overlap)
  - Non-overlapping    (stride == chunk_size produces disjoint windows)
  - Limit flag         (load_csv respects the tail-N semantics)
  - Edge cases         (empty DF, single row, fewer rows than chunk_size)
  - CLI smoke test     (main() exits 0 on a temp CSV)

Run with:
    uv run pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chat_insight.chunking import load_csv, main, make_chunks, write_chunks

# ── Helpers ───────────────────────────────────────────────────────────────────

COLS = ["timestamp", "chat_jid", "sender", "message", "message_id"]


def _make_df(n: int) -> pd.DataFrame:
    """Return a synthetic DataFrame with *n* messages."""
    return pd.DataFrame(
        [
            {
                "timestamp": f"2026-01-01T09:{i:02d}:00+00:00",
                "chat_jid": "100000000000000001@g.us",
                "sender": f"user_{i % 3}@lid",
                "message": f"Message number {i}",
                "message_id": f"MSG{i:04d}",
            }
            for i in range(n)
        ],
        dtype=str,
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# ── load_csv ──────────────────────────────────────────────────────────────────


class TestLoadCsv:
    def test_loads_file(self, tmp_path: Path):
        p = tmp_path / "msgs.csv"
        _write_csv(_make_df(20), p)
        df = load_csv(str(p))
        assert len(df) == 20

    def test_limit_returns_tail(self, tmp_path: Path):
        p = tmp_path / "msgs.csv"
        _write_csv(_make_df(50), p)
        df = load_csv(str(p), limit=10)
        assert len(df) == 10
        # tail means the last 10 rows (MSG0040–MSG0049)
        assert df["message_id"].iloc[0] == "MSG0040"

    def test_limit_larger_than_file(self, tmp_path: Path):
        p = tmp_path / "msgs.csv"
        _write_csv(_make_df(5), p)
        df = load_csv(str(p), limit=100)
        assert len(df) == 5

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")

    def test_raises_for_missing_columns(self, tmp_path: Path):
        p = tmp_path / "bad.csv"
        p.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_csv(str(p))

    def test_returns_string_dtype(self, tmp_path: Path):
        p = tmp_path / "msgs.csv"
        _write_csv(_make_df(5), p)
        df = load_csv(str(p))
        # pandas 2.x may return StringDtype; both are string-compatible
        assert pd.api.types.is_string_dtype(df["message_id"])


# ── make_chunks ───────────────────────────────────────────────────────────────


class TestMakeChunks:
    # ── shape ────────────────────────────────────────────────────────────────

    def test_returns_list_of_dicts(self):
        chunks = make_chunks(_make_df(10), chunk_size=5, stride=5)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_keys_present(self):
        chunks = make_chunks(_make_df(10), chunk_size=5, stride=5)
        for c in chunks:
            assert set(c.keys()) == {"id", "text", "meta"}

    def test_meta_keys_present(self):
        chunks = make_chunks(_make_df(10), chunk_size=5, stride=5)
        required = {"chunk_id", "chat_jid", "start_ts", "end_ts", "participants", "n_messages"}
        for c in chunks:
            assert required <= set(c["meta"].keys())

    def test_chunk_ids_are_sequential(self):
        chunks = make_chunks(_make_df(30), chunk_size=10, stride=10)
        for i, c in enumerate(chunks):
            assert c["id"] == f"chunk_{i:04d}"
            assert c["meta"]["chunk_id"] == i

    def test_n_messages_matches_text_lines(self):
        chunks = make_chunks(_make_df(30), chunk_size=10, stride=10)
        for c in chunks:
            assert c["text"].count("\n") + 1 == c["meta"]["n_messages"]

    def test_participants_are_sorted_unique(self):
        chunks = make_chunks(_make_df(30), chunk_size=10, stride=10)
        for c in chunks:
            p = c["meta"]["participants"]
            assert p == sorted(set(p))

    # ── no message loss ───────────────────────────────────────────────────────

    def test_no_message_loss_non_overlapping(self):
        """Every message must appear in at least one chunk's text."""
        df = _make_df(45)
        text = "\n".join(c["text"] for c in make_chunks(df, chunk_size=10, stride=10))
        for msg in df["message"]:
            assert f": {msg}\n" in text + "\n"

    def test_no_message_loss_overlapping(self):
        """Every message must appear in at least one overlapping chunk."""
        df = _make_df(50)
        text = "\n".join(c["text"] for c in make_chunks(df, chunk_size=15, stride=5))
        for msg in df["message"]:
            assert f": {msg}\n" in text + "\n"

    # ── no intra-chunk duplication ────────────────────────────────────────────

    def test_no_intra_chunk_duplication(self):
        """Within a single chunk, every line is unique."""
        df = _make_df(60)
        for chunk_size, stride in [(10, 10), (15, 5), (30, 15)]:
            chunks = make_chunks(df, chunk_size=chunk_size, stride=stride)
            for c in chunks:
                lines = c["text"].splitlines()
                assert len(lines) == len(set(lines)), (
                    f"Duplicate lines in {c['id']} (chunk_size={chunk_size}, stride={stride})"
                )

    # ── window sizes ─────────────────────────────────────────────────────────

    def test_non_overlapping_windows_cover_all(self):
        df = _make_df(30)
        chunks = make_chunks(df, chunk_size=10, stride=10)
        assert len(chunks) == 3
        for c in chunks:
            assert c["meta"]["n_messages"] == 10

    def test_overlapping_produces_more_chunks(self):
        df = _make_df(30)
        non_ol = make_chunks(df, chunk_size=10, stride=10)
        overlapping = make_chunks(df, chunk_size=10, stride=5)
        assert len(overlapping) > len(non_ol)

    def test_last_chunk_may_be_shorter(self):
        df = _make_df(25)
        chunks = make_chunks(df, chunk_size=10, stride=10)
        # 0–9, 10–19, 20–24
        assert chunks[-1]["meta"]["n_messages"] == 5

    def test_chunk_size_larger_than_df(self):
        df = _make_df(5)
        chunks = make_chunks(df, chunk_size=30, stride=30)
        assert len(chunks) == 1
        assert chunks[0]["meta"]["n_messages"] == 5

    def test_single_row_df(self):
        df = _make_df(1)
        chunks = make_chunks(df, chunk_size=10, stride=10)
        assert len(chunks) == 1
        assert chunks[0]["meta"]["n_messages"] == 1

    def test_empty_df_returns_empty(self):
        chunks = make_chunks(_make_df(0), chunk_size=10, stride=10)
        assert chunks == []

    # ── meta fields ──────────────────────────────────────────────────────────

    def test_start_and_end_ts(self):
        df = _make_df(10)
        chunks = make_chunks(df, chunk_size=5, stride=5)
        first = chunks[0]
        assert first["meta"]["start_ts"] == df["timestamp"].iloc[0]
        assert first["meta"]["end_ts"] == df["timestamp"].iloc[4]

    def test_chat_jid_correct(self):
        df = _make_df(10)
        chunks = make_chunks(df, chunk_size=5, stride=5)
        for c in chunks:
            assert c["meta"]["chat_jid"] == "100000000000000001@g.us"

    # ── text format ───────────────────────────────────────────────────────────

    def test_text_line_format(self):
        df = _make_df(3)
        chunks = make_chunks(df, chunk_size=3, stride=3)
        lines = chunks[0]["text"].splitlines()
        assert lines[0] == "[2026-01-01T09:00:00+00:00] user_0@lid: Message number 0"
        assert lines[1] == "[2026-01-01T09:01:00+00:00] user_1@lid: Message number 1"
        assert lines[2] == "[2026-01-01T09:02:00+00:00] user_2@lid: Message number 2"


# ── write_chunks ──────────────────────────────────────────────────────────────


class TestWriteChunks:
    def test_writes_valid_json(self, tmp_path: Path):
        chunks = make_chunks(_make_df(10), chunk_size=5, stride=5)
        out = tmp_path / "chunks.json"
        write_chunks(chunks, str(out))
        with out.open() as f:
            loaded = json.load(f)
        assert loaded == chunks

    def test_creates_parent_directory(self, tmp_path: Path):
        chunks = make_chunks(_make_df(5), chunk_size=5, stride=5)
        out = tmp_path / "sub" / "chunks.json"
        write_chunks(chunks, str(out))
        assert out.exists()

    def test_roundtrip_preserves_content(self, tmp_path: Path):
        df = _make_df(20)
        chunks = make_chunks(df, chunk_size=7, stride=7)
        out = tmp_path / "chunks.json"
        write_chunks(chunks, str(out))
        with out.open() as f:
            loaded = json.load(f)
        assert len(loaded) == len(chunks)
        assert loaded[0]["text"] == chunks[0]["text"]
        assert loaded[0]["meta"] == chunks[0]["meta"]


# ── CLI / main() ──────────────────────────────────────────────────────────────


class TestMain:
    def test_smoke_exits_zero(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "msgs.csv"
        out_path = tmp_path / "chunks.json"
        _write_csv(_make_df(200), csv_path)
        rc = main(
            [
                "--input",
                str(csv_path),
                "--out",
                str(out_path),
                "--chunk-size",
                "10",
                "--stride",
                "10",
                "--limit",
                "200",
            ]
        )
        assert rc == 0
        assert out_path.exists()

    def test_smoke_prints_first_chunk(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "msgs.csv"
        out_path = tmp_path / "chunks.json"
        _write_csv(_make_df(50), csv_path)
        main(
            [
                "--input",
                str(csv_path),
                "--out",
                str(out_path),
                "--chunk-size",
                "10",
                "--stride",
                "10",
            ]
        )
        captured = capsys.readouterr().out
        assert "chunk_0000" in captured
        assert "chat_jid" in captured

    def test_missing_input_exits_nonzero(self, tmp_path: Path):
        rc = main(
            [
                "--input",
                str(tmp_path / "nonexistent.csv"),
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
        assert rc == 1

    def test_invalid_chunk_size_exits_nonzero(self, tmp_path: Path):
        csv_path = tmp_path / "msgs.csv"
        _write_csv(_make_df(10), csv_path)
        rc = main(
            [
                "--input",
                str(csv_path),
                "--out",
                str(tmp_path / "out.json"),
                "--chunk-size",
                "0",
            ]
        )
        assert rc == 1

    def test_output_json_is_valid(self, tmp_path: Path):
        csv_path = tmp_path / "msgs.csv"
        out_path = tmp_path / "chunks.json"
        _write_csv(_make_df(30), csv_path)
        main(
            [
                "--input",
                str(csv_path),
                "--out",
                str(out_path),
                "--chunk-size",
                "10",
                "--stride",
                "10",
            ]
        )
        with out_path.open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3
