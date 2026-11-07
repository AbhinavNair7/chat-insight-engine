"""Question-file parsing, report naming and the top-level CLI."""

from __future__ import annotations

from pathlib import Path

from chat_insight import cli
from chat_insight.batch import parse_queries, result_path, safe_filename, write_result


def test_parse_queries_joins_multiline_bodies():
    text = (
        "Intro line that is ignored\n\n"
        "Query 1 — First\nline one\nline two\n\n"
        "Query 2 — Empty\n\n"
        "query 3 — Third\nbody\n"
    )
    assert parse_queries(text) == [
        {"title": "Query 1 — First", "query": "line one line two"},
        {"title": "query 3 — Third", "query": "body"},
    ]


def test_bundled_example_questions_parse():
    root = Path(__file__).resolve().parents[1]
    queries = parse_queries((root / "examples" / "questions.md").read_text(encoding="utf-8"))
    assert len(queries) == 4
    assert all(q["query"] for q in queries)


def test_safe_filename_strips_illegal_characters():
    assert safe_filename('Query 1: "A/B" <test>?') == "Query 1 AB test"


def test_write_result(tmp_path: Path):
    path = result_path(str(tmp_path / "reports"), "Query 1 — Title")
    write_result(path, "Query 1 — Title", "the question", "the answer")
    body = path.read_text(encoding="utf-8")
    assert body.startswith("# Query 1 — Title")
    assert "**Query:** the question" in body and body.rstrip().endswith("the answer")


def test_cli_help_lists_commands(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command in out


def test_cli_unknown_command(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_cli_dispatches_to_stage(tmp_path: Path):
    assert cli.main(["chunk", "--input", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "c.json")]) == 1


def test_failed_scan_writes_no_report_and_exits_nonzero(tmp_path: Path, monkeypatch):
    from chat_insight import batch

    def failing_scan(*a, **k):
        raise RuntimeError("quota exhausted")

    monkeypatch.setattr(batch, "open_collection", lambda *a, **k: object())
    monkeypatch.setattr(batch, "scan_corpus", failing_scan)
    questions = tmp_path / "q.md"
    questions.write_text("Query 1 — Only\nWhat happened?\n")
    reports = tmp_path / "reports"
    assert batch.main(["--queries", str(questions), "--results-dir", str(reports)]) == 1
    assert not reports.exists() or not any(reports.iterdir())


def test_export_rejects_non_positive_limit():
    from chat_insight import export

    assert export.main(["--db", "/nonexistent.db", "--group-jid", "x@g.us", "--limit", "0"]) == 1


def test_demo_data_ships_inside_the_package():
    from chat_insight.demo import sample_csv_path

    assert sample_csv_path().is_file()
