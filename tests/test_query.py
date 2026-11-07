"""Grounded answering, map-reduce scan and the auto router, with a scripted fake LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from chat_insight import query as q
from chat_insight.query import QueryOptions, build_user_message, handle_query, route_question
from chat_insight.scan import scan_corpus
from chat_insight.store import get_or_create_collection, upsert_chunks


class FakeLLM:
    """Records every call and answers by prompt type."""

    def __init__(self, route: str = "TARGETED", map_reply: str = '- Label "quote" [chunk_0000]'):
        self.calls: list[tuple[str, str]] = []
        self.route = route
        self.map_reply = map_reply

    def __call__(self, messages: list[dict], model: str) -> str:
        system = messages[0]["content"]
        if "You route research questions" in system:
            kind = "route"
            reply = self.route
        elif "extract every piece of information" in system:
            kind = "map"
            reply = self.map_reply
        elif "Synthesise the findings" in system:
            kind = "reduce"
            reply = "SYNTHESIS"
        else:
            kind = "answer"
            reply = "ANSWER [chunk_0000]"
        self.calls.append((kind, model))
        return reply

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


@pytest.fixture
def collection(tmp_path: Path):
    col = get_or_create_collection(str(tmp_path / "chroma"), "test_col")
    chunks = [
        {
            "id": f"chunk_{i:04d}",
            "text": f"[2025-01-0{i + 1}T09:00:00+00:00] member_{i}: message {i}",
            "meta": {"chunk_id": i, "start_ts": f"2025-01-0{i + 1}", "end_ts": f"2025-01-0{i + 1}", "n_messages": 1},
        }
        for i in range(5)
    ]
    # Orthogonal-ish vectors so distances are predictable.
    vectors = [[1.0 if j == i else 0.0 for j in range(5)] for i in range(5)]
    upsert_chunks(col, chunks, vectors)
    return col


def _embed_as(vec: list[float]):
    return lambda texts, **_: [vec for _ in texts]


def test_user_message_labels_every_chunk_for_citation():
    msg = build_user_message(
        "What tools?",
        [{"id": "chunk_0007", "document": "text", "metadata": {"start_ts": "a", "end_ts": "b", "n_messages": 3}}],
    )
    assert "### [chunk_0007]  (a → b, 3 messages)" in msg
    assert msg.rstrip().endswith("## Query\nWhat tools?")


@pytest.mark.parametrize(("verdict", "expected"), [("BROAD", "scan"), ("broad.", "scan"), ("TARGETED", "rag")])
def test_router(verdict: str, expected: str):
    assert route_question("anything", "m", llm=FakeLLM(route=verdict)) == expected


def test_scan_maps_every_batch_chronologically_then_reduces(collection):
    llm = FakeLLM()
    result = scan_corpus(
        collection, "top themes?", chat_model="big", map_model="small", batch_size=2, llm=llm, echo=lambda *a, **k: None
    )
    assert result == "SYNTHESIS"
    assert llm.calls == [("map", "small")] * 3 + [("reduce", "big")]


def test_scan_returns_none_when_nothing_relevant(collection):
    llm = FakeLLM(map_reply="NONE")
    assert scan_corpus(collection, "q", batch_size=2, llm=llm, echo=lambda *a, **k: None) is None
    assert "reduce" not in llm.kinds()


def test_scan_raises_when_every_batch_fails(collection):
    def broken(messages, model):
        raise RuntimeError("invalid api key")

    with pytest.raises(RuntimeError, match="All 3 map batches failed"):
        scan_corpus(collection, "q", batch_size=2, llm=broken, echo=lambda *a, **k: None)


def test_scan_survives_partial_failure(collection):
    inner = FakeLLM()
    state = {"n": 0}

    def flaky(messages, model):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient")
        return inner(messages, model)

    assert scan_corpus(collection, "q", batch_size=2, llm=flaky, echo=lambda *a, **k: None) == "SYNTHESIS"


def test_rag_mode_answers_from_retrieved_chunks(collection, monkeypatch, capsys):
    monkeypatch.setattr(q, "embed_texts", _embed_as([1.0, 0, 0, 0, 0]))
    llm = FakeLLM()
    handle_query("what about message 0?", collection, QueryOptions(n_results=2), llm=llm)
    assert llm.kinds() == ["answer"]
    out = capsys.readouterr().out
    assert "ANSWER [chunk_0000]" in out
    assert "chunk_0000" in out.split("Chunks:")[1].splitlines()[0]


def test_auto_routes_broad_questions_to_scan(collection, monkeypatch):
    monkeypatch.setattr(q, "embed_texts", lambda *a, **k: pytest.fail("scan must not embed the query"))
    llm = FakeLLM(route="BROAD")
    handle_query("what are the top themes?", collection, QueryOptions(mode="auto"), llm=llm)
    assert llm.kinds()[0] == "route"
    assert llm.kinds()[-1] == "reduce"


def test_auto_escalates_to_scan_when_retrieval_is_poor(collection, monkeypatch):
    monkeypatch.setattr(q, "embed_texts", _embed_as([-1.0, -1.0, -1.0, -1.0, -1.0]))  # far from every chunk
    llm = FakeLLM(route="TARGETED")
    handle_query("something unrelated", collection, QueryOptions(mode="auto"), llm=llm)
    assert "answer" not in llm.kinds()
    assert llm.kinds()[-1] == "reduce"


def test_dry_run_never_calls_llm(collection, monkeypatch, capsys):
    monkeypatch.setattr(q, "embed_texts", _embed_as([1.0, 0, 0, 0, 0]))
    llm = FakeLLM()
    handle_query("x", collection, QueryOptions(mode="auto", dry_run=True, n_results=1), llm=llm)
    assert llm.calls == []
    assert "dist=" in capsys.readouterr().out


def test_find_quote_prints_context_and_chunk(tmp_path: Path, capsys):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(
        "timestamp,chat_jid,sender,message,message_id\n"
        "2025-01-01T09:00:00+00:00,g@g.us,member_a,first,M1\n"
        "2025-01-01T09:05:00+00:00,g@g.us,member_b,printing and re-scanning again,M2\n"
    )
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        '[{"id": "chunk_0000", "text": "", "meta": {"start_ts": "2025-01-01T09:00:00+00:00",'
        ' "end_ts": "2025-01-01T09:05:00+00:00"}}]'
    )
    assert q.find_quote_context("RE-SCANNING", str(csv_path), str(chunks_path), window=1) == 0
    out = capsys.readouterr().out
    assert "1 match(es)" in out and "chunk_0000" in out and "▶" in out
