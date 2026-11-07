"""Idempotent indexing: manifest bookkeeping and change detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chat_insight import ingest
from chat_insight.ingest import chunk_checksum, filter_new_chunks, load_manifest, save_manifest
from chat_insight.store import flatten_meta, open_collection


def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "meta": {"chunk_id": 0, "participants": ["a", "b"], "n_messages": 1}}


class TestFilterNewChunks:
    def test_all_new_when_manifest_empty(self):
        chunks = [_chunk("c1", "hello"), _chunk("c2", "world")]
        to_embed, to_skip = filter_new_chunks(chunks, {"embedded_chunks": {}}, "m1")
        assert [c["id"] for c in to_embed] == ["c1", "c2"]
        assert to_skip == []

    def test_unchanged_chunks_skipped(self):
        manifest = {"embedded_chunks": {"c1": chunk_checksum("hello")}, "embedding_model": "m1"}
        to_embed, to_skip = filter_new_chunks([_chunk("c1", "hello"), _chunk("c2", "new")], manifest, "m1")
        assert [c["id"] for c in to_embed] == ["c2"]
        assert [c["id"] for c in to_skip] == ["c1"]

    def test_changed_text_reembedded(self):
        manifest = {"embedded_chunks": {"c1": chunk_checksum("old")}, "embedding_model": "m1"}
        to_embed, _ = filter_new_chunks([_chunk("c1", "edited")], manifest, "m1")
        assert [c["id"] for c in to_embed] == ["c1"]

    def test_model_change_reembeds_everything(self):
        manifest = {"embedded_chunks": {"c1": chunk_checksum("hello")}, "embedding_model": "m1"}
        to_embed, to_skip = filter_new_chunks([_chunk("c1", "hello")], manifest, "m2")
        assert len(to_embed) == 1 and to_skip == []


class TestManifest:
    def test_missing_file_gives_blank_manifest(self, tmp_path: Path):
        assert load_manifest(str(tmp_path / "none.json")) == {"embedded_chunks": {}, "embedding_model": None}

    def test_roundtrip_and_no_temp_left(self, tmp_path: Path):
        path = tmp_path / "sub" / "manifest.json"
        data = {"embedded_chunks": {"c1": "abc"}, "embedding_model": "m1"}
        save_manifest(data, str(path))
        assert load_manifest(str(path)) == data
        assert not path.with_suffix(".tmp").exists()


def test_flatten_meta_serialises_lists():
    flat = flatten_meta({"n": 3, "who": ["a", "b"], "ts": "2025"})
    assert flat == {"n": 3, "who": '["a", "b"]', "ts": "2025"}


def test_run_embeds_once_then_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """Second run makes no embedding calls: the manifest proves idempotency end to end."""
    calls: list[int] = []

    def fake_embed(texts, **_):
        calls.append(len(texts))
        return [[float(len(t)), 1.0, 0.5] for t in texts]

    monkeypatch.setattr(ingest, "embed_texts", fake_embed)
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps([_chunk("c1", "alpha"), _chunk("c2", "beta gamma")]))
    kwargs = dict(
        chunks_path=str(chunks_path),
        manifest_path=str(tmp_path / "manifest.json"),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_col",
        batch_size=64,
        use_local=True,
        openai_model="unused",
        local_model="fake",
        sample=None,
    )

    assert ingest.run(**kwargs) == 0
    assert calls == [2]
    assert open_collection(str(tmp_path / "chroma"), "test_col").count() == 2

    assert ingest.run(**kwargs) == 0
    assert calls == [2]
    assert "Nothing new to embed" in capsys.readouterr().out


def test_status_reports_missing(tmp_path: Path, capsys):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps([_chunk("c1", "alpha")]))
    assert ingest.print_status(str(chunks_path), str(tmp_path / "manifest.json")) == 1
    assert "Missing: c1" in capsys.readouterr().out


def _run_kwargs(tmp_path: Path, local_model: str = "fake") -> dict:
    return dict(
        chunks_path=str(tmp_path / "chunks.json"),
        manifest_path=str(tmp_path / "manifest.json"),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_col",
        batch_size=64,
        use_local=True,
        openai_model="unused",
        local_model=local_model,
        sample=None,
    )


def test_rechunking_prunes_orphaned_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "embed_texts", lambda texts, **_: [[1.0, 0.5, 0.2] for _ in texts])
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps([_chunk("c1", "a"), _chunk("c2", "b"), _chunk("c3", "c")]))
    assert ingest.run(**_run_kwargs(tmp_path)) == 0

    chunks_path.write_text(json.dumps([_chunk("c1", "a")]))
    assert ingest.run(**_run_kwargs(tmp_path)) == 0

    assert open_collection(str(tmp_path / "chroma"), "test_col").get()["ids"] == ["c1"]
    assert list(load_manifest(str(tmp_path / "manifest.json"))["embedded_chunks"]) == ["c1"]


def test_model_change_rebuilds_index_with_new_dimension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "chunks.json").write_text(json.dumps([_chunk("c1", "a"), _chunk("c2", "b")]))
    monkeypatch.setattr(ingest, "embed_texts", lambda texts, **_: [[1.0, 0.5, 0.2] for _ in texts])
    assert ingest.run(**_run_kwargs(tmp_path, local_model="small")) == 0

    monkeypatch.setattr(ingest, "embed_texts", lambda texts, **_: [[1.0] * 8 for _ in texts])
    assert ingest.run(**_run_kwargs(tmp_path, local_model="bigger")) == 0

    collection = open_collection(str(tmp_path / "chroma"), "test_col")
    assert collection.count() == 2
    assert len(collection.get(include=["embeddings"])["embeddings"][0]) == 8
    assert load_manifest(str(tmp_path / "manifest.json"))["embedding_model"] == "bigger"
