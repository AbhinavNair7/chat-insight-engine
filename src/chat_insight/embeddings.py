"""Embedding backends: OpenAI (default) or a local sentence-transformers model (no API key)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def embed_openai(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY is not set. Add it to .env or use --local-embeddings.")
    response = OpenAI().embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


@lru_cache(maxsize=2)
def _local_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    vecs = _local_model(model_name).encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_texts(texts: list[str], *, use_local: bool, openai_model: str, local_model: str) -> list[list[float]]:
    if use_local:
        return embed_local(texts, local_model)
    return embed_openai(texts, openai_model)
