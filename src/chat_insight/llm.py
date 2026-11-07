"""Chat-completion helper with rate-limit backoff and model fallback."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Protocol

from chat_insight import config


class ChatFn(Protocol):
    def __call__(self, messages: list[dict], model: str) -> str: ...


def chat(
    messages: list[dict],
    model: str,
    *,
    max_retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Deterministic (temperature 0) completion.

    Retries 429s with exponential backoff (4, 8, 16 … s) and falls back to
    FALLBACK_CHAT_MODEL if the requested model is unavailable on this key.
    """
    from openai import NotFoundError, OpenAI, RateLimitError

    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY is not set in .env")
    client = OpenAI()

    current = model
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=current,
                messages=messages,
                temperature=0,  # type: ignore[arg-type]
            )
            return resp.choices[0].message.content or ""
        except NotFoundError:
            if current == config.FALLBACK_CHAT_MODEL:
                raise
            current = config.FALLBACK_CHAT_MODEL
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 2)
            print(f"\n    Rate-limited — retrying in {wait}s…", flush=True)
            sleep(wait)
    raise RuntimeError("Exceeded max retries for LLM call")
