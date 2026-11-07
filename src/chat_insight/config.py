"""Shared defaults. Every value can be overridden with an environment variable (see .env.example)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")

WHATSAPP_DB = os.getenv("WHATSAPP_DB", "./whatsapp-mcp/whatsapp-bridge/store/messages.db")
GROUP_NAME = os.getenv("GROUP_NAME") or None
EXPORT_CSV = os.getenv("EXPORT_CSV", f"{DATA_DIR}/export.csv")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", f"{DATA_DIR}/chunks.json")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", f"{DATA_DIR}/manifest.json")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", f"{DATA_DIR}/chroma")
COLLECTION = os.getenv("CHROMA_COLLECTION", "chat_chunks")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "30"))
STRIDE = int(os.getenv("STRIDE", "30"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gpt-4o-mini")
MAP_MODEL = os.getenv("MAP_MODEL", "gpt-4o-mini")
TOP_N = int(os.getenv("TOP_N", "8"))

# Injected into every prompt so the analyst persona matches the corpus.
CORPUS_DESCRIPTION = os.getenv(
    "CORPUS_DESCRIPTION",
    "a WhatsApp group of UK NHS clinicians (doctors, GPs and clinical staff) "
    "discussing AI, digital health and their day-to-day work",
)
