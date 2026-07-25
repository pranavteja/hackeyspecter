"""Shared configuration and OpenAI client creation for the story RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from the repository's local .env file."""
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# Existing system environment variables take precedence over local development values.
_load_local_env()

FEATURE_MODEL = os.getenv("RAG_FEATURE_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536


def get_client() -> OpenAI:
    """Create an authenticated SDK client without storing credentials in source code."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the RAG pipeline.")
    return OpenAI(api_key=api_key, max_retries=3, timeout=60.0)
