"""Shared configuration and OpenAI client creation for the story RAG pipeline."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from openai import OpenAI


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from the repository-local `.env` file."""
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            and value
            and key not in os.environ
        ):
            os.environ[key] = value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be zero or greater.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


# System environment variables intentionally override local development values.
_load_local_env()

# These defaults prioritize retrieval quality. Override them in `.env` when a
# cheaper or lower-latency deployment profile is required.
FEATURE_MODEL = os.getenv("RAG_FEATURE_MODEL", "gpt-5.6-sol")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gpt-5.6-terra")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
_DEFAULT_EMBEDDING_DIMENSIONS = 1536 if EMBEDDING_MODEL == "text-embedding-3-small" else 3072
EMBEDDING_DIMENSIONS = _positive_int("RAG_EMBEDDING_DIMENSIONS", _DEFAULT_EMBEDDING_DIMENSIONS)
REASONING_EFFORT = os.getenv("RAG_REASONING_EFFORT", "low")
OPENAI_TIMEOUT_SECONDS = _positive_float("OPENAI_TIMEOUT_SECONDS", 45.0)
OPENAI_MAX_RETRIES = _positive_int("OPENAI_MAX_RETRIES", 3)
# Interactive retrieval should fail fast rather than stacking the SDK's long
# timeout/retry window behind the search button. Offline preprocessing keeps the
# longer default client policy above.
QUERY_EMBEDDING_TIMEOUT_SECONDS = _positive_float("RAG_QUERY_EMBEDDING_TIMEOUT_SECONDS", 12.0)
QUERY_EMBEDDING_MAX_RETRIES = _nonnegative_int("RAG_QUERY_EMBEDDING_MAX_RETRIES", 1)
TTS_TIMEOUT_SECONDS = _positive_float("RAG_TTS_TIMEOUT_SECONDS", 90.0)
TTS_MAX_RETRIES = _nonnegative_int("RAG_TTS_MAX_RETRIES", 1)
QUERY_CACHE_SIZE = _positive_int("RAG_QUERY_CACHE_SIZE", 256)
TTS_CACHE_SIZE = _positive_int("RAG_TTS_CACHE_SIZE", 16)
# `gpt-audio-1.5` is the current quality-oriented voice model. Set this to
# `gpt-audio-mini` in .env when lower cost is more important than voice quality.
TTS_MODEL = os.getenv("RAG_TTS_MODEL", "gpt-audio-1.5")
TRANSCRIPTION_MODEL = os.getenv("RAG_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")

_MAX_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
if EMBEDDING_MODEL in _MAX_EMBEDDING_DIMENSIONS and EMBEDDING_DIMENSIONS > _MAX_EMBEDDING_DIMENSIONS[EMBEDDING_MODEL]:
    raise ValueError(
        f"{EMBEDDING_MODEL} supports at most {_MAX_EMBEDDING_DIMENSIONS[EMBEDDING_MODEL]} dimensions; "
        f"received {EMBEDDING_DIMENSIONS}."
    )
if REASONING_EFFORT not in {"none", "low", "medium", "high", "xhigh", "max"}:
    raise ValueError("RAG_REASONING_EFFORT must be one of none, low, medium, high, xhigh, or max.")


def response_reasoning_options(model: str) -> dict[str, object]:
    """Apply GPT-5 reasoning controls only to model families that support them."""
    if model.startswith("gpt-5"):
        return {"reasoning": {"effort": REASONING_EFFORT}}
    return {}


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return one configured, retrying SDK client per application process."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env or the deployment environment.")
    return OpenAI(
        api_key=api_key,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
