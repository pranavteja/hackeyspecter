"""Shared configuration and OpenAI client creation for the story RAG pipeline."""
from __future__ import annotations

import logging
import os
import re
import json
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

# Databricks secret path: scope=llm-secrets, key=openai-api-key.
# The valueFrom: reference in app.yaml handles injection on Databricks Apps;
# this function is a defensive fallback for environments where valueFrom is
# unavailable (local Databricks notebook, custom runtimes, etc.).
DATABRICKS_SECRET_SCOPE = "llm-secrets"
DATABRICKS_SECRET_KEY = "openai-api-key"


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
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in {"openai", "ollama", "groq", "gemini"}:
    raise ValueError("RAG_LLM_PROVIDER must be one of: openai, ollama, groq, gemini.")
if LLM_PROVIDER == "ollama":
    FEATURE_MODEL = os.getenv("RAG_FEATURE_MODEL", "llama3.2")
    RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "llama3.2")
    if FEATURE_MODEL.startswith("gpt-"):
        FEATURE_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    if RERANK_MODEL.startswith("gpt-"):
        RERANK_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
elif LLM_PROVIDER == "groq":
    FEATURE_MODEL = os.getenv("RAG_FEATURE_MODEL", "llama-3.3-70b-versatile")
    RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "llama-3.3-70b-versatile")
    if FEATURE_MODEL.startswith("gpt-"):
        FEATURE_MODEL = "llama-3.3-70b-versatile"
    if RERANK_MODEL.startswith("gpt-"):
        RERANK_MODEL = "llama-3.3-70b-versatile"
elif LLM_PROVIDER == "gemini":
    FEATURE_MODEL = os.getenv("RAG_FEATURE_MODEL", "gemini-2.0-flash")
    RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gemini-2.0-flash")
    if FEATURE_MODEL.startswith("gpt-"):
        FEATURE_MODEL = "gemini-2.0-flash"
    if RERANK_MODEL.startswith("gpt-"):
        RERANK_MODEL = "gemini-2.0-flash"
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


def _diagnostic_missing_key_help() -> str:
    """Return a checklist string explaining what to verify when the key is missing.

    Also dumps every env var name + length so we can see what the runtime actually
    exposes. Values are never included.
    """
    env_lines = [
        f"  {k} ({len(v)} chars)"
        for k, v in sorted(os.environ.items())
    ]
    env_block = "\n".join(env_lines) if env_lines else "  (no env vars visible)"
    return (
        "OPENAI_API_KEY is not set.\n"
        "On Databricks Apps, verify:\n"
        "  1. The secret is bound in App Resources UI\n"
        "  2. The 'Resource key' label in the UI matches app.yaml valueFrom\n"
        "  3. The deployment is current (push + redeploy after edits)\n"
        "Locally: set OPENAI_API_KEY in your .env file.\n"
        "\n"
        "ALL env vars visible to the process (no values):\n"
        f"{env_block}"
    )


def _discover_openai_key() -> str | None:
    """Search the entire environment for an OpenAI key under any name.

    Databricks Apps may inject the secret value under a different env var name
    than OPENAI_API_KEY depending on how it was bound. We scan every env var
    for a value that looks like an OpenAI key. First match wins.
    """
    # 1) Direct hit
    val = os.environ.get("OPENAI_API_KEY")
    if val and val.startswith("sk-"):
        return val
    # 2) Common alternate names
    for name in ("OPENAI_KEY", "DATABRICKS_OPENAI_API_KEY", "LLM_OPENAI_API_KEY"):
        val = os.environ.get(name)
        if val and val.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = val  # canonicalize for downstream
            logger.info("Found OpenAI key under env var %s", name)
            return val
    # 3) Brute-force: any env var whose value starts with sk- (OpenAI key prefix)
    for name, val in os.environ.items():
        if val.startswith("sk-") and "OPENAI" in name.upper():
            os.environ["OPENAI_API_KEY"] = val
            logger.info("Found OpenAI key under env var %s", name)
            return val
    return None


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return one configured, retrying SDK client per application process."""
    if LLM_PROVIDER == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            max_retries=0,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    if LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add a Groq API key or choose RAG_LLM_PROVIDER=ollama.")
        return OpenAI(
            api_key=api_key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            max_retries=OPENAI_MAX_RETRIES,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    api_key = _discover_openai_key()
    if not api_key and LLM_PROVIDER != "gemini":
        raise RuntimeError(_diagnostic_missing_key_help())
    if LLM_PROVIDER == "gemini":
        return OpenAI(api_key="gemini-placeholder", max_retries=0, timeout=OPENAI_TIMEOUT_SECONDS)
    return OpenAI(
        api_key=api_key,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )


def generate_json(model: str, instructions: str, user_input: str, schema: dict[str, object], *, max_output_tokens: int) -> str:
    """Generate JSON through the configured provider while keeping callers provider-agnostic."""
    if LLM_PROVIDER in {"openai"}:
        response = get_client().responses.create(
            model=model, instructions=instructions, input=user_input,
            text={"format": {"type": "json_schema", **schema}},
            max_output_tokens=max_output_tokens, store=False,
            **response_reasoning_options(model),
        )
        return response.output_text or ""
    if LLM_PROVIDER in {"ollama", "groq"}:
        response = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"},
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add a Gemini key or choose RAG_LLM_PROVIDER=ollama.")
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(model) + ":generateContent?key=" + urllib.parse.quote(api_key)
    payload = {"systemInstruction": {"parts": [{"text": instructions}]}, "contents": [{"role": "user", "parts": [{"text": user_input}]}], "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_output_tokens}}
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["candidates"][0]["content"]["parts"][0]["text"]
