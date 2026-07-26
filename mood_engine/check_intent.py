"""LLM intent extraction and summary-based search for the story library."""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_config import FEATURE_MODEL, get_client, response_reasoning_options

logger = logging.getLogger(__name__)
VECTOR_STORE_FILENAME = "ultimate_pocketfm_vector_store.npz"
DEFAULT_VECTOR_STORE_PATH = Path(__file__).resolve().parents[1] / VECTOR_STORE_FILENAME


def _resolve_vector_store_path() -> Path:
    """Resolve a local override first, then a Databricks UC volume resource."""
    explicit_path = os.getenv("RAG_VECTOR_STORE_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser()

    volume_path = os.getenv("RAG_VECTOR_STORE_VOLUME", "").strip()
    if volume_path:
        return Path(volume_path).expanduser() / VECTOR_STORE_FILENAME

    return DEFAULT_VECTOR_STORE_PATH


VECTOR_STORE_PATH = _resolve_vector_store_path()

MODEL = FEATURE_MODEL
MAX_INPUT_CHARACTERS = 4_000
INTENT_SCHEMA: dict[str, Any] = {
    "name": "story_search_intent",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "minLength": 1, "maxLength": 160},
            "keywords": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "query": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["intent", "keywords", "query"],
    },
}


def check_intent(user_input: str) -> dict[str, Any]:
    """Ask OpenAI for search intent and return a stable JSON-shaped payload."""
    if not isinstance(user_input, str):
        raise ValueError("Enter a story, mood, or theme as text.")
    text = user_input.strip()
    if not text:
        raise ValueError("Enter a story, mood, or theme to search for.")
    if len(text) > MAX_INPUT_CHARACTERS:
        raise ValueError(f"Search input must be at most {MAX_INPUT_CHARACTERS:,} characters.")

    logger.info("OpenAI intent model starting: model=%s", MODEL)
    response = get_client().responses.create(
        model=MODEL,
        instructions=(
            "Extract entertainment-search intent. Return a concise intent, 3-8 lowercase "
            "keywords, and a concise search query. Do not include unsupported claims."
        ),
        input=text,
        text={"format": {"type": "json_schema", **INTENT_SCHEMA}},
        max_output_tokens=220,
        store=False,
        prompt_cache_key="story-intent-v1",
        **response_reasoning_options(MODEL),
    )
    logger.info("OpenAI intent model output received: %d characters", len(response.output_text or ""))
    try:
        result = _parse_json_object(response.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("The model did not return valid JSON. Please try again.") from exc

    keywords = result.get("keywords", [])
    if not isinstance(keywords, list) or not 3 <= len(keywords) <= 8 or not all(isinstance(word, str) for word in keywords):
        raise ValueError("The model response did not include usable search keywords.")
    cleaned_keywords = [word.strip().lower() for word in keywords if word.strip()]
    if not 3 <= len(cleaned_keywords) <= 8:
        raise ValueError("The model response did not include usable search keywords.")
    return {
        "intent": str(result.get("intent", "story search")).strip() or "story search",
        "keywords": cleaned_keywords,
        "query": str(result.get("query", text)).strip() or text,
        "user_input": text,
    }


def _parse_json_object(model_output: str) -> dict[str, Any]:
    """Parse JSON even if the model wraps it in a Markdown code fence."""
    cleaned = model_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Expected a JSON object", cleaned, start)
    return value


def _vector_store_signature() -> str:
    if not VECTOR_STORE_PATH.is_file():
        raise FileNotFoundError(
            "The offline vector store is missing. Run import_precomputed_embeddings first."
        )
    stat = VECTOR_STORE_PATH.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@lru_cache(maxsize=2)
def _story_database(path: str, signature: str) -> Any:
    """Load the already-imported vectors, never the multi-gigabyte source JSON."""
    from search_engine import load_database

    logger.info("Loading offline story vectors from %s", path)
    return load_database(path)


def search_stories(intent: dict[str, Any], k: int = 5) -> list[tuple[dict[str, Any], float]]:
    """Preserve the legacy tuple contract while using the fast vector store."""
    if not isinstance(intent, dict):
        raise ValueError("intent must be an object returned by check_intent.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    query = intent.get("query", "")
    keywords = intent.get("keywords", [])
    if not isinstance(query, str) or not isinstance(keywords, list):
        raise ValueError("intent must contain a text query and a keyword list.")
    search_prompt = " ".join([query, *(word for word in keywords if isinstance(word, str))]).strip()
    if not search_prompt:
        return []
    from search_engine import vector_search

    database = _story_database(str(VECTOR_STORE_PATH), _vector_store_signature())
    matches = vector_search(search_prompt, database, top_k=k)
    results = [(item, float(item["vector_similarity"])) for item in matches]
    logger.info("Vector summary search completed: %d matches returned", len(results))
    return results
