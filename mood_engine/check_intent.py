"""LLM intent extraction and summary-based search for the story library."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_config import FEATURE_MODEL, get_client, response_reasoning_options

logger = logging.getLogger(__name__)
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "summary_1to16000.json"

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


def _terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", value.lower())


@lru_cache(maxsize=1)
def _story_summaries() -> list[dict[str, Any]]:
    """Load the supplied story-summary dataset once per running app process."""
    logger.info("Loading story summaries from %s", SUMMARY_PATH)
    with SUMMARY_PATH.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("summary_1to16000.json must contain a JSON list.")
    logger.info("Loaded %d story summaries", len(records))
    return records


def search_stories(intent: dict[str, Any], k: int = 5) -> list[tuple[dict[str, Any], float]]:
    """Rank the supplied story summaries using OpenAI's returned search terms."""
    if not isinstance(intent, dict):
        raise ValueError("intent must be an object returned by check_intent.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    query = intent.get("query", "")
    keywords = intent.get("keywords", [])
    if not isinstance(query, str) or not isinstance(keywords, list):
        raise ValueError("intent must contain a text query and a keyword list.")
    query_terms = Counter(_terms(" ".join([query, *(word for word in keywords if isinstance(word, str))])))
    if not query_terms:
        return []
    matches: list[tuple[dict[str, Any], float]] = []
    for item in _story_summaries():
        summary = " ".join([item.get("title", ""), item.get("summary", "")])
        summary_terms = Counter(_terms(summary))
        overlap = sum(query_terms[term] * summary_terms[term] for term in query_terms)
        if overlap:
            matches.append((item, round(overlap / max(1, sum(query_terms.values())), 2)))
    results = sorted(matches, key=lambda result: result[1], reverse=True)[:k]
    logger.info("Summary search completed: %d matches returned", len(results))
    return results
