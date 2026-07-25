"""LLM intent extraction and summary-based search for the story library."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_config import FEATURE_MODEL, get_client

logger = logging.getLogger(__name__)
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "summary_1to16000.json"

MODEL = FEATURE_MODEL


def check_intent(user_input: str) -> dict[str, Any]:
    """Ask OpenAI for search intent and return a stable JSON-shaped payload."""
    text = user_input.strip()
    if not text:
        raise ValueError("Enter a story, mood, or theme to search for.")

    logger.info("OpenAI intent model starting: model=%s, input=%r", MODEL, text)
    response = get_client().responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Extract entertainment-search intent. Return JSON only with intent (short phrase), keywords (3-8 lowercase strings), and query (a concise search query)."},
            {"role": "user", "content": text},
        ],
    )
    logger.info("OpenAI intent model output: %s", response.output_text)
    try:
        result = _parse_json_object(response.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("The model did not return valid JSON. Please try again.") from exc

    keywords = result.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
        raise ValueError("The model response did not include usable search keywords.")
    return {
        "intent": str(result.get("intent", "story search")).strip() or "story search",
        "keywords": [word.strip().lower() for word in keywords if word.strip()][:8],
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
    query_terms = Counter(_terms(" ".join([intent.get("query", ""), *intent.get("keywords", [])])))
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
