"""High-quality LLM reranking and pitch generation for retrieved story candidates."""
from __future__ import annotations

import json
import logging
from typing import Any

from rag_config import RERANK_MODEL, LLM_PROVIDER, generate_json, get_client, response_reasoning_options

logger = logging.getLogger(__name__)

# Feature-enriched stores carry atmospheric keywords and emotional axes; stores
# imported from supplied embeddings use the original summary as their compact
# fallback. Keeping the LLM context bounded still cuts rerank latency.
MAX_CANDIDATE_DESCRIPTION_CHARACTERS = 900
MAX_RERANK_OUTPUT_TOKENS = 260
MAX_RERANK_CANDIDATES = 5
MAX_USER_PROMPT_CHARACTERS = 4_000
RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "name": "final_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recommended_record_id": {"type": "string", "minLength": 1},
            "pitch_script": {"type": "string", "minLength": 1, "maxLength": 520},
            "emotional_match_reasons": {
                "type": "array", "minItems": 1, "maxItems": 4,
                "items": {"type": "string", "minLength": 1, "maxLength": 180},
            },
        },
        "required": ["recommended_record_id", "pitch_script", "emotional_match_reasons"],
    },
}


def _candidate_payload(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Every retrieved candidate must be an object.")
    record_id = str(item.get("record_id", "")).strip()
    if not record_id:
        raise ValueError("Every retrieved candidate must include a stable record_id.")
    raw_keywords = item.get("keywords", [])
    keywords = raw_keywords if isinstance(raw_keywords, list) else []
    raw_emotional_scores = item.get("emotional_scores", {})
    emotional_scores = raw_emotional_scores if isinstance(raw_emotional_scores, dict) else {}
    return {
        "record_id": record_id,
        "title": str(item.get("title", "Untitled")),
        "description": str(item.get("rich_descriptive_paragraph") or item.get("summary", ""))[
            :MAX_CANDIDATE_DESCRIPTION_CHARACTERS
        ],
        "keywords": [str(keyword).strip()[:80] for keyword in keywords if str(keyword).strip()][:8],
        "emotional_scores": emotional_scores,
        "vector_similarity": round(float(item.get("vector_similarity", 0.0)), 6),
    }


def generate_final_recommendation(user_prompt: str, top_5_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Rerank retrieved stories while deriving title and audio metadata only from source data."""
    if not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be text.")
    if not isinstance(top_5_candidates, list):
        raise ValueError("top_5_candidates must be a list.")
    prompt = user_prompt.strip()
    if not prompt:
        raise ValueError("user_prompt cannot be empty.")
    if len(prompt) > MAX_USER_PROMPT_CHARACTERS:
        raise ValueError(f"user_prompt must be at most {MAX_USER_PROMPT_CHARACTERS:,} characters.")
    if not top_5_candidates:
        raise ValueError("At least one retrieved candidate is required.")
    if len(top_5_candidates) > MAX_RERANK_CANDIDATES:
        raise ValueError(f"At most {MAX_RERANK_CANDIDATES} retrieved candidates may be reranked.")

    candidates = [_candidate_payload(item) for item in top_5_candidates]
    by_id = {candidate["record_id"]: original for candidate, original in zip(candidates, top_5_candidates)}
    if len(by_id) != len(candidates):
        raise ValueError("Retrieved candidates must have unique record IDs.")

    instructions = (
            "Analyze the user's requested mood and select exactly one supplied candidate. "
            "Never invent or select a record ID. Write exactly two compelling sentences for pitch_script. "
            "Explain the emotional fit in concise emotional_match_reasons."
    )
    user_input = json.dumps({"user_prompt": prompt, "candidates": candidates}, ensure_ascii=False)
    if LLM_PROVIDER == "openai":
        response = get_client().responses.create(
            model=RERANK_MODEL, instructions=instructions, input=user_input,
            text={"format": {"type": "json_schema", **RECOMMENDATION_SCHEMA}},
            max_output_tokens=MAX_RERANK_OUTPUT_TOKENS, store=False,
            prompt_cache_key="story-rerank-v2", **response_reasoning_options(RERANK_MODEL),
        )
        output_text = response.output_text or ""
    else:
        try:
            output_text = generate_json(RERANK_MODEL, instructions, user_input, RECOMMENDATION_SCHEMA, max_output_tokens=MAX_RERANK_OUTPUT_TOKENS)
        except Exception as exc:
            # Local mode must remain usable when Ollama is stopped or not installed.
            # The retrieved ranking is already deterministic, so select its first item
            # and build a bounded pitch without any network call.
            if LLM_PROVIDER != "ollama":
                raise
            logger.warning("Local pitch model unavailable; using deterministic fallback: %s", exc)
            selected = top_5_candidates[0]
            title = str(selected.get("title", "this story")).strip() or "this story"
            keywords = [str(value).strip().lower() for value in selected.get("keywords", []) if str(value).strip()][:3]
            reasons = [f"It matches your request for {prompt[:120]}."]
            if keywords:
                reasons.append("Its themes include " + ", ".join(keywords) + ".")
            return {
                "recommended_record_id": str(selected.get("record_id", "")),
                "recommended_book_title": title,
                "audio_url": str(selected.get("audio_url") or selected.get("audio_zip_url") or ""),
                "pitch_script": f"{title} is a strong match for the mood you described. Start here for a story shaped around that feeling.",
                "emotional_match_reasons": reasons,
            }
    if not output_text:
        raise ValueError("Reranking model returned empty content.")
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Reranking model returned invalid JSON.") from exc

    selected_id = str(result.get("recommended_record_id", ""))
    selected = by_id.get(selected_id)
    if selected is None:
        raise ValueError("Reranking model selected a story outside the retrieved candidates.")
    reasons = result.get("emotional_match_reasons")
    pitch = result.get("pitch_script")
    if not isinstance(pitch, str) or not pitch.strip() or len(pitch.strip()) > 520 or not isinstance(reasons, list):
        raise ValueError("Reranking model returned an incomplete recommendation.")
    cleaned_reasons = [str(reason).strip() for reason in reasons if str(reason).strip()]
    if not cleaned_reasons:
        raise ValueError("Reranking model returned no usable emotional match reasons.")

    # Source metadata is authoritative; the model never supplies title or URL.
    audio_url = str(selected.get("audio_url") or selected.get("audio_zip_url") or "")
    return {
        "recommended_record_id": selected_id,
        "recommended_book_title": str(selected.get("title", "Untitled")),
        "audio_url": audio_url,
        "pitch_script": pitch.strip(),
        "emotional_match_reasons": cleaned_reasons,
    }
