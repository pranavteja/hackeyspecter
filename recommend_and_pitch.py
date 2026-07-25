"""LLM reranking and pitch generation for retrieved story candidates."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from rag_config import RERANK_MODEL, get_client

RECOMMENDATION_SCHEMA: Dict[str, Any] = {
    "name": "final_recommendation", "strict": True,
    "schema": {"type": "object", "additionalProperties": False, "properties": {
        "recommended_book_title": {"type": "string"}, "audio_url": {"type": "string"},
        "pitch_script": {"type": "string"}, "emotional_match_reasons": {"type": "array", "items": {"type": "string"}},
    }, "required": ["recommended_book_title", "audio_url", "pitch_script", "emotional_match_reasons"]},
}


def generate_final_recommendation(user_prompt: str, top_5_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rerank retrieved stories and return a two-sentence, TTS-ready recommendation pitch."""
    if not top_5_candidates:
        raise ValueError("At least one retrieved candidate is required.")
    compact_candidates = [{
        "title": item.get("title"), "audio_url": item.get("audio_zip_url", ""),
        "description": item.get("rich_descriptive_paragraph", "")[:1800],
        "keywords": item.get("keywords", []), "emotional_scores": item.get("emotional_scores", {}),
    } for item in top_5_candidates]
    completion = get_client().chat.completions.create(
        model=RERANK_MODEL,
        response_format={"type": "json_schema", "json_schema": RECOMMENDATION_SCHEMA},
        messages=[
            {"role": "system", "content": "Select only from the supplied candidates. Analyze the user's mood, choose the closest emotional fit, and write exactly two compelling pitch sentences."},
            {"role": "user", "content": json.dumps({"user_prompt": user_prompt, "candidates": compact_candidates}, ensure_ascii=False)},
        ],
        temperature=0.5,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Reranking model returned empty content.")
    result = json.loads(content)
    valid_titles = {candidate["title"] for candidate in compact_candidates}
    if result["recommended_book_title"] not in valid_titles:
        raise ValueError("Reranking model selected a story outside the retrieved candidates.")
    return result
