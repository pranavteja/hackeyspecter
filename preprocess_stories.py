"""Offline feature extraction and embedding storage for story summaries."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rag_config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, FEATURE_MODEL, get_client

logger = logging.getLogger(__name__)
FEATURE_SCHEMA: Dict[str, Any] = {
    "name": "story_features",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rich_descriptive_paragraph": {"type": "string"},
            "keywords": {"type": "array", "minItems": 5, "maxItems": 8, "items": {"type": "string"}},
            "emotional_scores": {
                "type": "object", "additionalProperties": False,
                "properties": {axis: {"type": "number", "minimum": 0, "maximum": 1} for axis in (
                    "valence", "energy", "warmth", "tension", "depth", "romance", "mystery", "nostalgia", "hope", "melancholy", "wonder", "humor"
                )},
                "required": ["valence", "energy", "warmth", "tension", "depth", "romance", "mystery", "nostalgia", "hope", "melancholy", "wonder", "humor"],
            },
        },
        "required": ["rich_descriptive_paragraph", "keywords", "emotional_scores"],
    },
}


def _extract_features(record: Dict[str, Any]) -> Dict[str, Any]:
    title, summary = record.get("title", "Untitled"), record.get("summary", "")
    if not summary.strip():
        raise ValueError(f"{title!r} has no summary")
    completion = get_client().chat.completions.create(
        model=FEATURE_MODEL,
        response_format={"type": "json_schema", "json_schema": FEATURE_SCHEMA},
        messages=[
            {"role": "system", "content": "Extract retrieval features from book summaries. Be faithful to the supplied text; do not invent plot facts."},
            {"role": "user", "content": f"Title: {title}\n\nSummary:\n{summary}"},
        ],
        temperature=0.2,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError(f"Feature model returned empty content for {title!r}")
    features = json.loads(content)
    payload = f"Title: {title} | Atmosphere: {', '.join(features['keywords'])} | Summary: {features['rich_descriptive_paragraph']}"
    return {**record, **features, "search_payload": payload}


def _embed_payloads(payloads: List[str], batch_size: int = 100) -> np.ndarray:
    vectors: List[List[float]] = []
    client = get_client()
    for start in range(0, len(payloads), batch_size):
        batch = payloads[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIMENSIONS)
        vectors.extend(item.embedding for item in response.data)
        logger.info("Embedded %d/%d stories", min(start + len(batch), len(payloads)), len(payloads))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Expected embeddings with {EMBEDDING_DIMENSIONS} dimensions; got {matrix.shape}")
    return matrix


def process_and_embed_dataset(input_json_path: str, output_db_path: str) -> Path:
    """Extract features concurrently and persist records plus vectors in a compressed NumPy store."""
    source_path, destination = Path(input_json_path), Path(output_db_path)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of story records.")

    processed: List[Dict[str, Any] | None] = [None] * len(records)
    max_workers = min(12, max(1, len(records)))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="feature-extractor") as executor:
        pending = {executor.submit(_extract_features, record): index for index, record in enumerate(records)}
        for future in as_completed(pending):
            index = pending[future]
            try:
                processed[index] = future.result()
            except Exception as exc:
                logger.exception("Skipping record %d (%r): %s", index, records[index].get("title"), exc)

    completed = [record for record in processed if record is not None]
    if not completed:
        raise RuntimeError("No stories were successfully processed.")
    embeddings = _embed_payloads([record["search_payload"] for record in completed])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, embeddings=embeddings, records_json=json.dumps(completed, ensure_ascii=False))
    logger.info("Saved %d embedded stories to %s", len(completed), destination)
    return destination
