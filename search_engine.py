"""Runtime semantic vector search over the offline NumPy story store."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rag_config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, get_client


@dataclass(frozen=True)
class StoryDatabase:
    records: List[Dict[str, Any]]
    embeddings: np.ndarray


def load_database(db_path: str) -> StoryDatabase:
    """Load a compressed vector store produced by process_and_embed_dataset."""
    with np.load(Path(db_path), allow_pickle=False) as store:
        embeddings = store["embeddings"].astype(np.float32)
        records = json.loads(str(store["records_json"].item()))
    if embeddings.ndim != 2 or len(records) != len(embeddings):
        raise ValueError("Vector store records and embeddings are inconsistent.")
    return StoryDatabase(records=records, embeddings=embeddings)


def vector_search(user_prompt: str, database: StoryDatabase, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed a mood request and return its cosine-similar story candidates."""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=user_prompt, dimensions=EMBEDDING_DIMENSIONS)
    query = np.asarray(response.data[0].embedding, dtype=np.float32)
    norms = np.linalg.norm(database.embeddings, axis=1) * np.linalg.norm(query)
    similarities = np.divide(database.embeddings @ query, norms, out=np.zeros_like(norms), where=norms > 0)
    indexes = np.argsort(similarities)[::-1][:min(top_k, len(database.records))]
    return [{**database.records[int(index)], "vector_similarity": float(similarities[index])} for index in indexes]
