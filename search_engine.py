"""Fast runtime semantic search over the offline NumPy story vector store."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import Future
from threading import Lock
from time import perf_counter
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from rag_config import (
    QUERY_CACHE_SIZE,
    QUERY_EMBEDDING_MAX_RETRIES,
    QUERY_EMBEDDING_TIMEOUT_SECONDS,
    get_client,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    LLM_PROVIDER,
)

MAX_QUERY_CHARACTERS = 4_000
LEGACY_EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_STORE_VERSION = 2
NORMALIZATION_TOLERANCE = 5e-3

logger = logging.getLogger(__name__)

# lru_cache does not coalesce two simultaneous misses. This short-lived map does:
# duplicate clicks or concurrent sessions for the same query wait for one API call.
_embedding_flights: dict[tuple[str, str, int], Future[np.ndarray]] = {}
_embedding_flights_lock = Lock()


@dataclass(frozen=True)
class StoryDatabase:
    """Validated, unit-normalized vectors and their immutable source records."""

    records: tuple[dict[str, Any], ...]
    embeddings: np.ndarray
    embedding_model: str
    embedding_dimensions: int
    build_fingerprint: str


def _stable_record_id(record: dict[str, Any], index: int) -> str:
    existing_id = str(record.get("record_id") or record.get("cmu_id") or "").strip()
    if existing_id:
        return existing_id
    digest = hashlib.sha256(
        f"{record.get('title', '')}\0{record.get('summary', '')}".encode("utf-8")
    ).hexdigest()[:12]
    return f"story-{index}-{digest}"


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """Normalize corpus vectors once so query-time cosine search is a dot product."""
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("The vector store contains no valid embeddings.")
    if not np.isfinite(matrix).all():
        raise ValueError("The vector store contains non-finite embedding values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("The vector store contains zero-length embeddings.")
    normalized = np.ascontiguousarray(matrix / norms, dtype=np.float32)
    normalized.setflags(write=False)
    return normalized


def _validate_normalized_rows(embeddings: np.ndarray) -> np.ndarray:
    """Validate a modern, already-normalized store without creating another matrix."""
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("The vector store contains no valid embeddings.")
    if not np.isfinite(matrix).all():
        raise ValueError("The vector store contains non-finite embedding values.")
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)
    if np.any(np.abs(squared_norms - 1.0) > NORMALIZATION_TOLERANCE):
        raise ValueError("Vector-store metadata says vectors are normalized, but they are not. Rebuild the store.")
    matrix.setflags(write=False)
    return matrix


def canonicalize_query(user_prompt: str) -> str:
    """Normalize harmless Unicode and whitespace differences for cache reuse.

    Case and punctuation are intentionally preserved: they can change a story query's
    meaning (for example, a named title or acronym), whereas duplicate whitespace does not.
    """
    return " ".join(unicodedata.normalize("NFC", user_prompt).split())


def load_database(db_path: str) -> StoryDatabase:
    """Load, validate, and normalize a vector store produced by preprocessing."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Vector store does not exist: {path}")
    with np.load(path, allow_pickle=False) as store:
        if "embeddings" not in store.files or "records_json" not in store.files:
            raise ValueError("Vector store must contain embeddings and records_json.")
        embeddings = store["embeddings"].astype(np.float32, copy=False)
        raw_records = json.loads(str(store["records_json"].item()))
        metadata = json.loads(str(store["metadata_json"].item())) if "metadata_json" in store.files else {}

    if not isinstance(raw_records, list):
        raise ValueError("Vector-store records_json must decode to a list.")
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError("Vector-store records must all be JSON objects.")
    if embeddings.ndim != 2 or len(raw_records) != embeddings.shape[0]:
        raise ValueError("Vector store records and embeddings are inconsistent.")
    vectors_are_normalized = bool(metadata.get("vectors_normalized"))
    if metadata:
        if metadata.get("store_version") != VECTOR_STORE_VERSION:
            raise ValueError("Unsupported vector-store version. Rebuild it with process_and_embed_dataset.")
        if metadata.get("record_count") != len(raw_records):
            raise ValueError("Vector-store record count does not match its metadata.")
        if vectors_are_normalized is not True:
            raise ValueError("Vector-store metadata requires normalized vectors. Rebuild the store.")

    try:
        dimensions = int(metadata.get("embedding_dimensions", embeddings.shape[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Vector-store embedding dimensions are invalid.") from exc
    if dimensions != embeddings.shape[1]:
        raise ValueError("Vector-store embedding dimensions do not match its metadata.")
    model = str(metadata.get("embedding_model") or LEGACY_EMBEDDING_MODEL)
    records = tuple(
        {**record, "record_id": _stable_record_id(record, index)}
        for index, record in enumerate(raw_records)
    )
    record_ids = [record["record_id"] for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Vector store contains duplicate record IDs. Rebuild it with unique source IDs.")
    stat = path.stat()
    fingerprint = str(metadata.get("build_fingerprint") or f"legacy-{stat.st_mtime_ns}-{stat.st_size}")
    # Log store details so runtime users can verify which corpus is loaded.
    logger.info(
        "Loaded vector store: path=%s records=%d embedding_model=%s embedding_dimensions=%d build_fingerprint=%s",
        str(path),
        len(raw_records),
        model,
        dimensions,
        fingerprint,
    )
    print(f"[search_engine] Loaded vector store: path={path} records={len(raw_records)} model={model} dims={dimensions} fingerprint={fingerprint}")
    return StoryDatabase(
        records=records,
        # New stores are written normalized. Avoid an otherwise expensive second
        # full-matrix division during cold start; legacy stores are normalized once.
        embeddings=_validate_normalized_rows(embeddings) if vectors_are_normalized else _normalize_rows(embeddings),
        embedding_model=model,
        embedding_dimensions=dimensions,
        build_fingerprint=fingerprint,
    )


@lru_cache(maxsize=QUERY_CACHE_SIZE)
def _embed_normalized_query(prompt: str, model: str, dimensions: int) -> np.ndarray:
    """Embed an exact repeated query once per process and return an immutable unit vector."""
    print(f"[search_engine] embedding request starting: model={model} dims={dimensions}")
    response = get_client().with_options(
        timeout=QUERY_EMBEDDING_TIMEOUT_SECONDS,
        max_retries=QUERY_EMBEDDING_MAX_RETRIES,
    ).embeddings.create(
        model=model,
        input=prompt,
        dimensions=dimensions,
    )
    if not response.data:
        raise ValueError("Embedding model returned no query vector.")
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise ValueError("Embedding model returned an invalid query vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding model returned a zero-length query vector.")
    normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
    # lru_cache returns the same object for a cache hit. Mark it read-only so a
    # caller cannot corrupt later searches by mutating the cached vector.
    normalized.setflags(write=False)
    print(f"[search_engine] embedding request finished: model={model} dims={dimensions} len={normalized.shape[0]}")
    return normalized


def _cache_hit_count() -> int:
    """Read lru-cache telemetry safely, including when tests replace the function."""
    cache_info = getattr(_embed_normalized_query, "cache_info", None)
    if not callable(cache_info):
        return 0
    info = cache_info()
    hits = getattr(info, "hits", None)
    return hits if isinstance(hits, int) else 0


def _get_normalized_query(prompt: str, model: str, dimensions: int) -> tuple[np.ndarray, bool]:
    """Return a query vector, coalescing concurrent requests for the same prompt.

    The second return value says whether the vector was reused from either the
    LRU cache or an in-flight request. It is only telemetry and never exposes a
    user's prompt in logs.
    """
    key = (prompt, model, dimensions)
    with _embedding_flights_lock:
        future = _embedding_flights.get(key)
        leader = future is None
        if leader:
            future = Future()
            _embedding_flights[key] = future

    if not leader:
        return future.result(), True

    hits_before = _cache_hit_count()
    try:
        vector = np.asarray(_embed_normalized_query(prompt, model, dimensions), dtype=np.float32)
        reused = _cache_hit_count() > hits_before
        future.set_result(vector)
        return vector, reused
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _embedding_flights_lock:
            _embedding_flights.pop(key, None)


def vector_search(user_prompt: str, database: StoryDatabase, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed one user query and return its top cosine-similar stored stories."""
    if not isinstance(user_prompt, str):
        raise ValueError("Search input must be text.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")
    prompt = canonicalize_query(user_prompt)
    if not prompt:
        raise ValueError("Enter a story, mood, or theme to search for.")
    if len(prompt) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"Search input must be at most {MAX_QUERY_CHARACTERS:,} characters.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not database.records:
        raise ValueError("The vector store contains no searchable stories.")
    if LLM_PROVIDER != "openai" and os.getenv("RAG_SEARCH_MODE", "auto").lower() in {"auto", "lexical"}:
        return _local_vector_search(prompt, database.records, top_k)
    if database.embeddings.ndim != 2 or database.embeddings.shape != (
        len(database.records),
        database.embedding_dimensions,
    ):
        raise ValueError("The in-memory vector store has inconsistent record or embedding dimensions.")

    embedding_started = perf_counter()
    query, embedding_reused = _get_normalized_query(
        prompt,
        database.embedding_model,
        database.embedding_dimensions,
    )
    # Emit runtime telemetry to help verify which embedding model and
    # corpus fingerprint were used for this search.
    if database.embedding_model != EMBEDDING_MODEL:
        logger.warning(
            "Query embedding model mismatch: runtime_embedding_model=%s store_embedding_model=%s",
            EMBEDDING_MODEL,
            database.embedding_model,
        )
        print(f"[search_engine] WARNING: Query embedding model mismatch: runtime={EMBEDDING_MODEL} store={database.embedding_model}")
    logger.info(
        "Vector search starting: corpus_embeddings=%d store_dims=%d requested_top_k=%d "
        "store_fingerprint=%s store_model=%s embedding_reused=%s",
        database.embeddings.shape[0],
        database.embedding_dimensions,
        top_k,
        getattr(database, "build_fingerprint", "unknown"),
        database.embedding_model,
        embedding_reused,
    )
    print(
        f"[search_engine] Vector search starting: corpus_embeddings={database.embeddings.shape[0]} "
        f"store_dims={database.embedding_dimensions} requested_top_k={top_k} "
        f"store_fingerprint={getattr(database, 'build_fingerprint', 'unknown')} "
        f"store_model={database.embedding_model} embedding_reused={embedding_reused}"
    )
    embedding_elapsed_ms = (perf_counter() - embedding_started) * 1_000
    ranking_started = perf_counter()
    scores = database.embeddings @ query
    limit = min(top_k, len(database.records))
    # Argpartition avoids sorting the entire corpus when the database grows.
    selected = np.argpartition(scores, scores.size - limit)[-limit:]
    ordered = selected[np.argsort(scores[selected])[::-1]]
    results = [
        {**database.records[int(index)], "vector_similarity": float(np.clip(scores[index], -1.0, 1.0))}
        for index in ordered
    ]
    ranking_elapsed_ms = (perf_counter() - ranking_started) * 1_000
    logger.info(
        "Story retrieval complete: embedding_reused=%s embedding_ms=%.1f ranking_ms=%.3f candidates=%d",
        embedding_reused,
        embedding_elapsed_ms,
        ranking_elapsed_ms,
        len(results),
    )
    return results


def _lexical_search(prompt: str, records: tuple[dict[str, Any], ...], top_k: int) -> list[dict[str, Any]]:
    """Backward-compatible alias for the local vector fallback."""
    return _local_vector_search(prompt, records, top_k)


def _local_vector_search(prompt: str, records: tuple[dict[str, Any], ...], top_k: int) -> list[dict[str, Any]]:
    """Offline TF-IDF cosine search when the neural embedding service is unavailable."""
    import heapq
    import re
    from collections import Counter
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "want", "find", "story", "about", "something"}
    def tokens(value: str) -> list[str]:
        return [term for term in re.findall(r"[a-z0-9]{3,}", value.lower()) if term not in stopwords]

    query_tokens = tokens(prompt)
    query_counts = Counter(query_tokens)
    document_frequency: Counter[str] = Counter()
    for record in records:
        title = str(record.get("title", "")).lower()
        # Repeat title tokens to make titles slightly more influential.
        text = " ".join(str(record.get(key, "")) for key in ("title", "summary", "rich_descriptive_paragraph", "keywords"))
        document_frequency.update(set(tokens(text) + tokens(title)))
    if not query_counts:
        return []
    total_documents = len(records)
    query_weights = {
        term: count * (np.log((1.0 + total_documents) / (1.0 + document_frequency[term])) + 1.0)
        for term, count in query_counts.items()
    }
    query_norm = max(sum(weight * weight for weight in query_weights.values()) ** 0.5, 1e-8)
    top: list[tuple[float, str, int]] = []
    for row, record in enumerate(records):
        title = str(record.get("title", "")).lower()
        text = " ".join(str(record.get(key, "")) for key in ("title", "summary", "rich_descriptive_paragraph", "keywords"))
        counts = Counter(tokens(text) + tokens(title))
        weights = {
            term: count * (np.log((1.0 + total_documents) / (1.0 + document_frequency[term])) + 1.0)
            for term, count in counts.items()
        }
        dot = sum(query_weights.get(term, 0.0) * weight for term, weight in weights.items())
        norm = max(sum(weight * weight for weight in weights.values()) ** 0.5, 1e-8)
        score = float(dot / (query_norm * norm))
        tie = hashlib.sha256(f"{prompt}\0{record.get('record_id', '')}".encode("utf-8")).hexdigest()
        item = (score, tie, row)
        if len(top) < top_k:
            heapq.heappush(top, item)
        elif item > top[0]:
            heapq.heapreplace(top, item)
    selected = sorted(top, reverse=True)
    return [{**records[row], "vector_similarity": score} for score, _, row in selected]
