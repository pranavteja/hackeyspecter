"""Offline, resumable feature extraction and embedding storage for story summaries."""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import numpy as np

from rag_config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FEATURE_MODEL,
    get_client,
    response_reasoning_options,
)

logger = logging.getLogger(__name__)
VECTOR_STORE_VERSION = 2
FEATURE_CACHE_VERSION = 1
FEATURE_CACHE_SUFFIX = ".features-cache.json"
MAX_SOURCE_SUMMARY_CHARACTERS = 12_000
DEFAULT_MAX_WORKERS = 8
DEFAULT_EMBEDDING_BATCH_SIZE = 96
FEATURE_EXTRACTION_ATTEMPTS = 2
JSON_STREAM_CHUNK_BYTES = 1_048_576
PRECOMPUTED_IMPORT_VERSION = 1
# The supplied master dataset has 3,072-dimensional vectors. That is the full
# size of OpenAI's text-embedding-3-large output. Override this argument only
# when you know the source JSON was embedded with a different model.
DEFAULT_PRECOMPUTED_EMBEDDING_MODEL = "text-embedding-3-large"
EMOTIONAL_AXES = (
    "valence", "energy", "warmth", "tension", "depth", "romance",
    "mystery", "nostalgia", "hope", "melancholy", "wonder", "humor",
)
FEATURE_SCHEMA: dict[str, Any] = {
    "name": "story_features",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rich_descriptive_paragraph": {"type": "string", "minLength": 1},
            "keywords": {
                "type": "array", "minItems": 5, "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
            "emotional_scores": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    axis: {"type": "number", "minimum": 0, "maximum": 1}
                    for axis in EMOTIONAL_AXES
                },
                "required": list(EMOTIONAL_AXES),
            },
        },
        "required": ["rich_descriptive_paragraph", "keywords", "emotional_scores"],
    },
}


def _record_id(record: dict[str, Any], index: int) -> str:
    supplied = str(record.get("record_id") or record.get("cmu_id") or "").strip()
    if supplied:
        return supplied
    digest = hashlib.sha256(
        f"{record.get('title', '')}\0{record.get('summary', '')}".encode("utf-8")
    ).hexdigest()[:12]
    return f"story-{index}-{digest}"


def _source_record_hash(record: dict[str, Any]) -> str:
    """Hash only fields that affect feature extraction and the search embedding."""
    material = json.dumps(
        {"title": str(record.get("title", "")), "summary": str(record.get("summary", ""))},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_features(features: Any, title: str) -> dict[str, Any]:
    if not isinstance(features, dict):
        raise ValueError(f"Feature model returned a non-object for {title!r}.")
    paragraph = features.get("rich_descriptive_paragraph")
    keywords = features.get("keywords")
    scores = features.get("emotional_scores")
    if not isinstance(paragraph, str) or not paragraph.strip():
        raise ValueError(f"Feature model returned no descriptive paragraph for {title!r}.")
    if not isinstance(keywords, list) or not 5 <= len(keywords) <= 8:
        raise ValueError(f"Feature model returned an invalid keyword list for {title!r}.")
    cleaned_keywords = [str(word).strip().lower() for word in keywords if str(word).strip()]
    if len(cleaned_keywords) != len(keywords):
        raise ValueError(f"Feature model returned an empty keyword for {title!r}.")
    if not isinstance(scores, dict) or set(scores) != set(EMOTIONAL_AXES):
        raise ValueError(f"Feature model returned incomplete emotional scores for {title!r}.")
    normalized_scores = {axis: float(scores[axis]) for axis in EMOTIONAL_AXES}
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized_scores.values()):
        raise ValueError(f"Feature model returned out-of-range emotional scores for {title!r}.")
    return {
        "rich_descriptive_paragraph": paragraph.strip(),
        "keywords": cleaned_keywords,
        "emotional_scores": normalized_scores,
    }


def _search_payload(title: str, features: dict[str, Any]) -> str:
    return (
        f"Title: {title} | Atmosphere: {', '.join(features['keywords'])} | "
        f"Summary: {features['rich_descriptive_paragraph']}"
    )


def _processed_record_from_features(record: dict[str, Any], index: int, features: dict[str, Any]) -> dict[str, Any]:
    title = str(record.get("title", "")).strip() or "Untitled"
    return {
        **record,
        "record_id": _record_id(record, index),
        "source_record_hash": _source_record_hash(record),
        **features,
        "search_payload": _search_payload(title, features),
    }


def _feature_cache_path(destination: Path) -> Path:
    return destination.with_suffix(FEATURE_CACHE_SUFFIX)


def _load_feature_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load durable, model-specific extraction results from an interrupted build."""
    if not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("cache_version") != FEATURE_CACHE_VERSION
            or payload.get("feature_model") != FEATURE_MODEL
            or not isinstance(payload.get("entries"), dict)
        ):
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for source_hash, features in payload["entries"].items():
            if not isinstance(source_hash, str):
                continue
            try:
                entries[source_hash] = _validate_features(features, "cached story")
            except ValueError:
                continue
        return entries
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Ignoring invalid feature checkpoint at %s", cache_path)
        return {}


def _write_feature_cache(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Atomically preserve successful feature calls before later embedding work begins."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "cache_version": FEATURE_CACHE_VERSION,
            "feature_model": FEATURE_MODEL,
            "entries": entries,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with NamedTemporaryFile(dir=cache_path.parent, suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
    try:
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_features(record: dict[str, Any], index: int) -> dict[str, Any]:
    title = str(record.get("title", "")).strip() or "Untitled"
    summary = str(record.get("summary", "")).strip()
    if not summary:
        raise ValueError(f"{title!r} has no summary.")
    source_text = summary[:MAX_SOURCE_SUMMARY_CHARACTERS]
    response = get_client().responses.create(
        model=FEATURE_MODEL,
        instructions=(
            "Extract high-fidelity retrieval features from the supplied book summary. "
            "Preserve plot facts; do not invent events, characters, or settings."
        ),
        input=f"Title: {title}\n\nSummary:\n{source_text}",
        text={"format": {"type": "json_schema", **FEATURE_SCHEMA}},
        max_output_tokens=900,
        store=False,
        prompt_cache_key="story-feature-extraction-v2",
        **response_reasoning_options(FEATURE_MODEL),
    )
    if not response.output_text:
        raise ValueError(f"Feature model returned empty content for {title!r}.")
    features = _validate_features(json.loads(response.output_text), title)
    return _processed_record_from_features(record, index, features)


def _extract_features_with_retry(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Retry a transient malformed/empty model response without retrying bad source data forever."""
    title = str(record.get("title", "Untitled"))
    for attempt in range(1, FEATURE_EXTRACTION_ATTEMPTS + 1):
        try:
            return _extract_features(record, index)
        except ValueError as exc:
            # A missing source summary cannot be repaired by making another API call.
            if "has no summary" in str(exc) or attempt == FEATURE_EXTRACTION_ATTEMPTS:
                raise
            logger.warning(
                "Feature response validation failed for %r; retrying (%d/%d).",
                title,
                attempt + 1,
                FEATURE_EXTRACTION_ATTEMPTS,
            )
        except Exception:
            if attempt == FEATURE_EXTRACTION_ATTEMPTS:
                raise
            logger.warning(
                "Feature request failed for %r; retrying (%d/%d).",
                title,
                attempt + 1,
                FEATURE_EXTRACTION_ATTEMPTS,
            )
    raise RuntimeError("Feature extraction retry loop ended unexpectedly.")


def _embed_payloads(payloads: list[str], batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    vectors: list[list[float]] = []
    client = get_client()
    for start in range(0, len(payloads), batch_size):
        batch = payloads[start:start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        returned_indexes = [item.index for item in ordered]
        if returned_indexes != list(range(len(batch))):
            raise ValueError("Embedding API returned missing, duplicate, or out-of-order batch indexes.")
        vectors.extend(item.embedding for item in ordered)
        logger.info("Embedded %d/%d stories", min(start + len(batch), len(payloads)), len(payloads))

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(payloads), EMBEDDING_DIMENSIONS) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite embeddings shaped ({len(payloads)}, {EMBEDDING_DIMENSIONS}); got {matrix.shape}.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding API returned a zero-length corpus vector.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _build_fingerprint(source_hash: str) -> str:
    material = json.dumps(
        {
            "store_version": VECTOR_STORE_VERSION,
            "source_hash": source_hash,
            "feature_model": FEATURE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _already_current(destination: Path, fingerprint: str) -> bool:
    if not destination.is_file():
        return False
    try:
        with np.load(destination, allow_pickle=False) as store:
            required = {"embeddings", "records_json", "metadata_json"}
            if not required.issubset(store.files):
                return False
            metadata = json.loads(str(store["metadata_json"].item()))
            embeddings = store["embeddings"]
            records = json.loads(str(store["records_json"].item()))
        if (
            metadata.get("store_version") != VECTOR_STORE_VERSION
            or metadata.get("build_fingerprint") != fingerprint
            or metadata.get("embedding_model") != EMBEDDING_MODEL
            or metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
            or metadata.get("record_count") != len(records)
            or metadata.get("vectors_normalized") is not True
            or not isinstance(records, list)
            or not all(isinstance(record, dict) for record in records)
            or embeddings.shape != (len(records), EMBEDDING_DIMENSIONS)
            or not np.isfinite(embeddings).all()
        ):
            return False
        squared_norms = np.einsum("ij,ij->i", embeddings, embeddings)
        return bool(np.all(np.abs(squared_norms - 1.0) <= 5e-3))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _load_reusable_entries(destination: Path) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    """Reuse unchanged records from a compatible prior store instead of re-calling APIs."""
    if not destination.is_file():
        return {}
    try:
        with np.load(destination, allow_pickle=False) as store:
            if not {"embeddings", "records_json", "metadata_json"}.issubset(store.files):
                return {}
            metadata = json.loads(str(store["metadata_json"].item()))
            if (
                metadata.get("store_version") != VECTOR_STORE_VERSION
                or metadata.get("feature_model") != FEATURE_MODEL
                or metadata.get("embedding_model") != EMBEDDING_MODEL
                or metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
            ):
                return {}
            records = json.loads(str(store["records_json"].item()))
            vectors = store["embeddings"].astype(np.float32, copy=True)
        if (
            not isinstance(records, list)
            or not all(isinstance(record, dict) for record in records)
            or vectors.shape != (len(records), EMBEDDING_DIMENSIONS)
        ):
            return {}
        entries: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for record, vector in zip(records, vectors):
            source_hash = str(record.get("source_record_hash", ""))
            required_features = (
                "rich_descriptive_paragraph",
                "keywords",
                "emotional_scores",
                "search_payload",
            )
            if (
                source_hash
                and all(field in record for field in required_features)
                and isinstance(record["search_payload"], str)
                and record["search_payload"].strip()
                and np.isfinite(vector).all()
                and float(np.linalg.norm(vector)) > 0
            ):
                try:
                    _validate_features(
                        {
                            "rich_descriptive_paragraph": record["rich_descriptive_paragraph"],
                            "keywords": record["keywords"],
                            "emotional_scores": record["emotional_scores"],
                        },
                        str(record.get("title", "Untitled")),
                    )
                except ValueError:
                    continue
                entries[source_hash] = (record, vector / np.linalg.norm(vector))
        return entries
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        logger.warning("Existing vector store could not be reused; rebuilding affected records.")
        return {}


def _write_store(destination: Path, embeddings: np.ndarray, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            embeddings=embeddings,
            records_json=json.dumps(records, ensure_ascii=False, separators=(",", ":")),
            metadata_json=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        )
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _iter_json_array(source_path: Path, *, chunk_bytes: int = JSON_STREAM_CHUNK_BYTES) -> Iterator[Any]:
    """Yield a top-level JSON array without loading a large dataset into RAM."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be at least 1.")

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    expecting_value = True
    saw_value = False
    ended = False
    reached_eof = False

    with source_path.open(encoding="utf-8-sig") as source:
        def refill() -> bool:
            nonlocal buffer, position, reached_eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = source.read(chunk_bytes)
            if not chunk:
                reached_eof = True
                return False
            buffer += chunk
            return True

        while not ended:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    if not started:
                        raise ValueError("Input JSON is empty; expected a top-level array.")
                    raise ValueError("Input JSON ended before its top-level array was closed.")

            if not started:
                if buffer[position] != "[":
                    raise ValueError("Input JSON must contain a top-level array.")
                started = True
                position += 1
                continue

            token = buffer[position]
            if expecting_value:
                if token == "]":
                    if saw_value:
                        raise ValueError("Input JSON contains a trailing comma in its top-level array.")
                    position += 1
                    ended = True
                    continue
                try:
                    value, end_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as exc:
                    if reached_eof or not refill():
                        raise ValueError("Input JSON contains an invalid top-level array item.") from exc
                    continue
                position = end_position
                expecting_value = False
                saw_value = True
                yield value
                continue

            if token == ",":
                position += 1
                expecting_value = True
                continue
            if token == "]":
                position += 1
                ended = True
                continue
            raise ValueError("Input JSON array items must be separated by commas.")

        # Reject a second JSON value or any other non-whitespace suffix.
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError("Input JSON contains data after its top-level array.")
            if not refill():
                return


def _precomputed_embedding_vector(record: dict[str, Any], index: int, embedding_field: str) -> np.ndarray:
    value = record.get(embedding_field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Story #{index} has no non-empty {embedding_field!r} list.")
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Story #{index} has a non-numeric {embedding_field!r} vector.") from exc
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"Story #{index} has an invalid {embedding_field!r} vector.")
    if float(np.linalg.norm(vector)) <= 0:
        raise ValueError(f"Story #{index} has a zero-length {embedding_field!r} vector.")
    return vector


def _inspect_precomputed_dataset(source_path: Path, embedding_field: str) -> tuple[int, int]:
    """Validate IDs and discover the fixed vector width in one streaming pass."""
    record_ids: set[str] = set()
    dimensions: int | None = None
    record_count = 0
    for index, raw_record in enumerate(_iter_json_array(source_path)):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Story #{index} is not a JSON object.")
        vector = _precomputed_embedding_vector(raw_record, index, embedding_field)
        if dimensions is None:
            dimensions = int(vector.size)
        elif vector.size != dimensions:
            raise ValueError(
                f"Story #{index} has {vector.size} embedding dimensions; expected {dimensions}."
            )
        record_id = _record_id(raw_record, index)
        if record_id in record_ids:
            raise ValueError(f"Input JSON contains duplicate record ID {record_id!r}.")
        record_ids.add(record_id)
        record_count += 1
    if not record_count or dimensions is None:
        raise ValueError("Input JSON must contain at least one story with an embedding.")
    return record_count, dimensions


def _precomputed_build_fingerprint(
    source_path: Path,
    *,
    embedding_field: str,
    embedding_model: str,
) -> str:
    """Use file metadata so an unchanged 1+ GB source does not need re-reading."""
    stat = source_path.stat()
    material = json.dumps(
        {
            "store_version": VECTOR_STORE_VERSION,
            "precomputed_import_version": PRECOMPUTED_IMPORT_VERSION,
            "source_file": source_path.name,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "embedding_field": embedding_field,
            "embedding_model": embedding_model,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _precomputed_store_is_current(destination: Path, fingerprint: str) -> bool:
    """Fast-path a completed import without parsing the source JSON again."""
    if not destination.is_file():
        return False
    try:
        with np.load(destination, allow_pickle=False) as store:
            required = {"embeddings", "records_json", "metadata_json"}
            if not required.issubset(store.files):
                return False
            metadata = json.loads(str(store["metadata_json"].item()))
            embeddings = store["embeddings"]
        return bool(
            metadata.get("store_version") == VECTOR_STORE_VERSION
            and metadata.get("precomputed_import_version") == PRECOMPUTED_IMPORT_VERSION
            and metadata.get("build_fingerprint") == fingerprint
            and metadata.get("record_count") == embeddings.shape[0]
            and metadata.get("embedding_dimensions") == embeddings.shape[1]
            and metadata.get("vectors_normalized") is True
            and embeddings.ndim == 2
            and embeddings.shape[0] > 0
            and np.isfinite(embeddings).all()
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
        return False


def import_precomputed_embeddings(
    input_json_path: str,
    output_db_path: str,
    *,
    embedding_field: str = "embedding",
    embedding_model: str = DEFAULT_PRECOMPUTED_EMBEDDING_MODEL,
) -> Path:
    """Convert an embedded JSON dataset into the app's fast local NumPy store.

    This performs no OpenAI API calls. It streams the large source file twice:
    once to validate/count it and once to normalize/write its supplied vectors.
    The stored records omit the original JSON embedding field, avoiding a second
    bulky copy of every vector alongside the compact NumPy matrix.
    """
    if not isinstance(embedding_field, str) or not embedding_field.strip():
        raise ValueError("embedding_field must be a non-empty string.")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string.")
    embedding_field = embedding_field.strip()
    embedding_model = embedding_model.strip()
    source_path = Path(input_json_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source_path}")
    destination = Path(output_db_path)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    if source_path.resolve() == destination.resolve():
        raise ValueError("The output vector store cannot overwrite its JSON source dataset.")

    fingerprint = _precomputed_build_fingerprint(
        source_path,
        embedding_field=embedding_field,
        embedding_model=embedding_model,
    )
    if _precomputed_store_is_current(destination, fingerprint):
        logger.info("Precomputed vector store is current; no import is needed: %s", destination)
        return destination

    record_count, dimensions = _inspect_precomputed_dataset(source_path, embedding_field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".npy", delete=False) as temporary:
        matrix_path = Path(temporary.name)

    embeddings: np.ndarray | None = None
    try:
        embeddings = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(record_count, dimensions),
        )
        records: list[dict[str, Any]] = []
        for index, raw_record in enumerate(_iter_json_array(source_path)):
            if not isinstance(raw_record, dict):
                raise ValueError(f"Story #{index} is not a JSON object.")
            vector = _precomputed_embedding_vector(raw_record, index, embedding_field)
            if vector.size != dimensions:
                raise ValueError(
                    f"Story #{index} has {vector.size} embedding dimensions; expected {dimensions}."
                )
            embeddings[index] = vector / np.linalg.norm(vector)
            # Keep the complete source metadata and summary, but do not duplicate
            # thousands of floats per record inside records_json.
            stored_record = {
                key: value for key, value in raw_record.items() if key != embedding_field
            }
            stored_record["record_id"] = _record_id(stored_record, index)
            records.append(stored_record)
            if (index + 1) % 1_000 == 0 or index + 1 == record_count:
                logger.info("Imported %d/%d precomputed story vectors", index + 1, record_count)

        embeddings.flush()
        metadata = {
            "store_version": VECTOR_STORE_VERSION,
            "build_fingerprint": fingerprint,
            "source_file": source_path.name,
            "source_size": source_path.stat().st_size,
            "source_mtime_ns": source_path.stat().st_mtime_ns,
            "record_count": record_count,
            "embedding_model": embedding_model,
            "embedding_dimensions": dimensions,
            "vectors_normalized": True,
            "precomputed_import_version": PRECOMPUTED_IMPORT_VERSION,
            "precomputed_embedding_field": embedding_field,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_store(destination, embeddings, records, metadata)
        logger.info(
            "Saved %d supplied %s-dimensional story vectors to %s without OpenAI embedding calls",
            record_count,
            dimensions,
            destination,
        )
        return destination
    finally:
        # Windows keeps an open memory-map handle until the Python reference is
        # released, so close it before atomically removing the scratch matrix.
        if embeddings is not None:
            mapped_file = getattr(embeddings, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
            del embeddings
        matrix_path.unlink(missing_ok=True)


def process_and_embed_dataset(
    input_json_path: str,
    output_db_path: str,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> Path:
    """Build the vector store once; skip all API work when input and config are unchanged."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be at least 1.")
    source_path = Path(input_json_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source_path}")
    destination = Path(output_db_path)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")

    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    fingerprint = _build_fingerprint(source_hash)
    if _already_current(destination, fingerprint):
        logger.info("Vector store is current; no feature extraction or embedding work is needed: %s", destination)
        return destination

    records = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Input JSON must contain a non-empty list of story records.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Each input JSON entry must be an object.")
    record_ids = [_record_id(record, index) for index, record in enumerate(records)]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Input JSON contains duplicate record_id or cmu_id values.")

    processed: list[dict[str, Any] | None] = [None] * len(records)
    reused_vectors: dict[int, np.ndarray] = {}
    pending_records: list[tuple[int, dict[str, Any]]] = []
    reusable_entries = _load_reusable_entries(destination)
    feature_cache_path = _feature_cache_path(destination)
    feature_cache = _load_feature_cache(feature_cache_path)
    reused_feature_count = 0
    for index, record in enumerate(records):
        source_record_hash = _source_record_hash(record)
        reusable = reusable_entries.get(source_record_hash)
        if reusable is not None:
            previous, vector = reusable
            features = _validate_features(
                {
                    "rich_descriptive_paragraph": previous["rich_descriptive_paragraph"],
                    "keywords": previous["keywords"],
                    "emotional_scores": previous["emotional_scores"],
                },
                str(record.get("title", "Untitled")),
            )
            processed[index] = _processed_record_from_features(record, index, features)
            reused_vectors[index] = vector
            continue
        cached_features = feature_cache.get(source_record_hash)
        if cached_features is not None:
            processed[index] = _processed_record_from_features(record, index, cached_features)
            reused_feature_count += 1
            continue
        pending_records.append((index, record))

    # Instantiate the SDK only if new/changed records actually need API work.
    if pending_records:
        get_client()
    failures: list[str] = []
    newly_extracted_feature_count = 0
    if pending_records:
        workers = min(max_workers, len(pending_records))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="story-features") as executor:
            futures = {
                executor.submit(_extract_features_with_retry, record, index): index
                for index, record in pending_records
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    extracted = future.result()
                    processed[index] = extracted
                    feature_cache[str(extracted["source_record_hash"])] = _validate_features(
                        {
                            "rich_descriptive_paragraph": extracted["rich_descriptive_paragraph"],
                            "keywords": extracted["keywords"],
                            "emotional_scores": extracted["emotional_scores"],
                        },
                        str(extracted.get("title", "Untitled")),
                    )
                    newly_extracted_feature_count += 1
                except Exception as exc:
                    failures.append(f"#{index} {records[index].get('title', 'Untitled')!r}: {exc}")

    # Write before raising extraction or embedding errors. A later retry can use
    # every successful feature result without repeating its OpenAI feature call.
    if newly_extracted_feature_count:
        _write_feature_cache(feature_cache_path, feature_cache)

    if failures:
        preview = "; ".join(failures[:5])
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)}/{len(records)} stories. "
            f"No partial vector store was written. Examples: {preview}"
        )

    completed = [record for record in processed if record is not None]
    if len(completed) != len(records):
        raise RuntimeError("Preprocessing did not produce one output record per input story.")
    embeddings = np.empty((len(records), EMBEDDING_DIMENSIONS), dtype=np.float32)
    for index, vector in reused_vectors.items():
        embeddings[index] = vector
    new_indices = [index for index in range(len(records)) if index not in reused_vectors]
    if new_indices:
        new_vectors = _embed_payloads(
            [processed[index]["search_payload"] for index in new_indices if processed[index] is not None],
            embedding_batch_size,
        )
        for index, vector in zip(new_indices, new_vectors):
            embeddings[index] = vector
    metadata = {
        "store_version": VECTOR_STORE_VERSION,
        "build_fingerprint": fingerprint,
        "source_sha256": source_hash,
        "source_file": source_path.name,
        "record_count": len(completed),
        "feature_model": FEATURE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "vectors_normalized": True,
        "reused_record_count": len(reused_vectors),
        "reused_feature_cache_record_count": reused_feature_count,
        "newly_extracted_feature_record_count": newly_extracted_feature_count,
        "newly_processed_record_count": len(new_indices),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_store(destination, embeddings, completed, metadata)
    logger.info("Saved %d normalized story vectors to %s", len(completed), destination)
    return destination
