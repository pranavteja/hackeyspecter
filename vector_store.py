"""Resolve local vector stores and materialize Unity Catalog volume files for Apps."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir


@lru_cache(maxsize=4)
def materialize_vector_store(source_path: str) -> Path:
    """Return a local readable NPZ path, downloading UC-volume files for Apps.

    Databricks Apps cannot directly open ``/Volumes/...`` paths even when the
    app is authorized for the volume. The Files API downloads the artifact with
    the App service principal's managed credentials instead.
    """
    source = Path(source_path)
    if source.is_file() or not source_path.startswith("/Volumes/"):
        return source

    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    destination = Path(gettempdir()) / f"hackeyspecter-vector-store-{digest}.npz"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    try:
        from databricks.sdk import WorkspaceClient

        WorkspaceClient().files.download_to(source_path, str(temporary), overwrite=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Databricks Files API returned an empty vector-store download.")
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download vector store from Unity Catalog volume {source_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return destination
