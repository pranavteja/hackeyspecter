"""Presentation-only cleanup for imported story records."""
from __future__ import annotations

import re


_METADATA_SUFFIX = re.compile(
    r"\n\s*(?:---\s*)?metadata(?:\s*---)?\s*:?[\s\S]*$",
    flags=re.IGNORECASE,
)
_STRAY_DIV_TAG = re.compile(r"</?div\b[^>]*>", flags=re.IGNORECASE)


def story_summary_preview(value: object, *, maximum_characters: int = 700) -> str:
    """Return reader-facing summary text without source metadata or stray div tags."""
    if maximum_characters < 1:
        raise ValueError("maximum_characters must be positive.")

    summary = _METADATA_SUFFIX.sub("", str(value or ""))
    summary = _STRAY_DIV_TAG.sub("", summary)
    summary = " ".join(summary.split())
    if len(summary) > maximum_characters:
        return summary[:maximum_characters].rstrip() + "…"
    return summary
