"""
Unified catalog loader for Hackey Specter.

Loads the two CSV datasets (Netflix titles + K-dramas) from a Databricks
Unity Catalog volume when running on Databricks Apps, and falls back to
the local ~/Downloads copies for dev. Each row is normalized into a
single item schema and assigned a locally-derived mood vector (no LLM)
so the mood engine can pre-rank the full ~12k-item catalog cheaply.

Volume path (Databricks): /Volumes/workspace/default/hackey-data
  - netflix_titles.csv
  - kdramas.csv

Local fallback (dev): ~/Downloads/netflix_titles.csv, ~/Downloads/archive/kdramas.csv
"""
from __future__ import annotations

import csv
import os
import re
from functools import lru_cache

from mood_engine.engine import MOOD_AXES

# Databricks Unity Catalog volume (set via env so it's overridable).
VOLUME_DIR = os.environ.get("HACKEY_DATA_VOLUME", "/Volumes/workspace/default/hackey-data")
LOCAL_NETFLIX = os.path.expanduser("~/Downloads/netflix_titles.csv")
LOCAL_KDRAMA = os.path.expanduser("~/Downloads/archive/kdramas.csv")


# ============================================================
# GENRE -> MOOD DELTAS
# ============================================================
# Coarse mapping from a genre/keyword to mood-vector deltas. Applied
# on top of the description-keyword parse so items with thin genres
# (e.g. documentaries) still get a usable signal from their synopsis.
GENRE_MOOD = {
    # tone
    "comedy": {"humor": 0.6, "valence": 0.4, "energy": 0.3},
    "romance": {"romance": 0.7, "warmth": 0.4, "hope": 0.2},
    "romantic": {"romance": 0.7, "warmth": 0.4, "hope": 0.2},
    "drama": {"depth": 0.4, "melancholy": 0.3},
    "tragedy": {"melancholy": 0.7, "valence": -0.5, "hope": -0.4},
    "thriller": {"tension": 0.7, "energy": 0.4, "mystery": 0.3},
    "horror": {"tension": 0.8, "energy": 0.4, "warmth": -0.4, "valence": -0.4},
    "mystery": {"mystery": 0.7, "tension": 0.3, "depth": 0.2},
    "documentary": {"depth": 0.5, "wonder": 0.3, "energy": -0.2},
    "documentaries": {"depth": 0.5, "wonder": 0.3, "energy": -0.2},
    "fantasy": {"wonder": 0.6, "mystery": 0.3, "energy": 0.2},
    "sci-fi": {"wonder": 0.5, "depth": 0.3, "mystery": 0.3},
    "science fiction": {"wonder": 0.5, "depth": 0.3, "mystery": 0.3},
    "adventure": {"energy": 0.5, "wonder": 0.4, "hope": 0.3},
    "action": {"energy": 0.7, "tension": 0.4, "valence": 0.2},
    "crime": {"tension": 0.5, "mystery": 0.4, "depth": 0.3, "warmth": -0.2},
    "family": {"warmth": 0.6, "hope": 0.5, "humor": 0.4, "valence": 0.4},
    "kids": {"warmth": 0.5, "humor": 0.5, "valence": 0.4, "energy": 0.3},
    "children": {"warmth": 0.5, "humor": 0.5, "valence": 0.4, "energy": 0.3},
    "animation": {"wonder": 0.4, "warmth": 0.3, "humor": 0.3},
    "anime": {"wonder": 0.4, "energy": 0.3, "mystery": 0.2},
    "history": {"depth": 0.5, "nostalgia": 0.4, "melancholy": 0.2},
    "historical": {"depth": 0.5, "nostalgia": 0.4, "melancholy": 0.2},
    "war": {"tension": 0.5, "melancholy": 0.5, "depth": 0.4, "valence": -0.3},
    "biography": {"depth": 0.5, "nostalgia": 0.3, "wonder": 0.2},
    "music": {"energy": 0.4, "warmth": 0.3, "nostalgia": 0.3, "humor": 0.2},
    "musical": {"energy": 0.4, "warmth": 0.3, "humor": 0.3, "valence": 0.3},
    "sport": {"energy": 0.5, "hope": 0.4, "valence": 0.3},
    "sports": {"energy": 0.5, "hope": 0.4, "valence": 0.3},
    "reality": {"humor": 0.3, "energy": 0.3, "valence": 0.2},
    "talk": {"humor": 0.3, "warmth": 0.2, "energy": 0.2},
    "western": {"nostalgia": 0.5, "tension": 0.3, "energy": 0.2},
    "korean": {"depth": 0.2, "romance": 0.2, "melancholy": 0.2},  # kdramas lean emotional
    "international": {"depth": 0.2, "wonder": 0.2},
    "independent": {"depth": 0.4, "melancholy": 0.2, "energy": -0.1},
    "classic": {"nostalgia": 0.6, "depth": 0.3},
    "cult": {"wonder": 0.3, "mystery": 0.2, "humor": 0.2},
    "faith": {"hope": 0.5, "warmth": 0.4, "depth": 0.3},
    "lgbtq": {"romance": 0.4, "warmth": 0.3, "depth": 0.3},
    "queer": {"romance": 0.4, "warmth": 0.3, "depth": 0.3},
}

# Description-keyword deltas (reuses the mood engine's keyword map at
# parse time; here we keep a small extra set tuned for plot text).
_DESC_KEYWORD_MOOD = {
    "love": {"romance": 0.5, "warmth": 0.4, "hope": 0.2},
    "heartbreak": {"melancholy": 0.6, "romance": 0.4, "valence": -0.4},
    "grief": {"melancholy": 0.7, "valence": -0.4, "hope": -0.3},
    "death": {"melancholy": 0.5, "tension": 0.2, "valence": -0.3},
    "survive": {"tension": 0.5, "hope": 0.4, "energy": 0.3},
    "mystery": {"mystery": 0.6, "tension": 0.3},
    "murder": {"tension": 0.5, "mystery": 0.4, "warmth": -0.3},
    "friend": {"warmth": 0.4, "hope": 0.2},
    "family": {"warmth": 0.4, "nostalgia": 0.3, "hope": 0.2},
    "war": {"tension": 0.5, "melancholy": 0.4, "depth": 0.3},
    "magic": {"wonder": 0.6, "mystery": 0.3},
    "space": {"wonder": 0.6, "mystery": 0.3},
    "future": {"wonder": 0.4, "depth": 0.3},
    "past": {"nostalgia": 0.5, "melancholy": 0.3},
    "childhood": {"nostalgia": 0.6, "warmth": 0.3, "hope": 0.2},
    "lonely": {"melancholy": 0.5, "warmth": -0.2},
    "hope": {"hope": 0.6, "valence": 0.3},
    "dream": {"wonder": 0.4, "hope": 0.3},
    "revenge": {"tension": 0.5, "energy": 0.3, "warmth": -0.3},
    "secret": {"mystery": 0.5, "tension": 0.2},
    "journey": {"wonder": 0.4, "hope": 0.3, "energy": 0.2},
    "funny": {"humor": 0.6, "valence": 0.3},
    "dark": {"warmth": -0.4, "tension": 0.3, "valence": -0.3},
    "haunting": {"mystery": 0.4, "melancholy": 0.3, "tension": 0.2},
}

_TOKEN_RE = re.compile(r"[a-z][a-z\-]+")
_PAREN_RE = re.compile(r"\(.*?\)")


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    text = _PAREN_RE.sub(" ", text)
    return _TOKEN_RE.findall(text)


def _derive_mood(genres: list[str], description: str) -> dict:
    """Derive a 12-axis mood vector locally from genres + description.

    Starts at neutral 0.5, applies genre deltas (weighted by count) then
    description-keyword deltas, and clamps to [0,1]. No LLM involved.
    """
    vec = {ax: 0.5 for ax in MOOD_AXES}

    # genre deltas
    for g in genres:
        key = g.lower().strip()
        deltas = GENRE_MOOD.get(key)
        if deltas:
            for ax, d in deltas.items():
                vec[ax] = vec.get(ax, 0.5) + d

    # description-keyword deltas (count-weighted, capped)
    tokens = _tokenize(description)
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    for kw, deltas in _DESC_KEYWORD_MOOD.items():
        n = counts.get(kw, 0)
        if n:
            weight = min(n, 3)  # cap repeated influence
            for ax, d in deltas.items():
                vec[ax] = vec.get(ax, 0.5) + d * weight * 0.5

    # clamp
    for ax in MOOD_AXES:
        vec[ax] = max(0.0, min(1.0, vec[ax]))
    return vec


def _themes_from_genres(genres: list[str]) -> list[str]:
    """Light theme extraction: use genres as themes, deduped, lowercased."""
    seen = []
    for g in genres:
        g = g.strip().lower()
        if g and g not in seen:
            seen.append(g)
    return seen[:6]


def _split_genres(raw: str) -> list[str]:
    if not raw:
        return []
    return [g.strip() for g in re.split(r"[,/]", raw) if g.strip()]


def _parse_year(raw: str) -> int | None:
    if not raw or raw == "\\N":
        return None
    m = re.search(r"(\d{4})", str(raw))
    return int(m.group(1)) if m else None


def _parse_runtime(raw: str) -> int | None:
    """Parse '90 min', '2 Seasons', or a bare number of minutes."""
    if not raw or raw == "\\N":
        return None
    m = re.search(r"(\d+)", str(raw))
    if not m:
        return None
    n = int(m.group(1))
    low = str(raw).lower()
    if "season" in low:
        return n  # seasons count kept as-is; downstream treats as runtime-ish
    return n


def _load_netflix(path: str) -> list[dict]:
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            genres = _split_genres(row.get("listed_in", ""))
            desc = (row.get("description") or "").strip()
            kind = (row.get("type") or "").strip().lower()
            media_type = "movie" if kind == "movie" else "tv"
            year = _parse_year(str(row.get("release_year") or ""))
            runtime = _parse_runtime(str(row.get("duration") or ""))
            director = (row.get("director") or "").strip()
            items.append({
                "id": "nf_" + (row.get("show_id") or "").strip(),
                "title": title,
                "creator": director or "Unknown",
                "year": year or 0,
                "type": media_type,
                "runtime": runtime,
                "genres": genres,
                "themes": _themes_from_genres(genres),
                "description": desc,
                "pitch": desc,  # use the synopsis as the pitch
                "emotional_arc": "",  # not available; LLM fills the "why"
                "mood": _derive_mood(genres, desc),
                "source": "netflix",
            })
    return items


def _load_kdrama(path: str) -> list[dict]:
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            genres = _split_genres(row.get("genres", ""))
            desc = (row.get("synopsis") or "").strip()
            year = _parse_year(str(row.get("startYear") or ""))
            runtime = _parse_runtime(str(row.get("runtimeMinutes") or ""))
            lead = (row.get("mainLead1") or "").strip()
            items.append({
                "id": "kd_" + (row.get("id") or "").strip(),
                "title": title,
                "creator": lead or "Unknown",
                "year": year or 0,
                "type": "tv",
                "runtime": runtime,
                "genres": genres,
                "themes": _themes_from_genres(genres),
                "description": desc,
                "pitch": desc,
                "emotional_arc": "",
                "mood": _derive_mood(genres, desc),
                "source": "kdrama",
            })
    return items


def _resolve_paths() -> tuple[str, str]:
    """Return (netflix_path, kdrama_path), preferring the Unity volume."""
    nf_vol = os.path.join(VOLUME_DIR, "netflix_titles.csv")
    kd_vol = os.path.join(VOLUME_DIR, "kdramas.csv")
    if os.path.exists(nf_vol) and os.path.exists(kd_vol):
        return nf_vol, kd_vol
    # local dev fallback
    return LOCAL_NETFLIX, LOCAL_KDRAMA


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """Load and cache the unified catalog (both datasets)."""
    nf_path, kd_path = _resolve_paths()
    items: list[dict] = []
    if os.path.exists(nf_path):
        items.extend(_load_netflix(nf_path))
    if os.path.exists(kd_path):
        items.extend(_load_kdrama(kd_path))
    return items


def catalog_summary() -> dict:
    """Quick stats for UI/debugging."""
    items = load_catalog()
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    return {
        "total": len(items),
        "by_type": by_type,
        "by_source": by_source,
        "volume_dir": VOLUME_DIR,
        "netflix_path": LOCAL_NETFLIX,
        "kdrama_path": LOCAL_KDRAMA,
    }