"""
Unified catalog loader for Hackey Specter.

Loads the audiobook summaries JSON from a Databricks Unity Catalog volume
when running on Databricks Apps, and falls back to the local ~/Downloads
copy for dev. Each row is normalized into a single item schema and
assigned a locally-derived mood vector (no LLM) so the mood engine can
pre-rank the catalog cheaply.

Volume path (Databricks): /Volumes/workspace/default/hackey-data
  - summary_1to16000.json

Local fallback (dev): ~/Downloads/summary_1to16000.json
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from mood_engine.engine import MOOD_AXES

# Databricks Unity Catalog volume (set via env so it's overridable).
VOLUME_DIR = os.environ.get("HACKEY_DATA_VOLUME", "/Volumes/workspace/default/hackey-data")
LOCAL_SUMMARY = os.path.expanduser("~/Downloads/summary_1to16000.json")
SUMMARY_FILENAME = "summary_1to16000.json"


# ============================================================
# MOOD KEYWORD DELTAS (for deriving mood from plot summaries)
# ============================================================
# Merged from the mood engine's MOOD_KEYWORDS + plot-specific terms.
# Applied to the summary text to produce a 12-axis mood vector locally.
_KEYWORD_MOOD = {
    # tone / atmosphere
    "rainy": {"melancholy": 0.5, "warmth": 0.4, "nostalgia": 0.4, "energy": -0.5},
    "sunday": {"nostalgia": 0.4, "warmth": 0.3, "melancholy": 0.2, "energy": -0.3},
    "heartbreak": {"melancholy": 0.7, "warmth": 0.3, "romance": 0.5, "valence": -0.4, "hope": -0.3},
    "breakup": {"melancholy": 0.6, "romance": 0.4, "warmth": 0.2, "valence": -0.3, "hope": -0.3},
    "lonely": {"melancholy": 0.5, "warmth": 0.3, "romance": 0.2, "energy": -0.3},
    "loneliness": {"melancholy": 0.5, "warmth": -0.2, "energy": -0.3},
    "crying": {"melancholy": 0.6, "energy": -0.5, "hope": -0.2},
    "cozy": {"warmth": 0.6, "energy": -0.3, "humor": 0.2, "nostalgia": 0.3},
    "comfort": {"warmth": 0.6, "nostalgia": 0.4, "hope": 0.3, "energy": -0.3},
    "quiet": {"energy": -0.6, "tension": -0.4, "warmth": 0.3},
    "slow": {"energy": -0.5, "depth": 0.2},
    # upbeat
    "happy": {"valence": 0.6, "hope": 0.4, "humor": 0.3, "energy": 0.3},
    "joyful": {"valence": 0.7, "hope": 0.5, "energy": 0.4},
    "joy": {"valence": 0.6, "hope": 0.4, "energy": 0.3},
    "fun": {"humor": 0.6, "energy": 0.4, "valence": 0.5},
    "funny": {"humor": 0.7, "valence": 0.4},
    "humor": {"humor": 0.6, "valence": 0.3},
    "comedy": {"humor": 0.6, "valence": 0.4, "energy": 0.3},
    "uplifting": {"hope": 0.6, "valence": 0.5, "energy": 0.3},
    "inspiring": {"hope": 0.7, "wonder": 0.4, "energy": 0.3},
    "hopeful": {"hope": 0.7, "valence": 0.3, "melancholy": -0.3},
    "hope": {"hope": 0.6, "valence": 0.3},
    # dark / heavy
    "dark": {"warmth": -0.5, "valence": -0.4, "tension": 0.4, "depth": 0.3},
    "brooding": {"tension": 0.4, "melancholy": 0.4, "warmth": -0.3, "energy": -0.2},
    "bleak": {"hope": -0.5, "valence": -0.4, "warmth": -0.3, "melancholy": 0.3},
    "intense": {"energy": 0.6, "tension": 0.5, "depth": 0.2},
    "scary": {"tension": 0.7, "energy": 0.3, "warmth": -0.3},
    "spooky": {"mystery": 0.5, "tension": 0.5, "warmth": -0.2},
    "horror": {"tension": 0.8, "energy": 0.4, "warmth": -0.4, "valence": -0.4},
    "thrilling": {"tension": 0.7, "energy": 0.5},
    "suspenseful": {"tension": 0.7, "mystery": 0.4},
    "suspense": {"tension": 0.6, "mystery": 0.3},
    "mysterious": {"mystery": 0.7, "depth": 0.3, "tension": 0.2},
    "mystery": {"mystery": 0.6, "tension": 0.3},
    # high energy
    "exciting": {"energy": 0.6, "tension": 0.3, "valence": 0.3},
    "adventurous": {"energy": 0.5, "wonder": 0.5, "hope": 0.3},
    "adventure": {"energy": 0.5, "wonder": 0.4, "hope": 0.3},
    "epic": {"energy": 0.7, "wonder": 0.6, "depth": 0.3},
    # romance
    "romantic": {"romance": 0.8, "warmth": 0.5, "hope": 0.3},
    "romance": {"romance": 0.7, "warmth": 0.4, "hope": 0.2},
    "love": {"romance": 0.6, "warmth": 0.5, "hope": 0.3},
    "loved": {"romance": 0.5, "warmth": 0.4, "hope": 0.2},
    "yearning": {"romance": 0.6, "melancholy": 0.4, "nostalgia": 0.3},
    "longing": {"romance": 0.5, "melancholy": 0.5, "nostalgia": 0.4},
    "passion": {"romance": 0.6, "energy": 0.3, "warmth": 0.3},
    # thoughtful
    "deep": {"depth": 0.7, "energy": -0.2},
    "philosophical": {"depth": 0.8, "wonder": 0.4},
    "philosophy": {"depth": 0.7, "wonder": 0.3},
    "thoughtful": {"depth": 0.6, "energy": -0.2, "warmth": 0.2},
    "meaningful": {"depth": 0.6, "hope": 0.3, "wonder": 0.3},
    "reflective": {"depth": 0.6, "melancholy": 0.3, "nostalgia": 0.3},
    "introspective": {"depth": 0.7, "melancholy": 0.3, "energy": -0.3},
    # nostalgia
    "nostalgic": {"nostalgia": 0.7, "warmth": 0.3, "melancholy": 0.3},
    "nostalgia": {"nostalgia": 0.6, "warmth": 0.3, "melancholy": 0.2},
    "childhood": {"nostalgia": 0.8, "warmth": 0.4, "hope": 0.3},
    "memory": {"nostalgia": 0.7, "depth": 0.4, "melancholy": 0.3},
    "remember": {"nostalgia": 0.6, "melancholy": 0.3},
    "past": {"nostalgia": 0.5, "melancholy": 0.3},
    # awe / wonder
    "wonder": {"wonder": 0.7, "humor": -0.2, "hope": 0.3},
    "magical": {"wonder": 0.7, "warmth": 0.3},
    "magic": {"wonder": 0.6, "mystery": 0.3},
    "awe": {"wonder": 0.8, "depth": 0.3},
    "space": {"wonder": 0.7, "mystery": 0.4},
    "ocean": {"wonder": 0.6, "warmth": 0.3, "energy": -0.2},
    "nature": {"wonder": 0.5, "warmth": 0.4, "tension": -0.3},
    # plot-specific
    "grief": {"melancholy": 0.7, "valence": -0.4, "hope": -0.3},
    "death": {"melancholy": 0.5, "tension": 0.2, "valence": -0.3},
    "die": {"melancholy": 0.4, "valence": -0.2},
    "died": {"melancholy": 0.5, "valence": -0.3},
    "survive": {"tension": 0.5, "hope": 0.4, "energy": 0.3},
    "survival": {"tension": 0.5, "hope": 0.4, "energy": 0.3},
    "murder": {"tension": 0.5, "mystery": 0.4, "warmth": -0.3},
    "killed": {"tension": 0.4, "mystery": 0.3, "valence": -0.2},
    "war": {"tension": 0.5, "melancholy": 0.4, "depth": 0.3},
    "battle": {"tension": 0.4, "energy": 0.3, "depth": 0.2},
    "friend": {"warmth": 0.4, "hope": 0.2},
    "friendship": {"warmth": 0.5, "hope": 0.3},
    "family": {"warmth": 0.4, "nostalgia": 0.3, "hope": 0.2},
    "future": {"wonder": 0.4, "depth": 0.3},
    "dream": {"wonder": 0.4, "hope": 0.3},
    "dreams": {"wonder": 0.4, "hope": 0.3},
    "revenge": {"tension": 0.5, "energy": 0.3, "warmth": -0.3},
    "secret": {"mystery": 0.5, "tension": 0.2},
    "secrets": {"mystery": 0.5, "tension": 0.2},
    "journey": {"wonder": 0.4, "hope": 0.3, "energy": 0.2},
    "quest": {"energy": 0.4, "wonder": 0.4, "hope": 0.3},
    "haunting": {"mystery": 0.4, "melancholy": 0.3, "tension": 0.2},
    "ghost": {"mystery": 0.4, "tension": 0.3, "melancholy": 0.2},
    "tragedy": {"melancholy": 0.7, "valence": -0.5, "hope": -0.4},
    "tragic": {"melancholy": 0.6, "valence": -0.4, "hope": -0.3},
    "loss": {"melancholy": 0.6, "valence": -0.3, "hope": -0.2},
    "lost": {"melancholy": 0.4, "nostalgia": 0.3, "valence": -0.2},
    "betrayal": {"tension": 0.4, "melancholy": 0.3, "warmth": -0.3},
    "betray": {"tension": 0.4, "warmth": -0.3},
    "faith": {"hope": 0.5, "warmth": 0.4, "depth": 0.3},
    "redemption": {"hope": 0.6, "depth": 0.4, "valence": 0.3},
    "forgive": {"warmth": 0.4, "hope": 0.4},
    "forgiveness": {"warmth": 0.5, "hope": 0.5},
    "courage": {"hope": 0.5, "energy": 0.3, "depth": 0.3},
    "brave": {"hope": 0.4, "energy": 0.3},
    "fear": {"tension": 0.5, "energy": 0.2, "warmth": -0.2},
    "afraid": {"tension": 0.4, "warmth": -0.2},
    "danger": {"tension": 0.5, "energy": 0.3},
    "escape": {"tension": 0.4, "energy": 0.3, "hope": 0.3},
    "freedom": {"hope": 0.5, "energy": 0.3, "valence": 0.3},
    "justice": {"depth": 0.4, "hope": 0.3, "tension": 0.2},
    "power": {"energy": 0.4, "tension": 0.3, "depth": 0.3},
    "kingdom": {"wonder": 0.3, "energy": 0.2, "depth": 0.2},
    "king": {"depth": 0.3, "nostalgia": 0.2, "energy": 0.2},
    "queen": {"depth": 0.3, "nostalgia": 0.2, "energy": 0.2},
    "god": {"wonder": 0.4, "depth": 0.4},
    "gods": {"wonder": 0.4, "depth": 0.4, "mystery": 0.2},
    "spiritual": {"wonder": 0.4, "depth": 0.4, "hope": 0.3},
    "sacred": {"wonder": 0.3, "depth": 0.3, "warmth": 0.2},
}

_TOKEN_RE = re.compile(r"[a-z][a-z\-]+")


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)


def _derive_mood(summary: str) -> dict:
    """Derive a 12-axis mood vector locally from the summary text.

    Starts at neutral 0.5, applies keyword deltas (count-weighted, capped),
    and clamps to [0,1]. No LLM involved.
    """
    vec = {ax: 0.5 for ax in MOOD_AXES}
    tokens = _tokenize(summary)
    if not tokens:
        return vec

    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    for kw, deltas in _KEYWORD_MOOD.items():
        n = counts.get(kw, 0)
        if n:
            weight = min(n, 4)  # cap repeated influence
            for ax, d in deltas.items():
                vec[ax] = vec.get(ax, 0.5) + d * weight * 0.4

    # clamp
    for ax in MOOD_AXES:
        vec[ax] = max(0.0, min(1.0, vec[ax]))
    return vec


def _themes_from_summary(summary: str) -> list[str]:
    """Extract a few theme keywords from the summary (top mood keywords found)."""
    tokens = _tokenize(summary)
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    # pick keywords that appear and are in our mood map
    found = [(kw, counts[kw]) for kw in _KEYWORD_MOOD if counts.get(kw, 0) > 0]
    found.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in found[:6]]


def _resolve_summary_path() -> str:
    """Return the JSON path, preferring the Unity volume.

    Falls back to scanning the volume directory for any .json file if
    the exact filename isn't found, so it works even if the file was
    uploaded with a slightly different name.
    """
    # 1. exact name on the volume
    vol_path = os.path.join(VOLUME_DIR, SUMMARY_FILENAME)
    if os.path.exists(vol_path):
        return vol_path
    # 2. any .json in the volume directory (case-insensitive)
    if os.path.isdir(VOLUME_DIR):
        try:
            for entry in sorted(os.listdir(VOLUME_DIR)):
                if entry.lower().endswith(".json"):
                    return os.path.join(VOLUME_DIR, entry)
        except Exception:
            pass
    # 3. local dev fallback
    if os.path.exists(LOCAL_SUMMARY):
        return LOCAL_SUMMARY
    return LOCAL_SUMMARY  # returned even if missing — load_catalog reports it


# collect load errors for the diagnostic panel
_load_errors: list[str] = []


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """Load and cache the audiobook catalog from the JSON file."""
    _load_errors.clear()
    path = _resolve_summary_path()
    items: list[dict] = []

    if not os.path.exists(path):
        _load_errors.append(f"file not found at {path}")
        return items

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _load_errors.append(f"JSON parse error ({path}): {type(e).__name__}: {e}")
        return items

    if not isinstance(data, list):
        _load_errors.append(f"expected a JSON list, got {type(data).__name__}")
        return items

    for i, row in enumerate(data):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        summary = (row.get("summary") or "").strip()
        items.append({
            "id": f"ab_{i}",
            "title": title,
            "creator": "LibriVox",
            "year": 0,
            "type": "audiobook",
            "runtime": None,
            "genres": [],
            "themes": _themes_from_summary(summary),
            "description": summary,
            "pitch": (summary[:200] + "…") if len(summary) > 200 else summary,
            "emotional_arc": "",
            "mood": _derive_mood(summary),
            "source": "librivox",
            "audio_zip_url": row.get("audio_zip_url", ""),
            "librivox_project_url": row.get("librivox_project_url", ""),
        })

    return items


def _list_dir(path: str) -> str:
    """Best-effort directory listing for diagnostics; '' if unavailable."""
    try:
        if os.path.isdir(path):
            entries = os.listdir(path)
            return ", ".join(sorted(entries)[:20]) or "(empty dir)"
        if os.path.exists(path):
            return f"(exists but is not a dir: type={_file_kind(path)})"
        return f"(does not exist: {path!r})"
    except Exception as e:
        return f"(listing failed: {type(e).__name__}: {e})"


def _file_kind(path: str) -> str:
    try:
        import stat
        st = os.stat(path)
        if stat.S_ISDIR(st.st_mode):
            return "dir"
        if stat.S_ISREG(st.st_mode):
            return "file"
        if stat.S_ISLNK(st.st_mode):
            return "symlink"
    except Exception:
        pass
    return "unknown"


def _volumes_root_probe() -> str:
    """List the top of /Volumes so we can see what's actually attached."""
    root = "/Volumes"
    try:
        if not os.path.exists(root):
            return f"({root} does not exist — volume not attached)"
        if not os.path.isdir(root):
            return f"({root} exists but is not a dir)"
        entries = os.listdir(root)
        if not entries:
            return f"({root} exists but is empty)"
        # try a level deeper too
        lines = [f"{root}: " + ", ".join(sorted(entries))]
        for e in sorted(entries)[:5]:
            sub = os.path.join(root, e)
            if os.path.isdir(sub):
                try:
                    sub_entries = os.listdir(sub)
                    lines.append(f"  {sub}/: " + ", ".join(sorted(sub_entries)[:10]))
                except Exception:
                    pass
        return "\n".join(lines)
    except Exception as e:
        return f"(probe failed: {type(e).__name__}: {e})"


def catalog_summary() -> dict:
    """Quick stats for UI/debugging, including resolved path + errors."""
    items = load_catalog()
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    path = _resolve_summary_path()
    return {
        "total": len(items),
        "by_type": by_type,
        "by_source": by_source,
        "volume_dir": VOLUME_DIR,
        "volume_listing": _list_dir(VOLUME_DIR),
        "volumes_root": _volumes_root_probe(),
        "summary_path": path,
        "summary_exists": os.path.exists(path),
        "load_errors": list(_load_errors),
    }