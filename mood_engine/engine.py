"""
Local mood engine for Hackey Specter.

Replaces an LLM with a deterministic, fast, rule-based pipeline:
  1. parse free text -> target mood vector
  2. cosine-similarity rank content against the vector
  3. generate "why you'll love this" by reading the mood vector and themes
  4. plan a weekend by sequencing across media types

The architecture is LLM-swappable: each function takes a string or
mood vector and returns a dict. Replacing the parser with a real
embedding or LLM call is a single drop-in change.
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter
from datetime import date, timedelta
from typing import Iterable

from data.content import CONTENT, MOOD_KEYWORDS, SAMPLE_PROMPTS


MOOD_AXES = [
    "valence", "energy", "warmth", "tension", "depth", "romance",
    "mystery", "nostalgia", "hope", "melancholy", "wonder", "humor",
]


# ============================================================
# 1. MOOD PARSER
# ============================================================

def _tokenize(text: str) -> list[str]:
    """Light tokenizer: lowercase, strip punctuation, keep 2+ char tokens."""
    text = text.lower()
    # keep hyphens inside words (cozy-game, post-civil-war)
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = re.findall(r"[a-z][a-z\-]+", text)
    return tokens


def parse_mood(text: str) -> dict:
    """
    Turn free text into a mood vector.

    Algorithm:
      - start from a neutral 0.5 vector
      - for each recognized mood keyword, apply the delta
      - cap values to [0, 1]
      - if nothing matched, return neutral
    """
    tokens = _tokenize(text)
    if not tokens:
        return {axis: 0.5 for axis in MOOD_AXES}

    # also try bigrams so we catch "rainy sunday", "post civil war", etc.
    bigrams = [" ".join(p) for p in zip(tokens, tokens[1:])]
    haystack = " ".join(tokens + bigrams)

    # accumulate deltas
    delta: dict[str, float] = {}
    matched_keywords: list[str] = []

    # prefer multi-word keys first
    multiword_keys = sorted(
        [k for k in MOOD_KEYWORDS if " " in k or "-" in k],
        key=len, reverse=True,
    )
    singleword_keys = [k for k in MOOD_KEYWORDS if " " not in k and "-" not in k]

    # 1. multiword / hyphenated phrase matches
    for key in multiword_keys:
        # normalize hyphen keys to match space-separated text
        needle = key.replace("-", " ")
        if needle in haystack:
            for axis, change in MOOD_KEYWORDS[key].items():
                delta[axis] = delta.get(axis, 0.0) + change
            matched_keywords.append(key)
            # remove from haystack so we don't double-match
            haystack = haystack.replace(needle, " ")

    # 2. single-word matches (count occurrences)
    word_counts = Counter(tokens)
    for key in singleword_keys:
        if word_counts.get(key, 0) > 0:
            for axis, change in MOOD_KEYWORDS[key].items():
                delta[axis] = delta.get(axis, 0.0) + change * word_counts[key]
            matched_keywords.append(key)

    if not matched_keywords:
        return {axis: 0.5 for axis in MOOD_AXES}

    # apply deltas to neutral baseline
    vector = {axis: 0.5 for axis in MOOD_AXES}
    for axis, change in delta.items():
        vector[axis] = max(0.0, min(1.0, 0.5 + change))

    return vector


# ============================================================
# 2. SIMILARITY MATCHER
# ============================================================

def _cosine(a: dict, b: dict) -> float:
    """Cosine similarity over the shared mood vector axes."""
    axes = MOOD_AXES
    dot = sum(a[ax] * b[ax] for ax in axes)
    na = math.sqrt(sum(a[ax] ** 2 for ax in axes))
    nb = math.sqrt(sum(b[ax] ** 2 for ax in axes))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _theme_overlap(query_themes: list[str], item_themes: list[str]) -> float:
    """Jaccard-ish overlap of themes, weighted toward recall."""
    if not query_themes or not item_themes:
        return 0.0
    qs = {t.lower() for t in query_themes}
    iset = {t.lower() for t in item_themes}
    inter = qs & iset
    if not inter:
        return 0.0
    return len(inter) / math.sqrt(len(qs) * len(iset))


def rank(
    target: dict,
    pool: list[dict] | None = None,
    k: int = 5,
    types: Iterable[str] | None = None,
    exclude_ids: Iterable[str] | None = None,
    query_themes: list[str] | None = None,
) -> list[tuple[dict, float]]:
    """
    Return top-k items from `pool` ranked by mood similarity.
    Optionally filter by media type and exclude already-seen ids.
    """
    pool = pool if pool is not None else CONTENT
    exclude_ids = set(exclude_ids or [])
    types_set = set(types) if types else None

    scored: list[tuple[dict, float]] = []
    for item in pool:
        if item["id"] in exclude_ids:
            continue
        if types_set and item["type"] not in types_set:
            continue
        mood_score = _cosine(target, item["mood"])
        theme_bonus = _theme_overlap(query_themes or [], item.get("themes", [])) * 0.15
        scored.append((item, mood_score + theme_bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ============================================================
# 3. EXPLAIN-WHY GENERATOR
# ============================================================

# which axes we surface in "why you'll love this"
EMOTIONAL_AXES = [
    "valence", "energy", "warmth", "tension", "depth", "romance",
    "mystery", "nostalgia", "hope", "melancholy", "wonder", "humor",
]

AXIS_PHRASE = {
    "valence":    {"high": "leaning toward the warm side of the emotional spectrum",
                   "low":  "sitting in the hard, honest end of feeling"},
    "energy":     {"high": "buzzy and propulsive",
                   "low":  "slow-burn and unhurried"},
    "warmth":     {"high": "deeply human and affectionate",
                   "low":  "cool, distant, a little austere"},
    "tension":    {"high": "taut with suspense",
                   "low":  "calm and free of pressure"},
    "depth":      {"high": "asks something real about being alive",
                   "low":  "easy, surface-level fun"},
    "romance":    {"high": "threads love right through the center",
                   "low":  "not really about love"},
    "mystery":    {"high": "keeps you guessing",
                   "low":  "doesn't really hide things"},
    "nostalgia":  {"high": "smells like an old year",
                   "low":  "firmly anchored in the present"},
    "hope":       {"high": "ends on the side of light",
                   "low":  "unflinching about how bleak it can get"},
    "melancholy": {"high": "carries a real ache",
                   "low":  "doesn't lean on sadness"},
    "wonder":     {"high": "strange and awe-filled",
                   "low":  "rooted in the everyday"},
    "humor":      {"high": "funny in a way that lands",
                   "low":  "mostly plays it straight"},
}


def _classify(axis: str, value: float) -> str:
    if value >= 0.66:
        return "high"
    if value <= 0.34:
        return "low"
    return "mid"


def explain(item: dict, target: dict) -> str:
    """
    Generate a 'why you will love this' paragraph for an item,
    given the user's target mood vector.

    Approach: identify the 2-3 axes where item & target are most
    aligned and weave them together with the item's themes and
    emotional arc. Falls back to the item's pitch.
    """
    # pad target with neutral values for any missing axes
    full_target = {ax: target.get(ax, 0.5) for ax in MOOD_AXES}

    # axes where the item and target agree most strongly
    agreements = []
    for ax in MOOD_AXES:
        item_v = item["mood"][ax]
        tgt_v = full_target[ax]
        # agreement = how close the two are (in [0,1])
        agreement = 1.0 - abs(item_v - tgt_v)
        # weight by extremity: if both are extreme, the alignment
        # matters more than if both are middling
        extremity = abs(item_v - 0.5) * 2
        agreements.append((ax, agreement, extremity, item_v))
    # sort by agreement * (1 + extremity)
    agreements.sort(key=lambda x: x[1] * (1 + x[2]), reverse=True)
    top_axes = agreements[:3]

    parts = []
    parts.append(f"**{item['title']}** by {item['creator']} ({item['year']}).")
    parts.append(item.get("pitch", ""))

    # describe aligned feelings
    descriptors = []
    for ax, _, extremity, value in top_axes:
        cls = _classify(ax, value)
        if cls == "mid":
            continue
        phrase = AXIS_PHRASE[ax][cls]
        descriptors.append(phrase)

    if descriptors:
        joined = ", ".join(descriptors[:-1])
        if len(descriptors) > 1:
            joined = joined + " and " + descriptors[-1]
        else:
            joined = descriptors[0]
        parts.append(f"It is {joined} — which lines up with what you said you wanted.")

    themes = item.get("themes", [])
    if themes:
        sample = themes[:3]
        parts.append(
            "It touches on " + ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0]) + "."
        )

    arc = item.get("emotional_arc")
    if arc:
        parts.append(arc + ".")

    return " ".join(parts)


# ============================================================
# 3b. LLM-POWERED EXPLAINER (OpenAI, optional)
# ============================================================
# Falls back to the rule-based `explain()` when no API key is set,
# so the app keeps working in local dev or if the secret is absent.

_OPENAI_CLIENT = None  # lazily initialized, cached across calls


def _get_openai_client():
    """Return an OpenAI client if OPENAI_API_KEY is set, else None."""
    global _OPENAI_CLIENT
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    if _OPENAI_CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError:
            return None
        _OPENAI_CLIENT = OpenAI(api_key=key)
    return _OPENAI_CLIENT


def _mood_to_phrase(axis: str, value: float) -> str:
    """Human-readable label for one mood axis value, for the prompt."""
    cls = _classify(axis, value)
    if cls == "mid":
        return f"neutral {axis}"
    return AXIS_PHRASE[axis][cls]


def explain_llm(item: dict, target: dict) -> str:
    """
    Generate the 'why you will love this' paragraph with OpenAI when a
    key is available; otherwise fall back to the deterministic `explain`.

    The LLM is given the item's metadata and the user's target mood
    vector and asked for a single short, evocative paragraph in the
    cozy-literary voice of the app. Output is plain text (no markdown).
    """
    client = _get_openai_client()
    if client is None:
        return explain(item, target)

    # Build a compact description of the user's target mood: only the
    # axes that meaningfully deviate from neutral.
    full_target = {ax: target.get(ax, 0.5) for ax in MOOD_AXES}
    mood_bits = [
        f"{ax}: {_mood_to_phrase(ax, v)}"
        for ax, v in full_target.items()
        if abs(v - 0.5) > 0.15
    ]
    target_mood_str = ", ".join(mood_bits) if mood_bits else "a quiet, open feeling"

    themes = ", ".join(item.get("themes", [])) or "none"
    pitch = item.get("pitch", "")
    arc = item.get("emotional_arc", "")

    system = (
        "You are the voice of Hackey Specter, a mood-first entertainment "
        "recommendation app. You write one short, evocative paragraph (2-4 "
        "sentences, ~50 words) explaining why a specific piece of media is "
        "the right match for how the user wants to feel. Warm, literary, "
        "specific, never generic. No marketing fluff, no exclamation marks, "
        "no markdown, no lists. Speak to the reader in second person."
    )
    user = (
        f"The user wants to feel: {target_mood_str}.\n"
        f"Recommendation: {item['title']} ({item['type']}, {item['year']}) "
        f"by {item['creator']}.\n"
        f"Pitch: {pitch}\n"
        f"Emotional arc: {arc}\n"
        f"Themes: {themes}\n"
        f"Write the 'Why you will love this' paragraph."
    )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text.strip() if hasattr(resp, "output_text") else ""
        if not text:
            # older response shapes
            text = (resp.choices[0].message.content or "").strip()
        return text or explain(item, target)
    except Exception:
        # Never let an LLM hiccup break the UI — fall back gracefully.
        return explain(item, target)


# ============================================================
# 4. WEEKEND CONCIERGE
# ============================================================

WEEKEND_TEMPLATE = [
    # (slot, media_types, target_axes_with_weight)
    ("Friday night — settle in",          ["movie"],        {"warmth": 0.7, "energy": 0.4, "mystery": 0.3}),
    ("Saturday morning — slowly",         ["book", "podcast"], {"warmth": 0.7, "nostalgia": 0.6, "energy": 0.3}),
    ("Saturday afternoon — go deeper",    ["book", "game"], {"depth": 0.7, "wonder": 0.6, "energy": 0.5}),
    ("Saturday evening — main event",     ["movie"],        {"valence": 0.6, "depth": 0.7, "wonder": 0.5}),
    ("Sunday morning — quiet",            ["podcast", "book"], {"nostalgia": 0.6, "warmth": 0.7, "energy": 0.3}),
    ("Sunday afternoon — playful",        ["game"],         {"humor": 0.7, "hope": 0.6, "warmth": 0.7}),
    ("Sunday night — landing",            ["podcast", "movie"], {"nostalgia": 0.6, "melancholy": 0.4, "warmth": 0.7}),
]


def _blend(base: dict, slot_target: dict, weight: float = 0.7) -> dict:
    """Mix the user's mood with the slot's preferred tone."""
    out = {}
    for ax in MOOD_AXES:
        out[ax] = base[ax] * (1 - weight) + slot_target.get(ax, 0.5) * weight
        out[ax] = max(0.0, min(1.0, out[ax]))
    return out


def plan_weekend(mood: dict, used_ids: set[str] | None = None) -> list[dict]:
    """
    Build a 7-slot weekend itinerary. Each slot pulls the best
    matching item from the requested media types, biased toward
    the slot's tonal target. Ensures at least one of each medium
    appears across the weekend.
    """
    used_ids = set(used_ids or [])
    itinerary = []
    types_used: set[str] = set()

    for slot, types, slot_target in WEEKEND_TEMPLATE:
        target = _blend(mood, slot_target, weight=0.55)

        # Prefer a type we haven't used yet if one is available
        # and the slot allows it. This guarantees media-type variety.
        preferred_types = list(types)
        unused_allowed = [t for t in types if t not in types_used]
        if unused_allowed:
            # bias: try unused types first
            preferred_types = unused_allowed + [t for t in types if t in types_used]

        chosen_item = None
        chosen_score = 0.0
        chosen_type = None
        for media_type in preferred_types:
            results = rank(
                target=target,
                k=2,
                types=[media_type],
                exclude_ids=used_ids,
            )
            if not results:
                continue
            for item, s in results:
                if item["id"] in used_ids:
                    continue
                chosen_item = item
                chosen_score = s
                chosen_type = media_type
                break
            if chosen_item is not None:
                break

        # fallback: any matching item in the slot's allowed types
        if chosen_item is None:
            results = rank(
                target=target,
                k=3,
                types=types,
                exclude_ids=used_ids,
            )
            if results:
                chosen_item, chosen_score = results[0]
                chosen_type = chosen_item["type"]

        if chosen_item is None:
            continue
        used_ids.add(chosen_item["id"])
        types_used.add(chosen_type or chosen_item["type"])
        itinerary.append({
            "slot": slot,
            "types": types,
            "item": chosen_item,
            "score": round(chosen_score, 3),
        })
    return itinerary


# ============================================================
# 5. CROSS-MEDIA DISCOVERY
# ============================================================

def unlock_related(seed_id: str, k_per_type: int = 2) -> list[dict]:
    """
    Given a finished item, surface items in OTHER media types
    that share its emotional arc.
    """
    seed = next((c for c in CONTENT if c["id"] == seed_id), None)
    if seed is None:
        return []
    seed_type = seed["type"]

    out = []
    for media_type in {"movie", "book", "podcast", "game", "music"} - {seed_type}:
        results = rank(
            target=seed["mood"],
            k=k_per_type,
            types=[media_type],
            query_themes=seed.get("themes", []),
        )
        for item, score in results:
            out.append({
                "item": item,
                "score": round(score, 3),
                "shared_themes": list(set(seed.get("themes", [])) & set(item.get("themes", []))),
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ============================================================
# 6. FESTIVAL GENERATOR
# ============================================================

FESTIVAL_TEMPLATES = [
    {
        "name": "The Lonely Sundays Festival",
        "tagline": "Three days for quiet grief, slow coffee, and the books that make you cry in a good way.",
        "mood_bias": {"melancholy": 0.7, "warmth": 0.6, "nostalgia": 0.7, "energy": 0.3, "romance": 0.4},
        "items": 5,
    },
    {
        "name": "The Wonder Engine Festival",
        "tagline": "Awe-soaked hours of strange skies, deep sea, and big questions.",
        "mood_bias": {"wonder": 0.85, "depth": 0.7, "mystery": 0.5, "energy": 0.5},
        "items": 5,
    },
    {
        "name": "The Big Hug Festival",
        "tagline": "Wholesome, hopeful, and gently funny — the cinematic equivalent of a friend's couch.",
        "mood_bias": {"warmth": 0.85, "hope": 0.8, "humor": 0.6, "nostalgia": 0.5, "valence": 0.7},
        "items": 5,
    },
    {
        "name": "The Knife-Edge Festival",
        "tagline": "Taut, twisty, and dark. Best consumed in one sitting with the lights low.",
        "mood_bias": {"tension": 0.85, "mystery": 0.8, "depth": 0.7, "warmth": 0.3, "valence": 0.35},
        "items": 5,
    },
    {
        "name": "The Late-Night Longing Festival",
        "tagline": "For 2 a.m. — yearning, romance, and a window you keep looking out of.",
        "mood_bias": {"romance": 0.8, "melancholy": 0.6, "nostalgia": 0.6, "warmth": 0.6, "energy": 0.3},
        "items": 5,
    },
]


def _this_weekend_window() -> tuple[date, date]:
    """Return (Friday, Sunday) of the upcoming weekend (or current if Friday)."""
    today = date.today()
    days_to_friday = (4 - today.weekday()) % 7
    fri = today + timedelta(days=days_to_friday)
    sun = fri + timedelta(days=2)
    return fri, sun


def generate_festival(template: dict, mood: dict | None = None) -> dict:
    """
    Pick a festival template (or pick one based on the user's mood if not given),
    then build a 5-item festival lineup from the content database.
    """
    if template is None:
        # not currently used, but kept for future caller
        template = random.choice(FESTIVAL_TEMPLATES)

    # bias the template's mood toward the user's preferences
    # initialize with all axes so partial-mood blends don't KeyError
    target = {ax: 0.5 for ax in MOOD_AXES}
    target.update(template["mood_bias"])
    if mood:
        for ax, val in mood.items():
            target[ax] = 0.6 * target[ax] + 0.4 * val

    # build a balanced lineup: 2 movies, 1 book, 1 game, 1 podcast
    lineup_specs = [
        ("movie",   2),
        ("book",    1),
        ("game",    1),
        ("podcast", 1),
    ]
    used = set()
    lineup = []
    for media_type, n in lineup_specs:
        results = rank(target=target, k=n, types=[media_type], exclude_ids=used)
        for item, score in results:
            if item["id"] in used:
                continue
            used.add(item["id"])
            lineup.append({"item": item, "score": round(score, 3)})
            if len([x for x in lineup if x["item"]["type"] == media_type]) >= n:
                break

    fri, sun = _this_weekend_window()
    return {
        "name": template["name"],
        "tagline": template["tagline"],
        "starts": fri.isoformat(),
        "ends": sun.isoformat(),
        "lineup": lineup,
    }


# ============================================================
# 7. UTILITIES
# ============================================================

def all_items() -> list[dict]:
    return CONTENT


def item_by_id(item_id: str) -> dict | None:
    return next((c for c in CONTENT if c["id"] == item_id), None)


def media_type_emoji(t: str) -> str:
    return {"movie": "🎬", "book": "📖", "podcast": "🎧", "game": "🎮", "music": "🎵", "tv": "📺"}.get(t, "✨")


# ============================================================
# 8. CATALOG-BACKED RANKING (Unity Catalog volume, ~12k items)
# ============================================================
# Two-stage retrieval to use the LLM sparingly:
#   1. rank_catalog()  — local cosine pre-rank over the full CSV
#      catalog (no LLM). Returns top-N candidates.
#   2. rerank_llm()    — ONE LLM call per query that re-ranks the
#      top-N candidates against the user's mood and writes a short
#      "why" for each kept pick. Falls back to the local order when
#      no key is set.
#
# The curated CONTENT list is kept as a small fallback for the
# cross-media / festival features that rely on books/podcasts/games
# which the two CSVs don't cover.

# NOTE: imported lazily inside the functions to avoid a circular import
# (data/catalog.py imports MOOD_AXES from this module at load time).
# How many candidates the local stage surfaces to the LLM re-ranker.
CATALOG_CANDIDATES = 20


def rank_catalog(
    target: dict,
    k: int = 5,
    types: Iterable[str] | None = None,
    exclude_ids: Iterable[str] | None = None,
    candidates: int = CATALOG_CANDIDATES,
) -> list[tuple[dict, float]]:
    """Local cosine pre-rank over the full CSV catalog. No LLM.

    Returns up to `candidates` (item, score) pairs sorted by mood
    similarity, filtered by media type and exclusions.
    """
    from data.catalog import load_catalog  # lazy: avoid circular import
    pool = load_catalog()
    exclude_ids = set(exclude_ids or [])
    types_set = set(types) if types else None

    scored: list[tuple[dict, float]] = []
    for item in pool:
        if item["id"] in exclude_ids:
            continue
        if types_set and item["type"] not in types_set:
            continue
        scored.append((item, _cosine(target, item["mood"])))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(k, candidates)]


def _mood_phrase_for_prompt(target: dict) -> str:
    """Compact human-readable mood description for the LLM prompt."""
    full = {ax: target.get(ax, 0.5) for ax in MOOD_AXES}
    bits = [
        f"{ax}: {_mood_to_phrase(ax, v)}"
        for ax, v in full.items()
        if abs(v - 0.5) > 0.15
    ]
    return ", ".join(bits) if bits else "a quiet, open feeling"


def rerank_llm(
    target: dict,
    candidates: list[tuple[dict, float]],
    k: int = 5,
) -> list[dict]:
    """Re-rank `candidates` with ONE LLM call and attach a 'why' to each.

    Returns a list of {item, score, why} of length <= k. Falls back to
    the local order + rule-based explain() when no key is set or the
    call fails — so the app always works.
    """
    if not candidates:
        return []

    client = _get_openai_client()
    # No key -> keep local order, use rule-based why.
    if client is None:
        out = []
        for item, score in candidates[:k]:
            out.append({
                "item": item,
                "score": round(score, 3),
                "why": explain(item, target),
            })
        return out

    # Build a compact candidate list for the prompt. Cap descriptions
    # to keep the prompt small (one call, cheap).
    mood_str = _mood_phrase_for_prompt(target)
    cand_lines = []
    for i, (item, score) in enumerate(candidates):
        desc = (item.get("description") or item.get("pitch") or "").strip()
        desc = (desc[:240] + "…") if len(desc) > 240 else desc
        genres = ", ".join(item.get("genres", [])[:4]) or "n/a"
        cand_lines.append(
            f"{i+1}. {item['title']} ({item['type']}, {item['year']}) "
            f"[genres: {genres}] — {desc}"
        )
    cand_block = "\n".join(cand_lines)

    system = (
        "You are the re-ranker for Hackey Specter, a mood-first "
        "recommendation app. You receive a user's target mood and a list "
        "of candidate titles (pre-ranked by a local mood-similarity model). "
        "Pick the top titles that best match how the user wants to feel, "
        "and for each write one short, evocative sentence (<=30 words) "
        "explaining why it fits that mood. Return STRICT JSON only: "
        '{"picks":[{"rank":1,"reason":"..."}]}. The number of picks must '
        "be <= the requested k. Use the candidate numbers as ranks. "
        "No markdown, no prose outside the JSON."
    )
    user = (
        f"The user wants to feel: {mood_str}.\n"
        f"Candidates (pre-ranked):\n{cand_block}\n"
        f"Return the top {k} as JSON."
    )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text.strip() if hasattr(resp, "output_text") else ""
        if not text:
            text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""

    # Parse the LLM's re-ranked picks; fall back to local order on any
    # parse failure so the UI never breaks.
    order: list[int] = []  # 1-based candidate indices, in LLM-preferred order
    whys: dict[int, str] = {}
    if text:
        try:
            import json as _json
            # tolerate stray markdown fences
            t = text.strip()
            if t.startswith("```"):
                t = t.strip("`")
                t = t[t.find("{"):t.rfind("}") + 1]
            data = _json.loads(t)
            for p in data.get("picks", []):
                r = int(p.get("rank"))
                if 1 <= r <= len(candidates):
                    order.append(r)
                    reason = (p.get("reason") or "").strip()
                    if reason:
                        whys[r] = reason
        except Exception:
            order = []

    if not order:
        order = list(range(1, len(candidates) + 1))

    out = []
    for r in order[:k]:
        item, score = candidates[r - 1]
        out.append({
            "item": item,
            "score": round(score, 3),
            "why": whys.get(r) or explain(item, target),
        })
    return out


def recommend(
    target: dict,
    k: int = 5,
    types: Iterable[str] | None = None,
    exclude_ids: Iterable[str] | None = None,
) -> list[dict]:
    """End-to-end catalog recommendation: local pre-rank + LLM re-rank.

    This is the single entry point the UI should call. It does at most
    ONE LLM call per query (in rerank_llm), regardless of catalog size.
    """
    candidates = rank_catalog(
        target=target, k=k, types=types, exclude_ids=exclude_ids,
        candidates=CATALOG_CANDIDATES,
    )
    return rerank_llm(target, candidates, k=k)


# ============================================================
# 9. SPEECH: STT (microphone) + TTS (speaker)
# ============================================================
# Both use the same OpenAI client as the LLM explainer. They fall back
# gracefully (return None / empty) when no key is set so the app keeps
# working in local dev.

# OpenAI TTS has a ~4096 char input limit; truncate to be safe.
_TTS_MAX_CHARS = 3000


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str | None:
    """Transcribe microphone audio to text using OpenAI Whisper (STT).

    Returns the transcribed text, or None if no key is set or the call
    fails. `mime_type` should match the recorded audio format (Streamlit's
    st.audio_input typically produces audio/webm).
    """
    client = _get_openai_client()
    if client is None:
        return None
    if not audio_bytes:
        return None
    # derive a filename with the right extension for the API
    ext = "webm"
    if "mp3" in mime_type:
        ext = "mp3"
    elif "wav" in mime_type:
        ext = "wav"
    elif "ogg" in mime_type:
        ext = "ogg"
    filename = f"recording.{ext}"
    model = os.environ.get("OPENAI_STT_MODEL", "whisper-1")
    try:
        resp = client.audio.transcriptions.create(
            model=model,
            file=(filename, audio_bytes, mime_type),
        )
        # the SDK returns a string by default (response_format=text)
        if isinstance(resp, str):
            return resp.strip() or None
        # verbose response objects have .text
        return (getattr(resp, "text", "") or "").strip() or None
    except Exception:
        return None


def speak_text(text: str, voice: str = "alloy") -> bytes | None:
    """Synthesize speech from text using OpenAI TTS.

    Returns MP3 audio bytes, or None if no key is set or the call fails.
    The text is truncated to _TTS_MAX_CHARS to stay within the API limit.
    """
    client = _get_openai_client()
    if client is None:
        return None
    if not text:
        return None
    text = text.strip()[:_TTS_MAX_CHARS]
    model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        # HttpxBinaryResponseContent — .content holds the bytes
        return resp.content
    except Exception:
        return None
