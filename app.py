"""
Hackey Specter — main Streamlit app.

A 6-hour hackathon prototype that recommends entertainment based on mood,
plans weekend itineraries across media, and links stories across formats.

Run: streamlit run app.py
"""
from __future__ import annotations

import random
import logging
import hashlib
import html
import os
import time
import traceback
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urlparse

import streamlit as st

import mood_engine as M
from data.content import SAMPLE_PROMPTS
from audio_utils import generate_summary_audio, is_playable_audio_url, transcribe_voice_input
from recommend_and_pitch import generate_final_recommendation
from search_engine import StoryDatabase, canonicalize_query, load_database, vector_search
from vector_store import materialize_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VECTOR_STORE_FILENAME = "ultimate_pocketfm_vector_store.npz"
DEFAULT_VECTOR_STORE_PATH = Path(__file__).resolve().with_name(VECTOR_STORE_FILENAME)


def _resolve_vector_store_path() -> Path:
    """Resolve a local override first, then a Databricks UC volume resource."""
    explicit_path = os.getenv("RAG_VECTOR_STORE_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser()

    volume_path = os.getenv("RAG_VECTOR_STORE_VOLUME", "").strip()
    if volume_path:
        return Path(volume_path).expanduser() / VECTOR_STORE_FILENAME

    return DEFAULT_VECTOR_STORE_PATH


VECTOR_STORE_PATH = _resolve_vector_store_path()
MAX_SESSION_SEARCH_CACHE_ENTRIES = 20
T = TypeVar("T")
_loaded_vector_store_signature: str | None = None


def _vector_store_signature(vector_store_path: Path) -> str:
    if not vector_store_path.exists():
        return "missing"
    stat = vector_store_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_resource(show_spinner=False)
def get_story_database(path: str, signature: str) -> StoryDatabase:
    """Load the offline-built vector store; never rebuild embeddings during app use."""
    vector_path = Path(path)
    try:
        vector_path.stat()
    except OSError as exc:
        raise FileNotFoundError(
            "The offline vector store cannot be accessed at "
            f"{vector_path}: {type(exc).__name__} [errno={exc.errno}] {exc.strerror}. "
            "RAG_VECTOR_STORE_VOLUME is "
            f"{os.getenv('RAG_VECTOR_STORE_VOLUME', '<not set>')!r}. "
            "Verify the file path and the app service principal's UC volume grants."
        ) from exc
    if not vector_path.is_file():
        raise FileNotFoundError(f"The vector store path is not a file: {vector_path}.")
    return load_database(str(vector_path))


def get_active_story_database() -> StoryDatabase:
    """Refresh the cached database automatically when the offline store is replaced."""
    global _loaded_vector_store_signature
    vector_store_path = materialize_vector_store(str(VECTOR_STORE_PATH))
    signature = _vector_store_signature(vector_store_path)
    if _loaded_vector_store_signature not in (None, signature):
        # Avoid retaining an old full matrix after repeated offline rebuilds.
        get_story_database.clear()
    _loaded_vector_store_signature = signature
    return get_story_database(str(vector_store_path), signature)


def _story_cache_key(database: StoryDatabase, search_request: str, namespace: str) -> str:
    """Build a privacy-preserving, whitespace-stable key for session-only caches."""
    material = f"{namespace}\0{database.build_fingerprint}\0{canonicalize_query(search_request)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _session_cache_get_or_set(cache_name: str, cache_key: str, factory: Callable[[], T]) -> T:
    """Use a small LRU-like session cache without retaining unlimited user prompts."""
    cache = st.session_state.setdefault(cache_name, {})
    if cache_key in cache:
        value = cache.pop(cache_key)
        cache[cache_key] = value
        return value
    value = factory()
    cache[cache_key] = value
    while len(cache) > MAX_SESSION_SEARCH_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))
    return value


def run_story_retrieval(search_request: str) -> list[dict]:
    """Run only the fast retrieval stage; a GPT pitch is intentionally separate."""
    database = get_active_story_database()
    cache_key = _story_cache_key(database, search_request, "retrieval")
    return _session_cache_get_or_set(
        "rag_retrieval_cache",
        cache_key,
        lambda: vector_search(search_request, database, top_k=5),
    )


def run_story_rerank(search_request: str, candidates: list[dict]) -> dict:
    """Generate a cached optional pitch after instant vector results are visible."""
    database = get_active_story_database()
    candidate_signature = ";".join(
        f"{item.get('record_id', '')}:{float(item.get('vector_similarity', 0.0)):.6f}"
        for item in candidates
    )
    cache_key = hashlib.sha256(
        f"{_story_cache_key(database, search_request, 'rerank')}\0{candidate_signature}".encode("utf-8")
    ).hexdigest()
    return _session_cache_get_or_set(
        "rag_rerank_cache",
        cache_key,
        lambda: generate_final_recommendation(search_request, candidates),
    )

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hackey Specter",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# THEME — cozy, moody, literary. Hand-tuned CSS, no external assets.
# ============================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg: #faf6ef;          /* warm cream paper */
  --bg-soft: #f1ebde;
  --bg-card: #ffffff;     /* slightly brighter card */
  --bg-card-hover: #f7f1e3;
  --ink: #2a2520;         /* deep warm brown */
  --ink-soft: #5a4f44;
  --ink-dim: #8a7f72;
  --accent: #b87a7a;      /* deeper dusty rose */
  --accent-warm: #b88040; /* deeper amber */
  --accent-cool: #5a8a7a; /* deeper teal */
  --border: #e6dfd0;
  --shadow: 0 4px 20px rgba(80, 60, 40, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

h1, h2, h3, h4 {
  font-family: 'Fraunces', Georgia, serif !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}

.hero-title {
  font-family: 'Fraunces', serif;
  font-size: 3.4rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
  line-height: 1.05;
}
.hero-title .accent { color: var(--accent); font-style: italic; }
.hero-sub {
  font-family: 'Inter', sans-serif;
  font-size: 1.05rem;
  color: var(--ink-soft);
  margin-top: 0.5rem;
  max-width: 640px;
  line-height: 1.55;
}

[data-testid="stTabs"] [role="tablist"] {
  gap: 0.5rem;
  background: transparent;
  border-bottom: 1px solid var(--border);
  padding: 0;
}
[data-testid="stTabs"] button[role="tab"] {
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  color: var(--ink-dim);
  background: transparent;
  border: none;
  padding: 0.85rem 1.1rem;
  border-radius: 0;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--ink-soft);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  margin-bottom: 1.25rem;
  box-shadow: 0 6px 24px rgba(80, 60, 40, 0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 32px rgba(80, 60, 40, 0.12);
}
.card-inner {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
}
.card-thumbnail {
  position: relative;
  flex: 0 0 170px;
  min-height: 170px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--bg-soft), var(--border));
}
.card-thumbnail.movie { background: linear-gradient(135deg, #f3d5d5, #c9a0a0); }
.card-thumbnail.book { background: linear-gradient(135deg, #f3e0c5, #d4b88c); }
.card-thumbnail.podcast { background: linear-gradient(135deg, #cde7de, #8ab8aa); }
.card-thumbnail.game { background: linear-gradient(135deg, #e4d9f2, #b8a7d1); }
.card-thumbnail.music { background: linear-gradient(135deg, #e6d4e6, #c4a8c4); }
.card-art {
  font-size: 3.2rem;
  opacity: 0.85;
  filter: drop-shadow(0 2px 4px rgba(42,37,32,0.12));
}
.card-body {
  flex: 1 1 280px;
  padding: 1.25rem 1.4rem;
  min-width: 0;
}
.card .type-chip {
  display: inline-block;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0.22rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--ink-soft);
  margin-bottom: 0.7rem;
}
.card .type-chip.movie   { color: var(--accent); border-color: rgba(184,122,122,0.35); }
.card .type-chip.book    { color: var(--accent-warm); border-color: rgba(184,128,64,0.35); }
.card .type-chip.podcast { color: var(--accent-cool); border-color: rgba(90,138,122,0.35); }
.card .type-chip.game    { color: #7b5fa6; border-color: rgba(123,95,166,0.30); }
.card-title {
  margin: 0 0 0.35rem 0;
  font-size: 1.45rem;
  line-height: 1.15;
  color: var(--ink);
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 500;
}
.card-score {
  font-size: 0.8rem;
  color: var(--ink-dim);
  font-weight: 400;
  font-family: 'Inter', sans-serif;
  margin-left: 0.4rem;
}
.card-creator {
  font-size: 0.9rem;
  color: var(--ink-dim);
  margin-bottom: 0.75rem;
}
.card-pitch {
  font-size: 0.96rem;
  color: var(--ink-soft);
  line-height: 1.55;
}
.card .themes {
  margin-top: 0.75rem;
  font-size: 0.82rem;
  color: var(--ink-dim);
}
@media (max-width: 640px) {
  .card-thumbnail {
    flex: 1 1 100%;
    height: 180px;
  }
  .card-body {
    flex: 1 1 100%;
    padding: 1rem;
  }
  }
}

.why-box {
  background: linear-gradient(135deg, rgba(184,122,122,0.07), rgba(184,128,64,0.05));
  border-left: 3px solid var(--accent);
  border-radius: 0 10px 10px 0;
  padding: 1rem 1.2rem;
  margin-top: 0.7rem;
  font-size: 0.92rem;
  line-height: 1.6;
  color: var(--ink-soft);
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
}

.festival-hero {
  background: radial-gradient(ellipse at top, rgba(184,122,122,0.10), transparent 70%),
              linear-gradient(180deg, #ffffff, var(--bg-soft));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow);
}
.festival-hero h2 { margin: 0; font-size: 2.2rem; font-style: italic; color: var(--accent); }
.festival-hero .tagline {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.05rem;
  font-style: italic;
  color: var(--ink-soft);
  margin-top: 0.5rem;
  max-width: 600px;
  line-height: 1.5;
}
.festival-hero .window {
  margin-top: 1rem;
  font-size: 0.8rem;
  color: var(--ink-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.itinerary-slot {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  box-shadow: var(--shadow);
}
.slot-label {
  flex: 0 0 200px;
  font-size: 0.78rem;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 500;
}
.slot-item { flex: 1; }
.slot-item h4 { margin: 0; font-size: 1.05rem; color: var(--ink); font-family: 'Fraunces', serif; }
.slot-item .meta { font-size: 0.78rem; color: var(--ink-dim); }
.slot-emoji { font-size: 1.5rem; flex: 0 0 2rem; text-align: center; }

.mood-pill {
  display: inline-block;
  background: rgba(90,138,122,0.10);
  border: 1px solid rgba(90,138,122,0.28);
  color: var(--accent-cool);
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  font-size: 0.78rem;
  margin: 0.15rem;
}

.mood-vec-bar {
  display: inline-block;
  width: 80px;
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
  vertical-align: middle;
  margin-left: 0.4rem;
}
.mood-vec-bar > span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--accent-cool), var(--accent), var(--accent-warm));
}

.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.6rem 1.4rem !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: var(--accent-warm) !important;
  transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

.stTextInput input, .stTextArea textarea {
  background: var(--bg-soft) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--ink) !important;
  font-family: 'Fraunces', Georgia, serif !important;
  font-size: 1.05rem !important;
  padding: 0.7rem 1rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(184,122,122,0.15) !important;
}

.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-soft) !important;
  border: 1px solid var(--border) !important;
  color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stMarkdownContainer"] p { line-height: 1.6; }

.divider {
  height: 1px;
  background: var(--border);
  margin: 2rem 0;
}

.section-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-dim);
  margin-bottom: 0.8rem;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []   # list of (text, parsed_mood, top_item)
if "finished" not in st.session_state:
    st.session_state.finished = []  # list of item ids
if "weekend_rag" not in st.session_state:
    st.session_state.weekend_rag = None
if "festival_rag" not in st.session_state:
    st.session_state.festival_rag = None


# ============================================================
# HELPERS
# ============================================================
def _set_mood_sample(sample: str) -> None:
    """Run before widgets are instantiated so Streamlit can safely update the input."""
    st.session_state["mood_input"] = sample


def _set_weekend_sample() -> None:
    st.session_state["weekend_input"] = random.choice(SAMPLE_PROMPTS)


def _safe_text(value: object) -> str:
    """Escape dynamic text before it is placed in unsafe_allow_html markup."""
    return html.escape(str(value), quote=True)


def _is_safe_http_url(value: object) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _show_operation_error(action: str, exc: Exception) -> None:
    """Print the real error on screen. No abstraction, no hiding."""
    logger.exception("%s failed", action, exc_info=exc)
    st.error(f"{action} failed: {type(exc).__name__}: {exc}")
    with st.expander(f"Traceback for: {action}", expanded=True):
        st.code(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
            or f"(no traceback available)\n{exc!r}",
            language="text",
        )


def render_env_status() -> None:
    """Tiny env-status block. No buttons, no panels — just the facts."""
    with st.expander("Environment", expanded=False):
        key = os.environ.get("OPENAI_API_KEY", "")
        st.markdown(f"- `OPENAI_API_KEY`: {'set' if key else 'NOT SET'} ({len(key)} chars)")
        for var in (
            "RAG_RERANK_MODEL",
            "RAG_EMBEDDING_MODEL",
            "RAG_FEATURE_MODEL",
            "RAG_TTS_MODEL",
            "RAG_TRANSCRIPTION_MODEL",
        ):
            st.markdown(f"- `{var}`: `{os.environ.get(var, '<default>')}`")
        # Surface any other env vars that look like API keys — catches cases
        # where the Databricks runtime injected the secret under a different name.
        st.markdown("---")
        st.markdown("**Other env vars matching `OPENAI` or `API_KEY`:**")
        matches = sorted(
            k for k in os.environ.keys()
            if ("OPENAI" in k.upper() or "API_KEY" in k.upper()) and k != "OPENAI_API_KEY"
        )
        if matches:
            for k in matches:
                v = os.environ[k]
                st.markdown(f"- `{k}`: set ({len(v)} chars)")
        else:
            st.caption("(none)")
        st.markdown("---")
        with st.expander("All env var names (no values)", expanded=False):
            for k in sorted(os.environ.keys()):
                v = os.environ[k]
                st.markdown(f"- `{k}` ({len(v)} chars)")


# Keep the environment diagnostic available for local troubleshooting, but do
# not render it in the production UI.
# render_env_status()


def render_card(item: dict, score: float | None = None) -> str:
    raw_type = str(item.get("type", "book"))
    safe_type = raw_type if raw_type in {"movie", "book", "podcast", "game", "music"} else "book"
    emoji = _safe_text(M.media_type_emoji(safe_type))
    score_html = f'<span class="card-score">{float(score):.2f} match</span>' if score is not None else ""
    themes_html = ""
    if item.get("themes"):
        themes_html = f'<div class="themes">{" · ".join(_safe_text(theme) for theme in item["themes"][:5])}</div>'
    return f"""
    <div class="card">
      <div class="card-inner">
        <div class="card-thumbnail {safe_type}">
          <span class="card-art">{emoji}</span>
        </div>
        <div class="card-body">
          <div class="type-chip {safe_type}">{emoji} {_safe_text(raw_type)}</div>
          <h3 class="card-title">{_safe_text(item.get('title', 'Untitled'))}{score_html}</h3>
          <div class="card-creator">{_safe_text(item.get('creator', 'Unknown'))} · {_safe_text(item.get('year', ''))}</div>
          <div class="card-pitch">{_safe_text(item.get('pitch', ''))}</div>
          {themes_html}
        </div>
      </div>
    </div>
    """


def render_summary_card(item: dict, score: float) -> str:
    """Render a result from the imported offline story-vector store."""
    summary = str(item.get("summary", "")).strip()
    preview = summary[:700] + ("…" if len(summary) > 700 else "")
    title = _safe_text(item.get("title", "Untitled"))
    link = str(item.get("librivox_project_url", ""))
    safe_link = _safe_text(link) if _is_safe_http_url(link) else ""
    link_html = f'<a href="{safe_link}" target="_blank" rel="noopener noreferrer">Open on LibriVox</a>' if safe_link else ""
    emoji = _safe_text(M.media_type_emoji("book"))
    return f'''<div class="card">
      <div class="card-inner">
        <div class="card-thumbnail book">
          <span class="card-art">{emoji}</span>
        </div>
        <div class="card-body">
          <div class="type-chip book">Story summary · {score:.2f} match</div>
          <h3 class="card-title">{title}</h3>
          <div class="card-pitch">{_safe_text(preview)}</div>
          {link_html}
        </div>
      </div>
    </div>'''


def render_why(item: dict, target: dict) -> str:
    return f"""
    <div class="why-box">
      <strong style="font-style: normal; color: var(--accent);">Why you will love this:</strong><br>
      {_safe_text(M.explain(item, target))}
    </div>
    """


def render_mood_summary(target: dict) -> str:
    """Render the parsed mood as pills with tiny value bars."""
    # only show axes that aren't exactly 0.5
    deltas = {k: round((v - 0.5) * 2, 2) for k, v in target.items() if abs(v - 0.5) > 0.15}
    if not deltas:
        return '<div class="mood-pill">a quiet, open feeling</div>'
    parts = []
    for ax, delta in sorted(deltas.items(), key=lambda x: -abs(x[1])):
        direction = "↑" if delta > 0 else "↓"
        bar_width = int(abs(delta) * 80)
        bar_color = "var(--accent-warm)" if delta > 0 else "var(--accent-cool)"
        parts.append(
            f'<span class="mood-pill">{ax} {direction}'
            f'<span class="mood-vec-bar"><span style="width:{bar_width}px; background:{bar_color};"></span></span>'
            f'</span>'
        )
    return "".join(parts[:8])


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div style="padding: 1.5rem 0 1rem 0;">
      <h1 class="hero-title">Hackey <span class="accent">Specter</span></h1>
      <p class="hero-sub">
        Tell us how you want to feel. We'll find the movie, book, podcast, or game that gets you there —
        and explain why it'll be the one that stays with you.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================
tab_mood, tab_concierge, tab_festival = st.tabs([
    "🔎  Tell us your mood",
    "🎯  Weekend Concierge",
    "🎬  Festivals",
])


# ============================================================
# TAB 2: WEEKEND CONCIERGE
# ============================================================
with tab_concierge:
    st.markdown('<div class="section-label">Plan a whole weekend around a feeling</div>', unsafe_allow_html=True)
    concierge_prompt = st.text_input(
        "weekend_mood",
        placeholder="Like 3am with my best friend talking about life",
        label_visibility="collapsed",
        key="weekend_input",
    )

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        plan_clicked = st.button("Plan my weekend", type="primary", use_container_width=True)
    with c2:
        st.button("Try a sample", use_container_width=True, on_click=_set_weekend_sample)

    active_weekend_prompt = concierge_prompt.strip()
    if plan_clicked and active_weekend_prompt:
        try:
            logger.info("[RAG] weekend concierge search submitted: query=%r", active_weekend_prompt)
            with st.spinner("Finding stories for your weekend..."):
                candidates = run_story_retrieval(active_weekend_prompt)
                st.session_state.weekend_rag = {
                    "query": active_weekend_prompt,
                    "candidates": candidates,
                    "recommendation": None,
                    "rerank_state": "pending",
                }
        except Exception as exc:
            _show_operation_error("Weekend search", exc)
    elif plan_clicked:
        st.warning("Describe the kind of weekend you want before generating a plan.")

    saved_weekend = st.session_state.get("weekend_rag")
    if saved_weekend:
        results = saved_weekend.get("candidates", [])
        recommendation = saved_weekend.get("recommendation")
        if results:
            st.markdown(f"### Stories for your weekend")
            for item in results:
                st.markdown(
                    render_summary_card(item, item.get("vector_similarity", 0.0)),
                    unsafe_allow_html=True,
                )
                rec_id = item.get("record_id", "")
                if rec_id:
                    audio_key = f"audio_{rec_id}"
                    if st.button("▶ Play summary", key=f"play_weekend_{rec_id}"):
                        with st.spinner("Generating audio..."):
                            st.session_state[audio_key] = generate_summary_audio(item)
                    if audio_key in st.session_state:
                        st.audio(st.session_state[audio_key], format="audio/mpeg")
            if isinstance(recommendation, dict):
                recommended_story = next(
                    (item for item in results if item.get("record_id") == recommendation.get("recommended_record_id")),
                    results[0],
                )
                st.subheader(f"Recommended: {recommendation.get('recommended_book_title', 'Untitled')}")
                st.markdown(str(recommendation.get("pitch_script", "")))
                reasons = recommendation.get("emotional_match_reasons", [])
                if isinstance(reasons, list) and reasons:
                    st.markdown("**Why it fits:** " + " · ".join(str(reason) for reason in reasons))
                if recommended_story.get("vector_similarity", 0) < 0.4:
                    st.info(
                        "We couldn't find a perfect match for your request. "
                        "The recommendation above is our closest available story, "
                        "but it may not fully capture the mood you described. "
                        "Try rephrasing your request or exploring a different theme."
                    )
            if recommendation is None and saved_weekend.get("rerank_state", "pending") == "pending":
                try:
                    logger.info("[RAG] weekend concierge starting LLM rerank for query=%r", saved_weekend.get("query", ""))
                    with st.spinner("Writing your personalized recommendation..."):
                        saved_weekend["recommendation"] = run_story_rerank(
                            str(saved_weekend.get("query", "")),
                            results,
                        )
                        saved_weekend["rerank_state"] = "complete"
                        st.session_state.weekend_rag = saved_weekend
                    st.rerun()
                except Exception as exc:
                    logger.warning("[RAG] weekend rerank FAILED: %s: %s", type(exc).__name__, exc)
                    saved_weekend["rerank_state"] = "failed"
                    st.session_state.weekend_rag = saved_weekend
                    _show_operation_error("Personalized pitch", exc)
            elif recommendation is None and saved_weekend.get("rerank_state") == "failed":
                st.warning("The semantic matches are available, but the personalized pitch could not be generated.")
        else:
            st.info("No matching stories found for that mood. Try a different description.")



# ============================================================
# TAB 3: FESTIVALS
# ============================================================
with tab_festival:
    st.markdown('<div class="section-label">Festival-themed stories, curated by mood</div>', unsafe_allow_html=True)

    fest_prompt = st.text_input(
        "fest_mood",
        placeholder="Describe the festival mood you're in the mood for...",
        label_visibility="collapsed",
        key="fest_input",
    )

    _, c2 = st.columns([3, 1])
    with c2:
        gen_clicked = st.button("Generate festival stories", type="primary", use_container_width=True)

    if gen_clicked:
        if fest_prompt.strip():
            query = f"Festival themed stories for {fest_prompt.strip()}"
            try:
                logger.info("[RAG] festival search submitted: query=%r", query)
                with st.spinner("Finding festival stories..."):
                    candidates = run_story_retrieval(query)
                    st.session_state.festival_rag = {
                        "query": query,
                        "candidates": candidates,
                        "recommendation": None,
                        "rerank_state": "pending",
                    }
            except Exception as exc:
                _show_operation_error("Festival search", exc)
        else:
            st.warning("Enter a mood or theme for the festival.")

    saved_festival = st.session_state.get("festival_rag")
    if saved_festival:
        results = saved_festival.get("candidates", [])
        recommendation = saved_festival.get("recommendation")
        if results:
            st.markdown(f"### Festival stories")
            for item in results:
                st.markdown(
                    render_summary_card(item, item.get("vector_similarity", 0.0)),
                    unsafe_allow_html=True,
                )
                rec_id = item.get("record_id", "")
                if rec_id:
                    audio_key = f"audio_{rec_id}"
                    if st.button("▶ Play summary", key=f"play_festival_{rec_id}"):
                        with st.spinner("Generating audio..."):
                            st.session_state[audio_key] = generate_summary_audio(item)
                    if audio_key in st.session_state:
                        st.audio(st.session_state[audio_key], format="audio/mpeg")
            if isinstance(recommendation, dict):
                recommended_story = next(
                    (item for item in results if item.get("record_id") == recommendation.get("recommended_record_id")),
                    results[0],
                )
                st.subheader(f"Recommended: {recommendation.get('recommended_book_title', 'Untitled')}")
                st.markdown(str(recommendation.get("pitch_script", "")))
                reasons = recommendation.get("emotional_match_reasons", [])
                if isinstance(reasons, list) and reasons:
                    st.markdown("**Why it fits:** " + " · ".join(str(reason) for reason in reasons))
                if recommended_story.get("vector_similarity", 0) < 0.4:
                    st.info(
                        "We couldn't find a perfect match for your request. "
                        "The recommendation above is our closest available story, "
                        "but it may not fully capture the mood you described. "
                        "Try rephrasing your request or exploring a different theme."
                    )
            if recommendation is None and saved_festival.get("rerank_state", "pending") == "pending":
                try:
                    logger.info("[RAG] festival starting LLM rerank for query=%r", saved_festival.get("query", ""))
                    with st.spinner("Writing your personalized recommendation..."):
                        saved_festival["recommendation"] = run_story_rerank(
                            str(saved_festival.get("query", "")),
                            results,
                        )
                        saved_festival["rerank_state"] = "complete"
                        st.session_state.festival_rag = saved_festival
                    st.rerun()
                except Exception as exc:
                    logger.warning("[RAG] festival rerank FAILED: %s: %s", type(exc).__name__, exc)
                    saved_festival["rerank_state"] = "failed"
                    st.session_state.festival_rag = saved_festival
                    _show_operation_error("Personalized pitch", exc)
            elif recommendation is None and saved_festival.get("rerank_state") == "failed":
                st.warning("The semantic matches are available, but the personalized pitch could not be generated.")
        else:
            st.info("No matching stories found for that theme. Try a different description.")


# ============================================================
# TAB 1: TELL US YOUR MOOD
# ============================================================
with tab_mood:
    st.markdown('<div class="section-label">Semantic story search</div>', unsafe_allow_html=True)
    voice_recording = st.audio_input("🎙 Speak your story request")
    if voice_recording:
        voice_bytes = voice_recording.getvalue()
        recording_id = hashlib.sha256(voice_bytes).hexdigest()
        if recording_id != st.session_state.get("last_voice_recording_id"):
            # Mark this recording before calling the API so a failed request is
            # not retried on every unrelated Streamlit rerun.
            st.session_state.last_voice_recording_id = recording_id
            try:
                with st.spinner("You finished speaking — transcribing with OpenAI…"):
                    transcript = transcribe_voice_input(
                        voice_bytes,
                        voice_recording.name or "story-request.wav",
                        voice_recording.type or "audio/wav",
                    )
                    st.session_state.voice_story_request = transcript
                    # This code runs before the text widget is created, which is
                    # the safe point to populate it during a Streamlit rerun.
                    st.session_state["story_search_input"] = transcript
                    st.session_state["pending_voice_search"] = True
                    st.session_state.pop("failed_voice_recording_id", None)
            except Exception as exc:
                st.session_state["failed_voice_recording_id"] = recording_id
                _show_operation_error("Voice transcription", exc)

    last_voice_recording_id = st.session_state.get("last_voice_recording_id")
    if (
        last_voice_recording_id
        and st.session_state.get("failed_voice_recording_id") == last_voice_recording_id
    ):
        st.caption("The current recording was not transcribed. You can retry it or record a new request.")
        if st.button("Retry transcription", key="retry_voice_transcription"):
            st.session_state.pop("last_voice_recording_id", None)
            st.session_state.pop("failed_voice_recording_id", None)
            st.rerun()

    voice_story_request = st.session_state.get("voice_story_request", "")
    if voice_story_request:
        voice_caption, clear_voice = st.columns([5, 1])
        with voice_caption:
            st.caption(f"Voice input: “{voice_story_request}”")
        with clear_voice:
            if st.button("Clear voice", key="clear_voice_story_request"):
                st.session_state.pop("voice_story_request", None)
                st.rerun()
    # A form prevents Streamlit from rerunning the whole page on every keystroke.
    with st.form("story_search_form"):
        story_request = st.text_input(
            "story_search",
            placeholder="A hopeful story about rebuilding after a breakup",
            label_visibility="collapsed",
            key="story_search_input",
        )
        search_submitted = st.form_submit_button("Search stories", type="primary", use_container_width=True)

    auto_search_requested = bool(st.session_state.pop("pending_voice_search", False))
    search_request = story_request.strip() or voice_story_request
    if search_submitted or auto_search_requested:
        if not search_request.strip():
            st.warning("Enter a story, mood, or theme before searching.")
        else:
            try:
                # This critical path makes one query-embedding request, then exact
                # local NumPy ranking. GPT reranking is deliberately deferred.
                logger.info("[RAG] story search submitted: query=%r", search_request)
                with st.spinner("Finding the closest semantic story matches..."):
                    candidates = run_story_retrieval(search_request)
                    logger.info(
                        "[RAG] vector search returned %d candidates: %s",
                        len(candidates),
                        ", ".join(
                            f"{c.get('title', '?')[:40]}@{float(c.get('vector_similarity', 0.0)):.3f}"
                            for c in candidates
                        ),
                    )
                    st.session_state.story_search = {
                        "query": search_request,
                        "candidates": candidates,
                        "recommendation": None,
                        "rerank_state": "pending",
                    }
                    st.session_state.pop("recommendation_audio", None)
                    if auto_search_requested or (not story_request.strip() and voice_story_request):
                        # Do not let a completed recording silently become the next
                        # blank search query. The recorder hash still prevents it
                        # from being transcribed again on the following rerun.
                        st.session_state.pop("voice_story_request", None)
            except Exception as exc:
                logger.warning(
                    "[RAG] retrieval FAILED for query=%r: %s: %s",
                    search_request, type(exc).__name__, exc,
                )
                _show_operation_error("Story search", exc)

    saved_search = st.session_state.get("story_search")
    # Discard session data produced by the previous one-step search implementation.
    if saved_search and not isinstance(saved_search, dict):
        st.session_state.pop("story_search", None)
        saved_search = None
    if saved_search:
        results = saved_search.get("candidates", [])
        recommendation = saved_search.get("recommendation")
        if not isinstance(results, list):
            results = []
        if not results:
            st.info("No stored story summaries matched those keywords. Try a different description.")
        else:
            recommended_story = results[0]
            if isinstance(recommendation, dict):
                recommended_story = next(
                    (item for item in results if item.get("record_id") == recommendation.get("recommended_record_id")),
                    results[0],
                )
                st.subheader(f"Recommended: {recommendation.get('recommended_book_title', 'Untitled')}")
                st.markdown(str(recommendation.get("pitch_script", "")))
                reasons = recommendation.get("emotional_match_reasons", [])
                if isinstance(reasons, list) and reasons:
                    st.markdown("**Why it fits:** " + " · ".join(str(reason) for reason in reasons))
            else:
                st.subheader(f"Best semantic match: {recommended_story.get('title', 'Untitled')}")
                st.caption(
                    "These results are ready from the saved vector index. Personalizing the final "
                    "recommendation with your exact mood…"
                )

            # Always derive audio metadata from the selected stored story, never from model text.
            source_audio_url = str(recommended_story.get("audio_url") or "")
            project_url = str(recommended_story.get("librivox_project_url") or "")
            if _is_safe_http_url(project_url):
                st.link_button("Open on LibriVox", project_url)
            if is_playable_audio_url(source_audio_url):
                if st.button("Play recommendation", key="play_recommendation_source"):
                    st.session_state.recommendation_audio = ("source", source_audio_url)
            elif st.button("Play story summary with OpenAI", key="play_recommendation_summary"):
                try:
                    with st.spinner("Creating summary audio..."):
                        st.session_state.recommendation_audio = (
                            "tts",
                            generate_summary_audio(recommended_story),
                        )
                except Exception as exc:
                    _show_operation_error("Summary audio", exc)

            saved_audio = st.session_state.get("recommendation_audio")
            if saved_audio:
                source, audio = saved_audio
                if source == "source":
                    st.audio(audio)
                else:
                    st.audio(audio, format="audio/mpeg")
        st.markdown("### Closest semantic matches")
        if results:
            for item in results:
                st.markdown(
                    render_summary_card(item, item.get("vector_similarity", 0.0)),
                    unsafe_allow_html=True,
                )
                rec_id = item.get("record_id", "")
                if rec_id:
                    audio_key = f"audio_{rec_id}"
                    if st.button("▶ Play summary", key=f"play_mood_{rec_id}"):
                        with st.spinner("Generating audio..."):
                            st.session_state[audio_key] = generate_summary_audio(item)
                    if audio_key in st.session_state:
                        st.audio(st.session_state[audio_key], format="audio/mpeg")
            if recommendation is None and saved_search.get("rerank_state", "pending") == "pending":
                try:
                    # This second spinner is intentionally independent from
                    # retrieval: the semantic cards above remain visible while
                    # the model writes the final personalized pitch.
                    logger.info("[RAG] starting LLM rerank for query=%r", saved_search.get("query", ""))
                    with st.spinner("Writing your personalized recommendation..."):
                        saved_search["recommendation"] = run_story_rerank(
                            str(saved_search.get("query", "")),
                            results,
                        )
                        rec = saved_search["recommendation"]
                        logger.info(
                            "[RAG] rerank complete: title=%r record_id=%s reasons=%d pitch_len=%d",
                            rec.get("recommended_book_title"),
                            rec.get("recommended_record_id"),
                            len(rec.get("emotional_match_reasons", [])),
                            len(rec.get("pitch_script", "")),
                        )
                        saved_search["rerank_state"] = "complete"
                        st.session_state.story_search = saved_search
                        st.session_state.pop("recommendation_audio", None)
                    st.rerun()
                except Exception as exc:
                    logger.warning(
                        "[RAG] rerank FAILED for query=%r: %s: %s",
                        saved_search.get("query", ""),
                        type(exc).__name__,
                        exc,
                    )
                    saved_search["rerank_state"] = "failed"
                    st.session_state.story_search = saved_search
                    _show_operation_error("Personalized pitch", exc)
            elif recommendation is None and saved_search.get("rerank_state") == "failed":
                st.warning("The semantic matches are available, but the personalized pitch could not be generated.")
        else:
            st.info("No stored story summaries matched those keywords. Try a different description.")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="divider"></div>
    <div style="text-align:center; color:var(--ink-dim); font-size:0.85rem; padding:1rem 0 2rem 0;">
      Hackey Specter · built in 6 hours · mood-first entertainment discovery
    </div>
    """,
    unsafe_allow_html=True,
)
