"""
Hackey Specter — main Streamlit app.

A 6-hour hackathon prototype that recommends entertainment based on mood,
plans weekend itineraries across media, and links stories across formats.

Run: streamlit run app.py
"""
from __future__ import annotations

import random
import logging
from pathlib import Path

import streamlit as st

import mood_engine as M
from data.content import SAMPLE_PROMPTS
from recommend_and_pitch import generate_final_recommendation
from search_engine import StoryDatabase, load_database, vector_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

VECTOR_STORE_PATH = Path(__file__).resolve().with_name("stories_vector_store.npz")


@st.cache_resource(show_spinner=False)
def get_story_database() -> StoryDatabase:
    """Load the offline-built vector store; never rebuild embeddings during app use."""
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "The offline vector store is missing. Run process_and_embed_dataset "
            "once to create stories_vector_store.npz."
        )
    return load_database(str(VECTOR_STORE_PATH))

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
  border-radius: 14px;
  padding: 1.25rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow);
  transition: transform 0.15s ease, background 0.15s ease;
}
.card:hover {
  background: var(--bg-card-hover);
  transform: translateY(-1px);
}
.card .type-chip {
  display: inline-block;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--ink-soft);
  margin-bottom: 0.6rem;
}
.card .type-chip.movie   { color: var(--accent); border-color: rgba(184,122,122,0.35); }
.card .type-chip.book    { color: var(--accent-warm); border-color: rgba(184,128,64,0.35); }
.card .type-chip.podcast { color: var(--accent-cool); border-color: rgba(90,138,122,0.35); }
.card .type-chip.game    { color: #7b5fa6; border-color: rgba(123,95,166,0.30); }
.card h3 { margin: 0 0 0.2rem 0; font-size: 1.35rem; }
.card .creator { font-size: 0.85rem; color: var(--ink-dim); margin-bottom: 0.5rem; }
.card .pitch { font-size: 0.95rem; color: var(--ink); line-height: 1.5; }
.card .themes { margin-top: 0.5rem; font-size: 0.8rem; color: var(--ink-dim); }

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
if "weekend" not in st.session_state:
    st.session_state.weekend = None
if "festival" not in st.session_state:
    st.session_state.festival = None


# ============================================================
# HELPERS
# ============================================================
def render_card(item: dict, score: float | None = None) -> str:
    score_html = f'<span style="color:var(--ink-dim); font-size:0.8rem;"> · {score:.2f} match</span>' if score is not None else ""
    themes_html = ""
    if item.get("themes"):
        themes_html = f'<div class="themes">{" · ".join(item["themes"][:5])}</div>'
    return f"""
    <div class="card">
      <div class="type-chip {item['type']}">{M.media_type_emoji(item['type'])} {item['type']}</div>
      <h3>{item['title']}{score_html}</h3>
      <div class="creator">{item['creator']} · {item['year']}</div>
      <div class="pitch">{item.get('pitch', '')}</div>
      {themes_html}
    </div>
    """


def render_summary_card(item: dict, score: float) -> str:
    """Render a result from summary_1to16000.json without demo metadata."""
    summary = item.get("summary", "").strip()
    preview = summary[:700] + ("…" if len(summary) > 700 else "")
    link = item.get("librivox_project_url", "")
    link_html = f'<a href="{link}" target="_blank">Open on LibriVox</a>' if link else ""
    return f'''<div class="card">
      <div class="type-chip book">Story summary · {score:.2f} match</div>
      <h3>{item.get("title", "Untitled")}</h3>
      <div class="pitch">{preview}</div>
      {link_html}
    </div>'''


def render_why(item: dict, target: dict) -> str:
    return f"""
    <div class="why-box">
      <strong style="font-style: normal; color: var(--accent);">Why you will love this:</strong><br>
      {M.explain(item, target)}
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
tab_search, tab_concierge, tab_discover, tab_festival, tab_context = st.tabs([
    "🔍  Mood First Search",
    "🎯  Weekend Concierge",
    "🔗  Cross-Media Discovery",
    "🎬  Festivals",
    "🔎  AI Story Search",
])


# ============================================================
# TAB 1: MOOD FIRST SEARCH
# ============================================================
with tab_search:
    st.markdown('<div class="section-label">In your own words</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.text_input(
            "mood",
            placeholder="I want something that feels like a rainy Sunday after heartbreak…",
            label_visibility="collapsed",
            key="mood_input",
        )
    with col2:
        media_filter = st.selectbox(
            "media",
            ["everything", "movies", "books", "podcasts", "games"],
            label_visibility="collapsed",
            key="mood_media",
        )

    sample_clicked = st.session_state.pop("active_sample", None)
    cols = st.columns(3)
    for i, sample in enumerate(SAMPLE_PROMPTS[:6]):
        with cols[i % 3]:
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                st.session_state["active_sample"] = sample
                st.rerun()

    # the active prompt is either what the user typed, or the last
    # sample they clicked. The text_input shows the active value
    # when it's non-empty, but we don't try to mutate the widget
    # (Streamlit forbids that).
    active_prompt = prompt or sample_clicked or ""

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not active_prompt:
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 0 1rem 0; color:var(--ink-soft);">
              <p style="font-family:'Fraunces',serif; font-style:italic; font-size:1.15rem; max-width:540px; margin: 0 auto 0.5rem auto;">
                "I want something that feels like a rainy Sunday after heartbreak."
              </p>
              <p style="font-size:0.85rem; color:var(--ink-dim);">
                Type a feeling, pick a sample above, and the mood engine will translate it into picks across movies, books, podcasts, and games.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if active_prompt:
        target = M.parse_mood(active_prompt)
        types_map = {
            "everything": None, "movies": ["movie"], "books": ["book"],
            "podcasts": ["podcast"], "games": ["game"],
        }
        types = types_map[media_filter]

        st.markdown(f"**You said:** *\"{active_prompt}\"*")
        st.markdown(render_mood_summary(target), unsafe_allow_html=True)
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        results = M.rank(target=target, k=5, types=types)
        if not results:
            st.warning("Nothing in the library for that yet — try a different feeling.")
        else:
            for idx, (item, score) in enumerate(results):
                with st.container():
                    st.markdown(render_card(item, score), unsafe_allow_html=True)
                    # auto-expand the "why" for the top recommendation
                    with st.expander("Why will I love this?", expanded=(idx == 0)):
                        st.markdown(render_why(item, target), unsafe_allow_html=True)
                    c1, c2, _ = st.columns([1, 1, 4])
                    with c1:
                        if st.button("🎬 I watched/finished this", key=f"finish_{item['id']}"):
                            if item["id"] not in st.session_state.finished:
                                st.session_state.finished.append(item["id"])
                            st.toast(f"Added **{item['title']}** to your finished shelf", icon="🕯️")
                    with c2:
                        if st.button("🔗 See what else fits", key=f"goto_{item['id']}"):
                            st.session_state["discover_seed"] = item["id"]
                            st.toast("Open the **Cross-Media Discovery** tab →", icon="🔗")

        # update history
        if results:
            st.session_state.history.insert(0, (active_prompt, target, results[0][0]))
            st.session_state.history = st.session_state.history[:6]


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

    weekend_sample = st.session_state.pop("active_weekend_sample", None)
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        plan_clicked = st.button("Plan my weekend", type="primary", use_container_width=True)
    with c2:
        if st.button("Try a sample", use_container_width=True):
            st.session_state["active_weekend_sample"] = random.choice(SAMPLE_PROMPTS)
            st.rerun()

    active_weekend_prompt = concierge_prompt or weekend_sample or ""
    if plan_clicked and active_weekend_prompt:
        target = M.parse_mood(active_weekend_prompt)
        itinerary = M.plan_weekend(target)
        st.session_state.weekend = {
            "prompt": active_weekend_prompt,
            "mood": target,
            "itinerary": itinerary,
        }
        # remember finished items
        for entry in itinerary:
            if entry["item"]["id"] not in st.session_state.finished:
                st.session_state.finished.append(entry["item"]["id"])

    if st.session_state.weekend:
        wk = st.session_state.weekend
        st.markdown(
            f"""
            <div class="festival-hero" style="margin-top:1rem;">
              <h2>Your Weekend</h2>
              <div class="tagline">Built around: "{wk['prompt']}"</div>
              <div style="margin-top:0.8rem;">{render_mood_summary(wk['mood'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for idx, entry in enumerate(wk["itinerary"]):
            item = entry["item"]
            st.markdown(
                f"""
                <div class="itinerary-slot">
                  <div class="slot-emoji">{M.media_type_emoji(item['type'])}</div>
                  <div class="slot-label">{entry['slot']}</div>
                  <div class="slot-item">
                    <h4>{item['title']}</h4>
                    <div class="meta">{item['creator']} · {item['year']} · {item['type']} · {entry['score']} match</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Why this slot?", expanded=(idx == 0)):
                st.markdown(render_why(item, wk["mood"]), unsafe_allow_html=True)


# ============================================================
# TAB 3: CROSS-MEDIA DISCOVERY
# ============================================================
with tab_discover:
    st.markdown('<div class="section-label">Finished something? Find its emotional cousins across every medium.</div>', unsafe_allow_html=True)

    finished_items_raw: list[dict | None] = [M.item_by_id(i) for i in st.session_state.finished]
    finished_items: list[dict] = [i for i in finished_items_raw if i is not None]
    first_id: str = finished_items[0]["id"] if finished_items else "m_eternal"
    default_seed: str = st.session_state.get("discover_seed") or first_id

    # If user has a finished shelf, show a quick view of it
    if finished_items:
        st.markdown("**Your finished shelf:**")
        chips = "".join(
            f'<span class="mood-pill">{M.media_type_emoji(i["type"])} {i["title"]}</span>'
            for i in finished_items[-8:]
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    all_ids = [i["id"] for i in M.all_items()]
    default_idx = all_ids.index(default_seed) if default_seed in all_ids else 0

    def _fmt(x: str) -> str:
        i = M.item_by_id(x)
        if i is None:
            return x
        return f'{M.media_type_emoji(i["type"])} {i["title"]} ({i["year"]})'

    seed_id = st.selectbox(
        "seed_item",
        options=all_ids,
        index=default_idx,
        format_func=_fmt,
        label_visibility="collapsed",
        key="discover_seed_select",
    )

    seed = M.item_by_id(seed_id)
    if seed:
        related = M.unlock_related(seed_id)
        st.markdown(
            f"""
            <div class="festival-hero" style="margin-top:1rem;">
              <div class="type-chip {seed['type']}" style="display:inline-block;">{M.media_type_emoji(seed['type'])} {seed['type']}</div>
              <h2>{seed['title']}</h2>
              <div class="tagline">{seed.get('pitch','')}</div>
              <div class="window">{seed['emotional_arc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Cross-media emotional cousins")
        st.caption("Each recommendation below shares the emotional fingerprint of what you just finished.")
        for r in related:
            item = r["item"]
            st.markdown(render_card(item, r["score"]), unsafe_allow_html=True)
            if r["shared_themes"]:
                st.markdown(
                    f'<div style="margin-top:-0.5rem; margin-bottom:0.5rem;">'
                    f'<span class="mood-pill" style="font-size:0.7rem;">shared themes: {", ".join(r["shared_themes"])}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("Why this matches", expanded=False):
                st.markdown(render_why(item, seed["mood"]), unsafe_allow_html=True)


# ============================================================
# TAB 4: FESTIVALS
# ============================================================
with tab_festival:
    st.markdown('<div class="section-label">AI-curated festivals, refreshed every week</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 2])
    with c1:
        fest_prompt = st.text_input(
            "fest_mood",
            placeholder="Tune the festival to a mood (optional)…",
            label_visibility="collapsed",
            key="fest_input",
        )
    with c2:
        fest_choices = [t["name"] for t in M.FESTIVAL_TEMPLATES]
        default_fest_idx = 0
        if st.session_state.get("fest_template_select") in fest_choices:
            default_fest_idx = fest_choices.index(st.session_state["fest_template_select"])
        fest_choice = st.selectbox(
            "festival_template",
            options=fest_choices,
            index=default_fest_idx,
            label_visibility="collapsed",
            key="fest_template_select_widget",
        )

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        gen_clicked = st.button("Generate this weekend's festival", type="primary", use_container_width=True)
    with c2:
        if st.button("Surprise me", use_container_width=True):
            st.session_state["fest_template_select"] = random.choice(fest_choices)
            st.rerun()

    if gen_clicked:
        template = next(t for t in M.FESTIVAL_TEMPLATES if t["name"] == fest_choice)
        user_mood = M.parse_mood(fest_prompt) if fest_prompt else None
        st.session_state.festival = M.generate_festival(template, user_mood)

    if st.session_state.festival:
        f = st.session_state.festival
        st.markdown(
            f"""
            <div class="festival-hero" style="margin-top:1rem;">
              <div class="window">Showing {f['starts']} → {f['ends']}</div>
              <h2>{f['name']}</h2>
              <div class="tagline">{f['tagline']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### The lineup")
        template = next(t for t in M.FESTIVAL_TEMPLATES if t["name"] == f["name"])
        for idx, entry in enumerate(f["lineup"]):
            item = entry["item"]
            st.markdown(render_card(item, entry["score"]), unsafe_allow_html=True)
            with st.expander("Why this one?", expanded=(idx == 0)):
                st.markdown(render_why(item, template["mood_bias"]), unsafe_allow_html=True)


# ============================================================
# TAB 5: AI STORY SEARCH
# ============================================================
with tab_context:
    st.markdown('<div class="section-label">Semantic story search</div>', unsafe_allow_html=True)
    st.caption("The story database is embedded offline once. This search embeds only your request, then reranks the closest saved vectors.")
    story_request = st.text_input("story_search", placeholder="A hopeful story about rebuilding after a breakup", label_visibility="collapsed")
    if st.button("Search with AI", type="primary"):
        try:
            with st.spinner("Embedding your request, searching saved vectors, and choosing the best match…"):
                candidates = vector_search(story_request, get_story_database(), top_k=5)
                recommendation = generate_final_recommendation(story_request, candidates)
                st.session_state.story_search = (recommendation, candidates)
        except Exception as exc:
            st.error(f"Search could not run: {exc}")

    saved_search = st.session_state.get("story_search")
    if saved_search:
        recommendation, results = saved_search
        st.markdown(f"### Recommended: {recommendation['recommended_book_title']}")
        st.markdown(recommendation["pitch_script"])
        if recommendation["emotional_match_reasons"]:
            st.markdown("**Why it fits:** " + " · ".join(recommendation["emotional_match_reasons"]))
        if recommendation["audio_url"]:
            st.markdown(f"[Listen on LibriVox]({recommendation['audio_url']})")
        st.markdown("### Closest semantic matches")
        if results:
            for item in results:
                st.markdown(
                    render_summary_card(item, item.get("vector_similarity", 0.0)),
                    unsafe_allow_html=True,
                )
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
