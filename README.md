# Hackey Specter 🕯️


python -c "from preprocess_stories import process_and_embed_dataset; process_and_embed_dataset('summary_1to16000.json', 'stories_vector_store.npz')"


> Tell us how you want to feel. We'll find the movie, book, podcast, or game
> that gets you there — and explain why it'll be the one that stays with you.

A 6-hour hackathon prototype for mood-first entertainment discovery.
Built in **Python + Streamlit**, deploys to **Databricks Apps** with no external API keys.

## What it does

| Feature | What happens |
|---|---|
| **🔍 Mood First Search** | Type a feeling in plain language ("a rainy Sunday after heartbreak"), get 5 ranked picks across media. |
| **🎯 AI Entertainment Concierge** | Get a 7-slot weekend plan (Fri-night → Sun-night) spanning movies, books, games, and podcasts. |
| **🔗 Cross-Media Discovery** | Finished a movie? Get 6 emotional cousins — books, games, and podcasts with the same fingerprint. |
| **💡 Explain Why I Will Love This** | Every pick comes with a generated explanation of which emotional axes line up with your mood. |
| **🎬 AI Curated Festivals** | 5 ready-to-go festival templates (Lonely Sundays, Wonder Engine, Big Hug, Knife-Edge, Late-Night Longing) that auto-build a 5-item lineup from the catalog. |
| **🔎 AI Story Search** | Embeds a natural-language request once, returns saved semantic matches immediately, and generates an optional personalized GPT pitch. |

## Architecture

```
hackeyspecter/
├── app.py                 # Streamlit UI — single page, 5 tabs
├── mood_engine/           # the "AI" — local, deterministic, LLM-swappable
│   ├── engine.py          #   parse_mood, rank, explain, plan_weekend,
│   │                      #   unlock_related, generate_festival
│   ├── check_intent.py    #   OpenAI intent extraction + local story-summary search
│   └── __init__.py
├── data/
│   └── content.py         # curated 42-item library + mood keyword map
├── requirements.txt
├── app.yaml               # Databricks Apps deployment config
└── README.md
```

**The "AI"** is a deterministic local engine — no API key needed.
Each piece of content is tagged with a 12-dimensional mood vector
(`valence, energy, warmth, tension, depth, romance, mystery, nostalgia, hope, melancholy, wonder, humor`).
Free-text prompts are tokenized and matched against a curated keyword map
that adjusts those axes; results are ranked by cosine similarity.
The "explain why" generator reads the alignment between your target mood
and each pick's mood vector and composes a paragraph from the strongest
matching axes plus the item's themes and emotional arc.

The architecture is **LLM-swappable**: each function takes a string
or mood vector and returns a dict. Drop in an OpenAI/Anthropic call
later and you keep the UI.

### AI story-search contract

The live story search has two deliberately separate stages. `vector_search()`
embeds only the user's query, performs exact NumPy cosine ranking against the
saved story vectors, and returns the closest five records immediately.
`generate_final_recommendation()` is invoked only when the user asks for a
personalized pitch; it reranks those five candidates and returns
`recommended_record_id`, `recommended_book_title`, `audio_url`,
`pitch_script`, and `emotional_match_reasons`. Set `OPENAI_API_KEY` in `.env`
or the deployment environment before use.

### RAG data lifecycle

Run `process_and_embed_dataset("summary_1to16000.json",
"stories_vector_store.npz")` once whenever the source summaries change. It
creates the saved feature and vector database. The running app never embeds the
full JSON file: it loads `stories_vector_store.npz`, embeds only the user's
query, and performs NumPy cosine search. The top five matches render before
the optional reranker is asked to write a pitch.

The default quality profile uses `gpt-5.6-sol` for offline feature extraction,
`gpt-5.6-terra` with low reasoning for live reranking, and
`text-embedding-3-large` at 3072 dimensions. After changing to this profile,
run preprocessing once to rebuild the existing legacy 1536-dimensional store.
Later runs reuse unchanged records and vectors, and a fully unchanged dataset
performs no OpenAI API work.

## Run locally

```bash
# with uv (recommended)
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py

# or with plain python
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open <http://127.0.0.1:8501>.

## Deploy to Databricks Apps

The included `app.yaml` is the Databricks Apps spec — it runs Streamlit
on port 8000, the port Databricks Apps expects.

```bash
databricks apps deploy hackey-specter --source .
```

(or use the Databricks UI → **Apps** → **Create app** → point at this folder.)

## Tech choices (and why)

| Choice | Why |
|---|---|
| **Streamlit** | Fastest path from idea to demo'd web app in Python. |
| **Local mood engine** | No API key required for the demo. Whole 42-item library ranks in <50ms. |
| **12-dim mood vector** | Captures nuance that single-tag genre/keyword can't. Cosine similarity gives meaningful rankings. |
| **4 tabs, not 4 pages** | Keeps the demo flow in one URL — easier to show, easier to deploy. |
| **Curated 42-item library** | Quality over quantity for the demo. Every item has a real pitch and emotional arc. |

## Stretch ideas (if there's more time)

- Real LLM behind the mood parser for open-ended vocabulary
- User accounts, watch-history, like/dislike signals
- Spotify / TMDB / OpenLibrary API integration for the catalog
- Vector DB (chroma/qdrant) for embedding-based retrieval
- Festival sharing — generate a link someone else can open
