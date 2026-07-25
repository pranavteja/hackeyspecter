# Hackey Specter — Movie catalog schema

Each movie in `data/content.py` has **10 metadata fields** and a
**12-axis mood vector** (the features used for matching).

> Note: `data/content.py`'s docstring says "10-dimensional mood vector"
> but the actual data has **12 axes** — that comment is stale.

## The 10 metadata fields per movie

| # | field | example (Amélie) |
|---|-------|------------------|
| 1 | `id` | `m_ameliemtl` |
| 2 | `title` | Amélie |
| 3 | `creator` | Jean-Pierre Jeunet |
| 4 | `year` | 2001 |
| 5 | `type` | movie |
| 6 | `runtime` | 122 (minutes) |
| 7 | `mood` | {valence, energy, …} — the 12-axis vector |
| 8 | `themes` | ["whimsy", "loneliness", "small joys", "Paris", "fate"] |
| 9 | `emotional_arc` | "Lonely daydreamer learns that small acts of kindness ripple into love." |
| 10 | `pitch` | "A shy Parisian waitress decides to secretly orchestrate the happiness of those around her." |

## The 12 mood-vector axes (the actual "features" used for recommendation)

All values are `0.0–1.0`:

| axis | 0 → 1 meaning |
|------|---------------|
| `valence` | sad → happy |
| `energy` | quiet → intense |
| `warmth` | cold/dark → warm/affectionate |
| `tension` | calm → suspenseful |
| `depth` | light/fun → philosophical |
| `romance` | none → romantic |
| `mystery` | none → mysterious |
| `nostalgia` | present → nostalgic |
| `hope` | bleak → uplifting |
| `melancholy` | upbeat → wistful |
| `wonder` | mundane → awe-inspiring |
| `humor` | serious → funny |

## How matching works

The recommender (`M.rank` in `mood_engine/engine.py`) does **cosine
similarity** between the user's parsed target mood vector and each
movie's `mood` vector across these 12 axes.

The `explain` / `explain_llm` functions then read the aligned axes +
themes + arc to write the "why you'll love this" paragraph.

## Catalog size

- **15 movies** currently in `data/content.py`
- Same schema applies to books, podcasts, and games (minus `runtime`
  for non-movie types where it isn't relevant).