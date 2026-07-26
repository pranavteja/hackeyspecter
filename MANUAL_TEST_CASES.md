# Manual Test Cases: Real-Time Story RAG

## Before testing

1. Put a valid `OPENAI_API_KEY` in `.env`.
2. Ensure `ultimate_pocketfm_vector_store.npz` exists. Build it once from the real embedded dataset:

   ```powershell
   python -c "from preprocess_stories import import_precomputed_embeddings; import_precomputed_embeddings(r'C:\Users\sumit\Downloads\ultimate_pocketfm_master_dataset_with_embeddings.json', 'ultimate_pocketfm_vector_store.npz')"
   ```

3. Start the app with `streamlit run app.py` and open **AI Story Search**.

## Direct vector-search baseline

The following values were generated from
`ultimate_pocketfm_vector_store.npz` using `vector_search()` only. They do not
include LLM reranking. Minor floating-point variation is acceptable, but the
title order should match when the same `text-embedding-3-large` query model and
saved vector store are used.

| Input | Expected direct top 5 (title — cosine similarity) |
|---|---|
| `I want something that feels like a rainy Sunday after heartbreak.` | 1. **Love Falls Like Rain** — `0.307989`<br>2. **Ordinary Days Of Taking Shelter From the Rain With A Gatekeeper Girl** — `0.267458`<br>3. **High Fidelity** — `0.264653`<br>4. **A Single Man** — `0.260843`<br>5. **Her Bucket List** — `0.259515` |
| `Give me a mysterious story in an unusual world, full of wonder and strange discoveries.` | 1. **The Knight** — `0.455892`<br>2. **Kinsmen of the Dragon** — `0.431306`<br>3. **Out of This World Watt-Evans** — `0.430502`<br>4. **In the Empire of Shadow** — `0.429753`<br>5. **The Debt Collector** — `0.421285` |
| `I need a funny, light escape after a difficult day.` | 1. **Snailogy** — `0.346608`<br>2. **Small World** — `0.322314`<br>3. **Brutally Honest** — `0.307408`<br>4. **New Life Project** — `0.301497`<br>5. **Illustrated Internet** — `0.293526` |
| `I want a tense story about survival in isolation.` | 1. **We Who Are About To...** — `0.474957`<br>2. **The Unforgiving Wind** — `0.452985`<br>3. **Island** — `0.440436`<br>4. **Deathwatch** — `0.425731`<br>5. **Polymath** — `0.424806` |
| `A hopeful book about rebuilding your life after loss.` | 1. **The Orphaned Anything's** — `0.441763`<br>2. **Miracle** — `0.422219`<br>3. **Pregnancy after a loss** — `0.420914`<br>4. **Hope Was Here** — `0.408203`<br>5. **Calling The Swan** — `0.404224` |

The app shows the direct top-five retrieval results first, then automatically
invokes the LLM reranker in a separate personalization step. Its final single
recommendation can differ from the first vector result because it considers the
candidate summaries, similarity signals, and any available extracted features.

| # | Input to paste | Expected output |
|---|---|---|
| 1 | `I want something that feels like a rainy Sunday after heartbreak.` | A brief retrieval spinner, then **Best semantic match:** and five or fewer semantic-match cards. A separate personalization spinner follows automatically, then a two-sentence recommendation and match reasons appear. The chosen story should feel reflective, melancholy, romantic, or hopeful. |
| 2 | `Give me a mysterious story in an unusual world, full of wonder and strange discoveries.` | An immediate semantic match with themes of mystery, fantasy/speculation, exploration, or wonder. Results must show a real title and summary from the imported master dataset. |
| 3 | `I need a funny, light escape after a difficult day.` | A lighter or comic semantic match, followed automatically by a pitch that explains the emotional fit in exactly two sentences. |
| 4 | `I want a tense story about survival in isolation.` | A suspenseful, adventurous, or survival-oriented match. Each returned match has a numeric similarity score. |
| 5 | `A hopeful book about rebuilding your life after loss.` | The automatic pitch references recovery, hope, resilience, or a related emotional arc. |
| 6 | *(leave the input empty and click Search stories)* | An error explaining that a story, mood, or theme is required. No recommendation cards should replace the previous successful search. |
| 7 | Temporarily rename `ultimate_pocketfm_vector_store.npz`, then search for `a quiet historical romance` | An error saying the offline vector store is missing and must be created with `import_precomputed_embeddings`. The app must **not** rebuild embeddings automatically. Restore the filename after this test. |
| 8 | Search the same prompt twice: `a quiet historical romance` | Both searches complete without rereading the master JSON; the second search reuses the session retrieval and rerank caches. |

## What to verify in a successful result

- The first spinner covers query embedding and local retrieval only; the top semantic cards appear before the personalization spinner.
- The automatic personalization spinner produces `recommended_book_title`, `audio_url`, `pitch_script`, and `emotional_match_reasons`.
- The displayed matches originate from the offline `.npz` vector database, not from `data/content.py`.
- The raw source summary displayed in each card belongs to the selected real story.
- Click **Play story summary with OpenAI** after a recommendation. For the supplied
  LibriVox ZIP links, the app should show an audio player containing an
  OpenAI-generated narration of the recommended story summary. If a future
  record supplies a direct MP3/WAV/M4A/OGG URL, **Play recommendation**
  should play that source audio instead.
- Click the microphone icon, record `I want a hopeful adventure with mystery`,
  then finish the recording using the recorder's stop control. The app should
  automatically transcribe it, insert the transcript into the text box, and
  launch the same search and personalization flow without requiring a button
  click.
- No OpenAI key is shown in the app, logs, source code, or Git-tracked files.
