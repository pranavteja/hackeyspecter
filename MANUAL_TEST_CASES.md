# Manual Test Cases: Real-Time Story RAG

## Before testing

1. Put a valid `OPENAI_API_KEY` in `.env`.
2. Ensure `stories_vector_store.npz` exists. Build it once from the real data:

   ```powershell
   python -c "from preprocess_stories import process_and_embed_dataset; process_and_embed_dataset('summary_1to16000.json', 'stories_vector_store.npz')"
   ```

3. Start the app with `streamlit run app.py` and open **AI Story Search**.

## Direct vector-search baseline

The following values were generated from the current `stories_vector_store.npz`
using `vector_search()` only. They do not include LLM reranking. Minor floating
point variation is acceptable, but the title order should match when the same
embedding model and saved vector store are used.

| Input | Expected direct top 5 (title — cosine similarity) |
|---|---|
| `I want something that feels like a rainy Sunday after heartbreak.` | 1. **What Dreams May Come** — `0.315578`<br>2. **One Day** — `0.308283`<br>3. **Second Chance** — `0.272324`<br>4. **Imre: A Memorandum** — `0.271962`<br>5. **Breaking Point** — `0.270858` |
| `Give me a mysterious story in an unusual world, full of wonder and strange discoveries.` | 1. **Widdershins** — `0.429918`<br>2. **Labyrinth** — `0.423269`<br>3. **Mystery** — `0.411811`<br>4. **Brood of the Witch Queen** — `0.404842`<br>5. **Pellucidar** — `0.396235` |
| `I need a funny, light escape after a difficult day.` | 1. **Romance** — `0.305772`<br>2. **One Day** — `0.300782`<br>3. **Love Among the Chickens** — `0.256888`<br>4. **Kai Lung's Golden Hours** — `0.252045`<br>5. **Psmith in the City** — `0.246890` |
| `I want a tense story about survival in isolation.` | 1. **Highways in Hiding** — `0.379378`<br>2. **Stone** — `0.360879`<br>3. **Kazan** — `0.357711`<br>4. **Fugitive Pieces** — `0.354044`<br>5. **Labyrinth** — `0.346725` |
| `A hopeful book about rebuilding your life after loss.` | 1. **What Dreams May Come** — `0.402694`<br>2. **Second Chance** — `0.382179`<br>3. **Sisters** — `0.364483`<br>4. **Dr. Heidenhoff's Process** — `0.362253`<br>5. **Disguise** — `0.343974` |

The app uses the same direct top-five retrieval results before it sends them to
the LLM reranker. The final single recommendation can differ from the first
vector result because reranking considers the extracted emotional features.

| # | Input to paste | Expected output |
|---|---|---|
| 1 | `I want something that feels like a rainy Sunday after heartbreak.` | Spinner appears, then one **Recommended:** title, a two-sentence pitch, one or more match reasons, an optional LibriVox link, and five or fewer semantic-match cards. The chosen story should feel reflective, melancholy, romantic, or hopeful. |
| 2 | `Give me a mysterious story in an unusual world, full of wonder and strange discoveries.` | A recommendation with themes of mystery, fantasy/speculation, exploration, or wonder. Results must show a real title and summary from `summary_1to16000.json`. |
| 3 | `I need a funny, light escape after a difficult day.` | A lighter or comic recommendation. The pitch must explain the emotional fit and remain exactly two sentences. |
| 4 | `I want a tense story about survival in isolation.` | A suspenseful, adventurous, or survival-oriented recommendation. Each returned match has a numeric similarity score. |
| 5 | `A hopeful book about rebuilding your life after loss.` | A recommendation whose pitch references recovery, hope, resilience, or a related emotional arc. |
| 6 | *(leave the input empty and click Search with AI)* | An error explaining that `user_prompt` cannot be empty. No recommendation cards should replace the previous successful search. |
| 7 | Temporarily rename `stories_vector_store.npz`, then search for `a quiet historical romance` | An error saying the offline vector store is missing and must be created with `process_and_embed_dataset`. The app must **not** rebuild embeddings automatically. Restore the filename after this test. |
| 8 | Search the same prompt twice: `a quiet historical romance` | Both searches complete without reprocessing `summary_1to16000.json`. The app should only embed the user query and rerank the retrieved candidates. Titles may differ slightly between runs. |

## What to verify in a successful result

- The app shows the loading spinner while API calls run.
- A recommendation contains `recommended_book_title`, `audio_url`, `pitch_script`, and `emotional_match_reasons`.
- The displayed matches originate from the offline `.npz` vector database, not from `data/content.py`.
- The raw source summary displayed in each card belongs to the selected real story.
- Click **▶ Play story summary with OpenAI** after a recommendation. For the supplied
  LibriVox ZIP links, the app should show an audio player containing an
  OpenAI-generated narration of the recommended story summary. If a future
  record supplies a direct MP3/WAV/M4A/OGG URL, **▶ Play recommendation**
  should play that source audio instead.
- Click the microphone icon, record `I want a hopeful adventure with mystery`,
  then finish the recording using the recorder's stop control. The app should
  automatically transcribe it and show the voice input; clicking **Search with
  AI** should use it when the typed input is empty.
- No OpenAI key is shown in the app, logs, source code, or Git-tracked files.
