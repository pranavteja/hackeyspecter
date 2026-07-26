"""End-to-end smoke test of the RAG recommendation pipeline."""
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path so we can import rag_config, search_engine, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Force loader to pick up .env
os.environ.pop("OPENAI_API_KEY", None)
import rag_config  # noqa: E402

# Import the actual UI-facing functions
from search_engine import vector_search, load_database  # noqa: E402
from recommend_and_pitch import generate_final_recommendation  # noqa: E402

QUERY = "a quiet story about grief and small kindnesses"

print("=" * 70)
print("STEP 1: Load database")
print("=" * 70)
t0 = time.time()
db = load_database("stories_vector_store.npz")
print(f"  records: {len(db.records)}")
print(f"  embeddings shape: {db.embeddings.shape}")
print(f"  loaded in {time.time() - t0:.2f}s")

print()
print("=" * 70)
print("STEP 2: Vector search")
print("=" * 70)
t0 = time.time()
hits = vector_search(QUERY, db, top_k=5)
print(f"  search took {time.time() - t0:.2f}s")
for i, h in enumerate(hits, 1):
    print(f"  {i}. sim={h.get('vector_similarity', 0):.3f}  {h.get('title', '?')[:60]}")

print()
print("=" * 70)
print("STEP 3: LLM rerank + pitch")
print("=" * 70)
t0 = time.time()
try:
    rec = generate_final_recommendation(QUERY, hits)
    print(f"  rerank took {time.time() - t0:.2f}s")
    print(f"  result keys: {list(rec.keys()) if isinstance(rec, dict) else type(rec).__name__}")
    if isinstance(rec, dict):
        print(f"  title:    {rec.get('recommended_book_title', '?')}")
        print(f"  record:   {rec.get('recommended_record_id', '?')}")
        print(f"  audio:    {(rec.get('audio_url') or 'none')[:80]}")
        pitch = rec.get("pitch_script", "")
        print(f"  pitch:    {pitch[:200]}{'...' if len(pitch) > 200 else ''}")
        reasons = rec.get("emotional_match_reasons", [])
        print(f"  reasons:  {len(reasons)} items")
        for r in reasons:
            print(f"             - {r[:120]}")
    print()
    print("RESULT: API IS WORKING")
except Exception as e:
    print(f"  FAILED in {time.time() - t0:.2f}s")
    print(f"  exception type: {type(e).__name__}")
    print(f"  message: {e}")
    print()
    print("TRACEBACK:")
    traceback.print_exc()
    print()
    print("RESULT: API FAILED")
    sys.exit(1)
