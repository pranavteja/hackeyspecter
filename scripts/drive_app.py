"""Drive the Streamlit app via AppTest to capture logs from a real Tab 5 submission."""
import io
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

# Capture all log records (the app's basicConfig writes to stderr; we also tee to a buffer)
log_buffer = io.StringIO()
handler = logging.StreamHandler(log_buffer)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

from streamlit.testing.v1 import AppTest

QUERY = "Crime and Punishment"

print("=" * 70)
print(f"AppTest run with query: {QUERY!r}")
print("=" * 70)

at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
at.run()

# Skip tab switching — AppTest exposes widgets across all tabs by key.
# Set the text area value via session_state, then click the form button.
at.session_state["story_search_input"] = QUERY
print(f"[driver] set session_state['story_search_input'] = {QUERY!r}")
at.run()

btns = [b for b in at.button if b.label == "Search stories"]
print(f"[driver] found {len(btns)} 'Search stories' button(s)")
if btns:
    btns[0].click()
    print("[driver] clicked Search stories")

def ss_get(key, default=None):
    """SafeSessionState has no .get(); use try/except."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default

# First run after click: vector search, sets rerank_state='pending', then st.rerun()
at.run()
ss = ss_get("story_search", {}) or {}
print(f"[driver] after 1st run: rerank_state={ss.get('rerank_state')!r}")

# Subsequent runs: LLM rerank spinner, then 'complete', then st.rerun()
for i in range(2, 6):
    at.run()
    ss = ss_get("story_search", {}) or {}
    print(f"[driver] after run {i}: rerank_state={ss.get('rerank_state')!r}")

# Print session_state summary
ss = ss_get("story_search", {}) or {}
candidates = ss.get("candidates", []) or []
recommendation = ss.get("recommendation")
print()
print("=" * 70)
print("SESSION STATE SUMMARY")
print("=" * 70)
print(f"query:          {ss.get('query')!r}")
print(f"candidates ({len(candidates)}):")
for i, c in enumerate(candidates, 1):
    print(f"  {i}. sim={float(c.get('vector_similarity', 0)):.3f}  {c.get('title')}")
if isinstance(recommendation, dict):
    print(f"recommendation: {recommendation.get('recommended_book_title')!r}")
    print(f"  record_id:    {recommendation.get('recommended_record_id')}")
    print(f"  pitch[:120]:  {str(recommendation.get('pitch_script', ''))[:120]!r}")
    print(f"  reasons:      {len(recommendation.get('emotional_match_reasons', []))}")
else:
    print("recommendation: None (fallback path)")

# Dump captured logs
print()
print("=" * 70)
print("CAPTURED LOGS (from app's logger)")
print("=" * 70)
log_text = log_buffer.getvalue()
print(log_text if log_text else "(no log records captured)")

# Also report any exceptions raised during the run
if at.exception:
    print()
    print("=" * 70)
    print("EXCEPTIONS RAISED BY APP")
    print("=" * 70)
    for ex in at.exception:
        print(f"  - {ex.value}")
