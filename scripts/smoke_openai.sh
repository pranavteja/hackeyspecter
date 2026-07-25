#!/usr/bin/env bash
# Smoke test for the OpenAI-powered `explain_llm` in Hackey Specter.
#
# Run this in a shell where OPENAI_API_KEY is already set:
#     bash scripts/smoke_openai.sh
#
# It writes a timestamped log to scripts/smoke_openai.log and also
# prints a short summary to stdout. Share the log file back for analysis.

set -u  # fail on unset vars; do NOT use -e (we want to keep going on errors)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOG="$HERE/smoke_openai.log"

# Clear any previous log
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

cd "$ROOT"

log "=== Hackey Specter OpenAI smoke test ==="
log "run time:    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
log "script:      scripts/smoke_openai.sh"
log "repo root:   $ROOT"
log "python:      .venv/bin/python"
log ""

# ---- 1. Environment diagnostics (no key value leaked) ----
log "--- 1. Environment ---"
if [ -z "${OPENAI_API_KEY:-}" ]; then
    log "OPENAI_API_KEY: NOT SET"
    log "Aborting: cannot test LLM path without a key."
    exit 2
fi
log "OPENAI_API_KEY: set (len=${#OPENAI_API_KEY}, prefix=${OPENAI_API_KEY:0:6}...)"
log "OPENAI_MODEL:    ${OPENAI_MODEL:-<unset -> defaults to gpt-4o-mini>}"
log "DATABRICKS_APP_PORT: ${DATABRICKS_APP_PORT:-<unset>}"
log ""

# ---- 2. Package import check ----
log "--- 2. Package import ---"
if .venv/bin/python -c "import openai" 2>&1 | tee -a "$LOG"; then
    .venv/bin/python -c "import openai; print('openai version:', openai.__version__)" | tee -a "$LOG"
else
    log "openai package NOT importable. Install with: uv pip install -r requirements.txt"
    exit 3
fi
log ""

# ---- 3. Client init ----
log "--- 3. OpenAI client init ---"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import os
from mood_engine.engine import _get_openai_client
c = _get_openai_client()
print("client initialized:", c is not None)
if c is not None:
    print("client type:", type(c).__name__)
PY
log ""

# ---- 4. Live explain_llm calls across several items/moods ----
log "--- 4. Live explain_llm calls ---"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json
import os
import time
import traceback

from mood_engine import explain_llm, explain, all_items, item_by_id, MOOD_AXES
from data.content import CONTENT

def mood_from(**kw):
    m = {ax: 0.5 for ax in MOOD_AXES}
    m.update(kw)
    return m

# A spread of items + target moods to exercise different content types
# and different mood profiles.
CASES = [
    ("m_ameliemtl",  mood_from(valence=0.85, warmth=0.9, romance=0.7),  "whimsy/warmth"),
    ("m_eternal",    mood_from(melancholy=0.85, romance=0.8, depth=0.7), "heartbreak"),
    ("m_lostintrans",mood_from(nostalgia=0.8, melancholy=0.7, depth=0.8),"3am disconnection"),
    ("b_piranesi",   mood_from(mystery=0.85, wonder=0.8, depth=0.7),    "mystery/wonder"),
    ("b_stoner",     mood_from(melancholy=0.8, depth=0.85, tension=0.4), "quiet tragedy"),
    ("g_journey",    mood_from(wonder=0.9, hope=0.8, energy=0.5),        "awe"),
]

print(f"running {len(CASES)} cases\n")

results = []
for i, (item_id, mood, label) in enumerate(CASES, 1):
    item = item_by_id(item_id)
    if item is None:
        # fall back to first content item if id not found
        item = CONTENT[0]
        item_id = item["id"]
    print(f"--- case {i}/{len(CASES)}: {item_id} ({item['type']}) | mood={label} ---")
    t0 = time.time()
    try:
        out = explain_llm(item, mood)
        dt = time.time() - t0
        print(f"elapsed: {dt:.2f}s")
        print(f"chars:   {len(out)}")
        print(f"output:\n{out}\n")
        results.append({
            "case": i, "item_id": item_id, "type": item["type"],
            "mood_label": label, "ok": True, "elapsed_s": round(dt, 3),
            "chars": len(out), "output": out,
        })
    except Exception as e:
        dt = time.time() - t0
        print(f"elapsed: {dt:.2f}s")
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("")
        results.append({
            "case": i, "item_id": item_id, "type": item["type"],
            "mood_label": label, "ok": False, "elapsed_s": round(dt, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })

# Also compare LLM vs rule-based for the first case to confirm they differ
print("--- comparison: LLM vs rule-based (case 1) ---")
item = item_by_id(CASES[0][0]) or CONTENT[0]
mood = CASES[0][1]
llm_out = explain_llm(item, mood)
rule_out = explain(item, mood)
print(f"LLM  ({len(llm_out)} chars): {llm_out[:200]}")
print(f"RULE ({len(rule_out)} chars): {rule_out[:200]}")
print(f"differ: {llm_out != rule_out}")

# Dump structured results to a JSON sidecar for easier analysis
with open("scripts/smoke_openai_results.json", "w") as f:
    json.dump({
        "openai_version": __import__("openai").__version__,
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "results": results,
        "comparison_case1": {
            "llm": llm_out, "rule": rule_out, "differ": llm_out != rule_out,
        },
    }, f, indent=2)
print("\nstructured results written to scripts/smoke_openai_results.json")
PY
log ""

# ---- 5. Summary ----
log "--- 5. Summary ---"
if [ -f scripts/smoke_openai_results.json ]; then
    .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json
with open("scripts/smoke_openai_results.json") as f:
    d = json.load(f)
n = len(d["results"])
ok = sum(1 for r in d["results"] if r.get("ok"))
print(f"cases: {ok}/{n} succeeded")
for r in d["results"]:
    status = "OK " if r.get("ok") else "FAIL"
    extra = f"{r['chars']}c {r['elapsed_s']}s" if r.get("ok") else r.get("error","")
    print(f"  [{status}] case {r['case']}: {r['item_id']} ({r['type']}) | {r['mood_label']} | {extra}")
print(f"model: {d['model']}")
print(f"openai version: {d['openai_version']}")
print(f"LLM differs from rule-based (case 1): {d['comparison_case1']['differ']}")
PY
else
    log "results JSON not produced — see log above for errors"
fi
log ""
log "=== smoke test complete ==="
log "log file:      $LOG"
log "results json:  scripts/smoke_openai_results.json"
log ""
log "Share these two files back for analysis:"
log "  scripts/smoke_openai.log"
log "  scripts/smoke_openai_results.json"