# Moodio on Databricks Apps

This app is configured to deploy on **Databricks Apps** with the included
`app.yaml` manifest.

## What `app.yaml` does

- **`command`**: launches Streamlit on the port Databricks injects
  (`$DATABRICKS_APP_PORT`) bound to all interfaces.
  The `DATABRICKS_APP_PORT` substitution is the only env-var expansion
  Databricks performs inside `command` (per the
  [Databricks app runtime docs](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/app-runtime)).
- **`env`**: disables Streamlit's telemetry and sets a light theme so
  the app matches the in-code palette.

## Deploy

### Option A — Databricks CLI

```bash
# from the project root
databricks apps deploy moodio --source .
```

### Option B — Databricks UI

1. In your workspace, go to **Apps** → **Create app**.
2. Choose **Custom app** (not a template) and point it at this folder.
3. The UI reads `app.yaml` and `requirements.txt` from the project root.

## Requirements file

Dependencies live in `requirements.txt` and are installed at build time:

```
streamlit>=1.31.0
streamlit-extras>=0.3.0
watchdog>=4.0.0
```

No external services (no LLM, no database) — the app runs entirely off
the curated catalog in `data/content.py`. If you later add an LLM
provider, put the key in a Databricks secret and reference it from
`app.yaml` with `valueFrom: <secret-name>`.

## Verifying locally that the manifest is correct

```bash
# quick sanity check
python3 -c "import yaml; yaml.safe_load(open('app.yaml'))" && echo "app.yaml is valid YAML"

# simulate the command locally (substitute 8000 for DATABRICKS_APP_PORT)
DATABRICKS_APP_PORT=8000 streamlit run app.py \
  --server.port=$DATABRICKS_APP_PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
```

## Port handling

Streamlit's default is `8501`. Databricks Apps expects the process to
listen on `$DATABRICKS_APP_PORT` (typically `8000`). The `--server.port`
flag in the command line handles that automatically — no code change
needed.
