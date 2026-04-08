# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

GTM Diligence Assistant is a local-first LangGraph application for answering finance diligence questions against a mounted dataroom of PDF filings. The active product surface is the workflow, the demo web app, the local JSONL dataset, and the LangSmith evaluation path.

## Architecture

- **Workflow**: explicit LangGraph graph with typed state and bounded retrieval/retry loops
- **Retrieval**: deterministic PDF tools plus page-window vector retrieval per company/FY folder
- **Web App**: FastAPI backend serving API routes and built frontend assets
- **Evaluation**: LangSmith experiments over the curated JSONL dataset

### Key Components

- `gtm_diligence_assistant/workflow.py` - state model, nodes, and graph edges
- `gtm_diligence_assistant/tools.py` - local PDF scanning and reading helpers
- `gtm_diligence_assistant/vector_index.py` - vector index build/load logic
- `gtm_diligence_assistant/web_app.py` - backend API and static asset hosting
- `gtm_diligence_assistant/evals.py` - LangSmith experiment runner
- `main.py` - request-level CLI entrypoint
- `inputs/diligence_eval_examples.jsonl` - curated examples
- `dataroom/` - financial PDF source tree

## Common Commands

### Docker Demo
```bash
cp .env.example .env
uv sync
docker compose up --build
```

### Local CLI
```bash
uv run python main.py --qid 6 --pretty
uv run python -m gtm_diligence_assistant.build_index --company Adobe --fy 2024
uv run python -m gtm_diligence_assistant.evals \
  --dataset-jsonl inputs/diligence_eval_examples.jsonl \
  --dataset-name gtm-diligence-assistant-local-v1 \
  --experiment-prefix gtm-diligence-assistant-local
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Runtime Notes

- Python 3.12+
- `uv` manages Python dependencies
- Needs a model provider key such as `OPENAI_API_KEY`
- Optional LangSmith configuration:
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING`
  - `LANGSMITH_PROJECT`
- Docker mounts `/app`, `/data`, `/inputs`, and `/outputs`

## Validation

```bash
uv run python -m unittest discover -s tests -v
cd frontend && npm run build
```
