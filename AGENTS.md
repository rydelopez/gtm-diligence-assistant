# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

GTM Diligence Assistant is a local-first LangGraph application for answering finance diligence questions against a mounted dataroom of PDF filings. The current submission surface is the LangGraph workflow, the FastAPI + React web app, the local evaluation dataset, and the LangSmith experiment path.

## Architecture

- **Workflow**: explicit LangGraph state machine for intake, source planning, retrieval, coverage assessment, reasoning, validation, verification, and response
- **Retrieval**: PyMuPDF-backed PDF tools plus a per-company/FY local vector index for semantic page discovery
- **Web App**: FastAPI backend serving API routes and compiled frontend assets
- **Evaluation**: JSONL-based local dataset plus LangSmith tracing and experiment runs

### Key Components

- `gtm_diligence_assistant/workflow.py` - core LangGraph workflow and state model
- `gtm_diligence_assistant/tools.py` - deterministic local PDF retrieval tools
- `gtm_diligence_assistant/vector_index.py` - page-window embedding index build/load logic
- `gtm_diligence_assistant/web_app.py` - FastAPI API plus static frontend hosting
- `gtm_diligence_assistant/evals.py` - LangSmith dataset sync and experiment runner
- `main.py` - single-request CLI entrypoint
- `inputs/diligence_eval_examples.jsonl` - curated local eval/demo fixture
- `dataroom/` - mounted financial documents organized by company and fiscal year

## Development Commands

### Docker Demo
```bash
cp .env.example .env
uv sync
docker compose up --build
```

Open `http://localhost:8000`.

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

- Requires Python 3.12+
- Uses `uv` for Python dependency management
- Requires at least one model API key such as `OPENAI_API_KEY`
- Optional LangSmith env vars:
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING`
  - `LANGSMITH_PROJECT`
- Docker mounts:
  - repo at `/app`
  - dataroom at `/data`
  - inputs at `/inputs`
  - outputs at `/outputs`

## Testing

```bash
uv run python -m unittest discover -s tests -v
cd frontend && npm run build
```
