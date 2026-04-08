from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from .config import load_local_env
from .dataset import DEFAULT_DATASET_JSONL, load_dataset_records
from .docker_runtime import DEFAULT_DATAROOM_ROOT, prepare_runtime_indexes
from .llm import create_chat_model
from .models import DiligenceRequest, DiligenceResponse
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR
from .workflow import DiligenceWorkflow


DEFAULT_UI_DIST_DIR = "frontend/dist"
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8000


class DemoExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    qid: int | None = None
    company: str | None = None
    fiscal_year: int | None = None
    question: str
    request_id: str | None = None
    label: str


class ExamplesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: list[DemoExample] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    workflow_ready: bool
    ui_enabled: bool
    examples_loaded: int
    startup_error: str | None = None
    index_prep: dict[str, Any] | None = None


class RunWorkflowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: DiligenceResponse
    telemetry: dict[str, Any] = Field(default_factory=dict)


def _safe_load_examples(dataset_jsonl: str | Path) -> tuple[list[DemoExample], str | None]:
    try:
        records = load_dataset_records(dataset_jsonl)
    except Exception as exc:
        return [], str(exc)

    examples: list[DemoExample] = []
    for record in records:
        inputs = record.get("inputs")
        metadata = record.get("metadata")
        if not isinstance(inputs, dict):
            continue
        metadata = metadata if isinstance(metadata, dict) else {}
        qid = metadata.get("qid")
        company = metadata.get("company")
        fiscal_year = metadata.get("fiscal_year")
        label = str(company or record.get("id", "Example"))
        examples.append(
            DemoExample(
                id=record.get("id"),
                qid=int(qid) if qid is not None else None,
                company=str(company) if company else None,
                fiscal_year=int(fiscal_year) if fiscal_year is not None else None,
                question=str(inputs.get("question", "")),
                request_id=str(inputs.get("request_id")) if inputs.get("request_id") else None,
                label=label,
            )
        )
    return examples, None


class WebAppRuntime:
    def __init__(
        self,
        workflow: DiligenceWorkflow | None,
        examples: list[DemoExample],
        *,
        startup_error: str | None = None,
        index_prep_summary: dict[str, Any] | None = None,
    ) -> None:
        self.workflow = workflow
        self.examples = examples
        self.startup_error = startup_error
        self.index_prep_summary = index_prep_summary or {}

    @classmethod
    def from_env(cls) -> "WebAppRuntime":
        load_local_env()
        dataset_jsonl = os.environ.get("DATASET_JSONL", DEFAULT_DATASET_JSONL)
        dataroom_root = Path(os.environ.get("DATAROOM_ROOT", DEFAULT_DATAROOM_ROOT))
        vector_index_cache_dir = Path(os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR))

        examples, example_error = _safe_load_examples(dataset_jsonl)
        embedding_model, index_prep_summary = prepare_runtime_indexes(
            dataset_jsonl=dataset_jsonl,
            dataroom_root=dataroom_root,
            vector_index_cache_dir=vector_index_cache_dir,
        )

        startup_errors: list[str] = []
        if example_error:
            startup_errors.append(f"Could not load demo examples: {example_error}")

        workflow: DiligenceWorkflow | None = None
        try:
            workflow = DiligenceWorkflow(
                llm=create_chat_model(),
                dataroom_root=dataroom_root,
                embedding_model=embedding_model,
                vector_index_cache_dir=vector_index_cache_dir,
            )
        except Exception as exc:
            startup_errors.append(str(exc))

        startup_error = " | ".join(startup_errors) if startup_errors else None
        return cls(
            workflow=workflow,
            examples=examples,
            startup_error=startup_error,
            index_prep_summary=index_prep_summary,
        )

    def _ensure_request_index(self, request: DiligenceRequest) -> None:
        if self.workflow is None or self.workflow.index_manager is None:
            return
        company, fiscal_year = self.workflow.infer_request_identity(request)
        if not company or fiscal_year is None:
            return

        company_dir = self.workflow._resolve_company_dir(company)
        fy_dir = self.workflow._resolve_fiscal_year_dir(company_dir, fiscal_year)
        if fy_dir is None:
            return
        if self.workflow.index_manager.load_fy_index(fy_dir) is not None:
            return

        try:
            self.workflow.index_manager.build_fy_index(fy_dir)
        except Exception as exc:
            index_errors = list(self.index_prep_summary.get("failed_folders", []))
            index_errors.append({"fy_dir": str(fy_dir), "error": str(exc)})
            self.index_prep_summary["failed_folders"] = index_errors

    def run_request(self, request: DiligenceRequest) -> tuple[DiligenceResponse, dict[str, Any]]:
        if self.workflow is None:
            raise RuntimeError(self.startup_error or "Workflow is unavailable due to startup configuration errors.")
        self._ensure_request_index(request)
        return self.workflow.run_request_with_trace(request)


def create_app(
    runtime: WebAppRuntime | None = None,
    ui_dist_dir: str | Path | None = None,
) -> FastAPI:
    resolved_ui_dist_dir = Path(
        ui_dist_dir
        or os.environ.get("UI_DIST_DIR", DEFAULT_UI_DIST_DIR)
    )
    index_file = resolved_ui_dist_dir / "index.html"
    assets_dir = resolved_ui_dist_dir / "assets"
    ui_enabled = index_file.exists()

    def _get_runtime() -> WebAppRuntime:
        current_runtime = getattr(app.state, "runtime", None)
        if current_runtime is None:
            current_runtime = WebAppRuntime.from_env()
            app.state.runtime = current_runtime
        return current_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime
        _get_runtime()
        yield

    app = FastAPI(
        title="GTM Diligence Assistant",
        summary="Demo-ready UI and API for the local-only GTM diligence workflow.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        runtime = _get_runtime()
        status = "ok" if runtime.workflow is not None else "degraded"
        return HealthResponse(
            status=status,
            workflow_ready=runtime.workflow is not None,
            ui_enabled=ui_enabled,
            examples_loaded=len(runtime.examples),
            startup_error=runtime.startup_error,
            index_prep=runtime.index_prep_summary,
        )

    @app.get("/api/examples", response_model=ExamplesResponse)
    def examples() -> ExamplesResponse:
        runtime = _get_runtime()
        return ExamplesResponse(examples=runtime.examples)

    @app.post("/api/run", response_model=RunWorkflowResponse)
    def run_workflow(request: DiligenceRequest) -> RunWorkflowResponse:
        runtime = _get_runtime()
        try:
            response, telemetry = runtime.run_request(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive HTTP wrapper
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc
        return RunWorkflowResponse(response=response, telemetry=telemetry)

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str = ""):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found.")
        if ui_enabled:
            return FileResponse(index_file)
        return JSONResponse(
            status_code=200,
            content={
                "message": (
                    "UI assets are not built yet. Run `npm install && npm run build` in the frontend directory "
                    "for a local static build, or use the Vite dev server during development."
                )
            },
        )

    return app


app = create_app()


def main() -> int:
    load_local_env()
    host = os.environ.get("WEB_HOST", DEFAULT_WEB_HOST)
    port = int(os.environ.get("WEB_PORT", str(DEFAULT_WEB_PORT)))
    uvicorn.run("gtm_diligence_assistant.web_app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
