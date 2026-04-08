from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import load_local_env
from .dataset import DEFAULT_DATASET_JSONL, load_dataset_records, request_from_dataset_record
from .embeddings import create_embedding_model
from .evaluation_metrics import compute_file_metrics, expected_files_from_record
from .llm import create_chat_model
from .models import DiligenceResponse
from .scoring import evaluate_numeric_answer
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR
from .workflow import DiligenceWorkflow


DEFAULT_DATAROOM_ROOT = "/data"
DEFAULT_OUT_JSONL = "/outputs/gtm_diligence_results.jsonl"
DEFAULT_OUT_SUMMARY_JSON = "/outputs/gtm_diligence_summary.json"


def evaluate_record_response(
    response: DiligenceResponse,
    record: dict[str, Any],
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = record.get("outputs", {})
    numeric = evaluate_numeric_answer(
        answer_kind=response.answer_kind,
        final_answer=response.final_answer,
        expected_kind=outputs.get("expected_kind") if isinstance(outputs, dict) else None,
        expected_value=outputs.get("expected_value") if isinstance(outputs, dict) else None,
    )
    citation_presence = len(response.citations) > 0
    file_metrics = compute_file_metrics(response, telemetry, expected_files_from_record(record))

    return {
        "numeric_within_tolerance": numeric["within_tolerance"],
        "numeric_tier": numeric["tier"],
        "relative_percent_error": numeric["relative_percent_error"],
        "citation_presence": citation_presence,
        "expected_file_behavior": file_metrics["expected_file_behavior"],
        "needs_human_review": response.needs_human_review,
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total_examples": 0,
            "numeric_within_tolerance_rate": 0.0,
            "citation_presence_rate": 0.0,
            "expected_file_behavior_mean": 0.0,
            "needs_human_review_rate": 0.0,
        }

    def _rate(key: str) -> float:
        hits = sum(1 for result in results if result["evaluation"].get(key))
        return round(hits / total, 4)

    human_review_hits = sum(1 for result in results if result["response"]["needs_human_review"])
    expected_file_behavior_total = sum(
        float(result["evaluation"].get("expected_file_behavior", 0.0))
        for result in results
    )
    return {
        "total_examples": total,
        "numeric_within_tolerance_rate": _rate("numeric_within_tolerance"),
        "citation_presence_rate": _rate("citation_presence"),
        "expected_file_behavior_mean": round(expected_file_behavior_total / total, 4),
        "needs_human_review_rate": round(human_review_hits / total, 4),
    }


def run_batch(
    dataset_jsonl: str | Path,
    dataroom_root: str | Path,
    out_jsonl: str | Path,
    out_summary_json: str | Path,
    workflow: DiligenceWorkflow | Any | None = None,
    index_prep_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = load_dataset_records(dataset_jsonl)
    workflow = workflow or DiligenceWorkflow(
        create_chat_model(),
        dataroom_root=Path(dataroom_root),
        embedding_model=create_embedding_model(),
        vector_index_cache_dir=Path(os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR)),
    )

    result_rows: list[dict[str, Any]] = []
    for record in records:
        request = request_from_dataset_record(record)
        if hasattr(workflow, "run_request_with_trace"):
            response, telemetry = workflow.run_request_with_trace(request)
        else:
            response = workflow.run_request(request)
            telemetry = {}
        evaluation = evaluate_record_response(response, record, telemetry)
        result_rows.append(
            {
                "id": record.get("id"),
                "request": request.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "telemetry": telemetry,
                "evaluation": evaluation,
            }
        )

    out_jsonl_path = Path(out_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row) + "\n")

    summary = _build_summary(result_rows)
    if index_prep_summary is not None:
        summary["index_prep"] = index_prep_summary
    out_summary_path = Path(out_summary_json)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    load_local_env()
    dataset_jsonl = os.environ.get("DATASET_JSONL", DEFAULT_DATASET_JSONL)
    dataroom_root = os.environ.get("DATAROOM_ROOT", DEFAULT_DATAROOM_ROOT)
    out_jsonl = os.environ.get("OUT_JSONL", DEFAULT_OUT_JSONL)
    out_summary_json = os.environ.get("OUT_SUMMARY_JSON", DEFAULT_OUT_SUMMARY_JSON)

    summary = run_batch(
        dataset_jsonl=dataset_jsonl,
        dataroom_root=dataroom_root,
        out_jsonl=out_jsonl,
        out_summary_json=out_summary_json,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
