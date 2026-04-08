from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langsmith import Client, configure
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.utils import LangSmithConflictError

from .config import load_local_env
from .dataset import DEFAULT_DATASET_JSONL, load_dataset_records
from .embeddings import create_embedding_model
from .evaluation_metrics import compute_file_metrics, expected_files_from_record
from .llm import create_chat_model
from .models import DiligenceRequest
from .scoring import evaluate_numeric_answer
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR
from .workflow import DiligenceWorkflow


DEFAULT_DATASET_NAME = "gtm-diligence-assistant-local-v1"


def _dataset_metadata(examples: list[dict[str, Any]]) -> dict[str, Any]:
    qids = [
        example["metadata"]["qid"]
        for example in examples
        if isinstance(example.get("metadata"), dict) and "qid" in example["metadata"]
    ]
    return {"workflow_version": "v1", "qids": qids}


def _normalize_example_record(record: dict[str, Any]) -> dict[str, Any]:
    example_id = record.get("id")
    if not example_id:
        raise ValueError("Each eval dataset record must include a stable 'id'.")
    inputs = record.get("inputs") if isinstance(record.get("inputs"), dict) else {}
    outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return {
        "local_id": str(example_id),
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "metadata": dict(metadata),
    }


def _normalize_remote_example(example: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(example, "id")),
        "inputs": dict(getattr(example, "inputs", {}) or {}),
        "outputs": dict(getattr(example, "outputs", {}) or {}),
        "metadata": dict(getattr(example, "metadata", {}) or {}),
    }


def _payload_changed(local_example: dict[str, Any], remote_example: Any) -> bool:
    remote_payload = _normalize_remote_example(remote_example)
    return (
        local_example["inputs"] != remote_payload["inputs"]
        or local_example["outputs"] != remote_payload["outputs"]
        or local_example["metadata"] != remote_payload["metadata"]
    )


def _langsmith_example_id(dataset_id: Any, local_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"gtm-diligence-assistant:{dataset_id}:{local_id}"))


def _create_or_update_example(client: Client, dataset_id: Any, example: dict[str, Any]) -> str:
    try:
        client.create_example(
            dataset_id=dataset_id,
            example_id=example["id"],
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
        )
        return "created"
    except LangSmithConflictError:
        client.update_example(
            example_id=example["id"],
            dataset_id=dataset_id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
        )
        return "updated"


def sync_langsmith_dataset(client: Client, dataset_name: str, examples: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    normalized_examples = [_normalize_example_record(example) for example in examples]
    local_examples_by_id: dict[str, dict[str, Any]] = {}
    for example in normalized_examples:
        if example["local_id"] in local_examples_by_id:
            raise ValueError(f"Duplicate eval dataset id detected: {example['local_id']}")
        local_examples_by_id[example["local_id"]] = example

    existing_dataset = next(client.list_datasets(dataset_name=dataset_name, limit=1), None)
    if existing_dataset is None:
        existing_dataset = client.create_dataset(
            dataset_name,
            description="Deterministic local-only diligence eval set loaded from the curated JSONL fixture.",
            metadata=_dataset_metadata(examples),
        )

    desired_examples_by_id: dict[str, dict[str, Any]] = {}
    for local_example in normalized_examples:
        remote_id = _langsmith_example_id(existing_dataset.id, local_example["local_id"])
        desired_examples_by_id[remote_id] = {
            "id": remote_id,
            "inputs": local_example["inputs"],
            "outputs": local_example["outputs"],
            "metadata": {
                **local_example["metadata"],
                "local_example_id": local_example["local_id"],
            },
        }

    remote_examples = list(client.list_examples(dataset_id=existing_dataset.id))
    remote_examples_by_id = {str(example.id): example for example in remote_examples}

    to_create = [
        local_example
        for example_id, local_example in desired_examples_by_id.items()
        if example_id not in remote_examples_by_id
    ]
    to_update = [
        local_example
        for example_id, local_example in desired_examples_by_id.items()
        if example_id in remote_examples_by_id and _payload_changed(local_example, remote_examples_by_id[example_id])
    ]
    to_delete = [example_id for example_id in remote_examples_by_id if example_id not in desired_examples_by_id]

    created = 0
    updated = 0
    for example in to_create:
        result = _create_or_update_example(client, existing_dataset.id, example)
        if result == "created":
            created += 1
        else:
            updated += 1

    if to_update:
        client.update_examples(dataset_id=existing_dataset.id, updates=to_update)
        updated += len(to_update)
    if to_delete:
        client.delete_examples(example_ids=to_delete)

    summary = {
        "dataset_name": dataset_name,
        "dataset_id": str(existing_dataset.id),
        "total_local_examples": len(local_examples_by_id),
        "created": created,
        "updated": updated,
        "deleted": len(to_delete),
        "unchanged": max(0, len(desired_examples_by_id) - created - updated),
    }
    return existing_dataset, summary


@run_evaluator
def numeric_accuracy_evaluator(run, example) -> EvaluationResult:
    outputs = run.outputs or {}
    response = outputs.get("response", outputs)
    reference_outputs = example.outputs if example else {}
    evaluation = evaluate_numeric_answer(
        answer_kind=response.get("answer_kind", "unknown"),
        final_answer=response.get("final_answer"),
        expected_kind=reference_outputs.get("expected_kind"),
        expected_value=reference_outputs.get("expected_value"),
    )
    return EvaluationResult(
        key="numeric_within_tolerance",
        score=1.0 if evaluation["within_tolerance"] else 0.0,
        comment=(
            f"tier={evaluation['tier']}, rpe={evaluation['relative_percent_error']}, "
            f"parsed={evaluation['parsed_value']}"
        ),
    )


@run_evaluator
def citation_presence_evaluator(run, example) -> EvaluationResult:
    outputs = run.outputs or {}
    response = outputs.get("response", outputs)
    citations = response.get("citations") or []
    return EvaluationResult(
        key="citation_presence",
        score=1.0 if len(citations) > 0 else 0.0,
        comment=f"citation_count={len(citations)}",
    )


@run_evaluator
def expected_file_behavior_evaluator(run, example) -> EvaluationResult:
    outputs = run.outputs or {}
    response = outputs.get("response", outputs)
    telemetry = outputs.get("telemetry", {})
    expected_files = expected_files_from_record(
        {
            "outputs": example.outputs if example else {},
            "metadata": example.metadata if example else {},
        }
    )
    metrics = compute_file_metrics(response, telemetry, expected_files)
    return EvaluationResult(
        key="expected_file_behavior",
        score=float(metrics["expected_file_behavior"]),
        comment=(
            f"expected={expected_files}, opened={metrics['opened_files']}, "
            f"main_cited={metrics['main_cited_file']}, primary_candidate={metrics['primary_candidate_file']}, "
            f"opened_expected={metrics['opened_expected_file']}, aligned={metrics['primary_file_alignment']}"
        ),
    )


def run_langsmith_experiment(
    dataset_jsonl: str | Path = DEFAULT_DATASET_JSONL,
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = "gtm-diligence-assistant",
) -> Any:
    examples = load_dataset_records(dataset_jsonl)
    client = Client(auto_batch_tracing=False)
    configure(
        client=client,
        enabled=True,
        project_name=os.environ.get("LANGSMITH_PROJECT"),
    )
    dataset, sync_summary = sync_langsmith_dataset(client, dataset_name, examples)
    print(json.dumps({"event": "langsmith_dataset_sync", **sync_summary}, indent=2), flush=True)
    workflow = DiligenceWorkflow(
        create_chat_model(),
        embedding_model=create_embedding_model(),
        vector_index_cache_dir=Path(os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR)),
    )

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        request = DiligenceRequest.model_validate(inputs)
        response, telemetry = workflow.run_request_with_trace(request)
        return {
            "response": response.model_dump(mode="json"),
            "telemetry": telemetry,
        }

    return client.evaluate(
        target,
        data=dataset_name,
        evaluators=[
            numeric_accuracy_evaluator,
            citation_presence_evaluator,
            expected_file_behavior_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        description="Deterministic GTM diligence assistant experiment",
        metadata={"workflow_version": "local_only_v1"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the curated local-only JSONL eval set in LangSmith.")
    parser.add_argument("--dataset-jsonl", default=os.environ.get("DATASET_JSONL", DEFAULT_DATASET_JSONL))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--experiment-prefix", default="gtm-diligence-assistant")
    return parser


def main() -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    result = run_langsmith_experiment(
        dataset_jsonl=args.dataset_jsonl,
        dataset_name=args.dataset_name,
        experiment_prefix=args.experiment_prefix,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
