from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .models import DiligenceRequest


DEFAULT_DATASET_JSONL = "inputs/diligence_eval_examples.jsonl"


def iter_dataset_records(path: str | Path) -> Iterator[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset JSONL not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Each dataset line must be a JSON object. Got {type(payload)!r} on line {line_number}.")
            yield payload


def load_dataset_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_dataset_records(path))


def request_from_dataset_record(record: dict[str, Any]) -> DiligenceRequest:
    runtime_inputs = record.get("inputs")
    if not isinstance(runtime_inputs, dict):
        raise ValueError("Dataset record is missing a JSON object under the 'inputs' key.")
    return DiligenceRequest.model_validate(runtime_inputs)


def load_request_from_dataset(path: str | Path, qid: int) -> DiligenceRequest:
    for record in iter_dataset_records(path):
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and int(metadata.get("qid", -1)) == qid:
            return request_from_dataset_record(record)
        inputs = record.get("inputs")
        if isinstance(inputs, dict) and int(inputs.get("qid", -1)) == qid:
            return request_from_dataset_record(record)
    raise ValueError(f"Could not find qid={qid} in dataset {path}.")
