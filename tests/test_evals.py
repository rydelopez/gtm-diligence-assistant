from __future__ import annotations

import unittest
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from gtm_diligence_assistant.evals import expected_file_behavior_evaluator, sync_langsmith_dataset
from langsmith.utils import LangSmithConflictError


@dataclass
class FakeDataset:
    id: str
    name: str


@dataclass
class FakeExample:
    id: str
    inputs: dict
    outputs: dict
    metadata: dict


@dataclass
class FakeRun:
    id: str
    outputs: dict


class FakeClient:
    def __init__(self, *, dataset: FakeDataset | None = None, examples: list[FakeExample] | None = None) -> None:
        self.dataset = dataset
        self.examples = {example.id: example for example in (examples or [])}
        self.created_datasets: list[dict] = []
        self.created_examples: list[dict] = []
        self.created_single_examples: list[dict] = []
        self.updated_examples: list[dict] = []
        self.updated_single_examples: list[dict] = []
        self.deleted_example_ids: list[str] = []
        self.conflict_on_create_ids: set[str] = set()

    def list_datasets(self, dataset_name: str, limit: int = 1):
        del dataset_name, limit
        if self.dataset is None:
            return iter([])
        return iter([self.dataset])

    def create_dataset(self, dataset_name: str, description: str | None = None, metadata: dict | None = None):
        self.created_datasets.append(
            {"dataset_name": dataset_name, "description": description, "metadata": metadata}
        )
        self.dataset = FakeDataset(id="dataset-1", name=dataset_name)
        return self.dataset

    def list_examples(self, dataset_id: str):
        del dataset_id
        return iter(self.examples.values())

    def create_examples(self, dataset_id: str, examples: list[dict]):
        del dataset_id
        self.created_examples.extend(examples)
        for example in examples:
            self.examples[str(example["id"])] = FakeExample(
                id=str(example["id"]),
                inputs=dict(example["inputs"]),
                outputs=dict(example["outputs"]),
                metadata=dict(example["metadata"]),
            )

    def create_example(self, dataset_id: str, example_id: str, inputs: dict, outputs: dict, metadata: dict):
        del dataset_id
        if str(example_id) in self.conflict_on_create_ids:
            raise LangSmithConflictError("already exists")
        payload = {"id": str(example_id), "inputs": inputs, "outputs": outputs, "metadata": metadata}
        self.created_single_examples.append(payload)
        self.examples[str(example_id)] = FakeExample(
            id=str(example_id),
            inputs=dict(inputs),
            outputs=dict(outputs),
            metadata=dict(metadata),
        )

    def update_examples(self, dataset_id: str, updates: list[dict]):
        del dataset_id
        self.updated_examples.extend(updates)
        for example in updates:
            self.examples[str(example["id"])] = FakeExample(
                id=str(example["id"]),
                inputs=dict(example["inputs"]),
                outputs=dict(example["outputs"]),
                metadata=dict(example["metadata"]),
            )

    def update_example(self, example_id: str, dataset_id: str, inputs: dict, outputs: dict, metadata: dict):
        del dataset_id
        payload = {"id": str(example_id), "inputs": inputs, "outputs": outputs, "metadata": metadata}
        self.updated_single_examples.append(payload)
        self.examples[str(example_id)] = FakeExample(
            id=str(example_id),
            inputs=dict(inputs),
            outputs=dict(outputs),
            metadata=dict(metadata),
        )

    def delete_examples(self, example_ids: list[str]):
        self.deleted_example_ids.extend(str(example_id) for example_id in example_ids)
        for example_id in example_ids:
            self.examples.pop(str(example_id), None)


def _example(example_id: str, question: str = "What is net debt?", expected_value: float = 10.0) -> dict:
    return {
        "id": example_id,
        "inputs": {
            "question": question,
            "request_id": f"request-{example_id}",
        },
        "outputs": {
            "expected_kind": "number",
            "expected_value": expected_value,
        },
        "metadata": {
            "qid": 1,
            "company": "Acme",
            "fiscal_year": 2024,
            "correct_any_of_files": ["Acme 2024 10-K.pdf"],
        },
    }


def _remote_example_id(dataset_id: str, local_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"gtm-diligence-assistant:{dataset_id}:{local_id}"))


class LangSmithDatasetSyncTests(unittest.TestCase):
    def test_sync_creates_dataset_and_all_examples_when_absent(self) -> None:
        client = FakeClient()
        dataset, summary = sync_langsmith_dataset(client, "evals", [_example("ex-1"), _example("ex-2")])

        self.assertEqual(dataset.name, "evals")
        self.assertEqual(len(client.created_datasets), 1)
        self.assertEqual(len(client.created_single_examples), 2)
        self.assertEqual(
            {payload["id"] for payload in client.created_single_examples},
            {_remote_example_id("dataset-1", "ex-1"), _remote_example_id("dataset-1", "ex-2")},
        )
        self.assertEqual(summary["created"], 2)
        self.assertEqual(summary["updated"], 0)
        self.assertEqual(summary["deleted"], 0)
        self.assertEqual(summary["unchanged"], 0)

    def test_sync_is_noop_when_dataset_examples_are_identical(self) -> None:
        example = _example("ex-1")
        remote_id = _remote_example_id("dataset-1", "ex-1")
        client = FakeClient(
            dataset=FakeDataset(id="dataset-1", name="evals"),
            examples=[
                FakeExample(
                    id=remote_id,
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                    metadata={**example["metadata"], "local_example_id": "ex-1"},
                )
            ],
        )

        _, summary = sync_langsmith_dataset(client, "evals", [example])

        self.assertEqual(client.created_examples, [])
        self.assertEqual(client.updated_examples, [])
        self.assertEqual(client.deleted_example_ids, [])
        self.assertEqual(summary["created"], 0)
        self.assertEqual(summary["updated"], 0)
        self.assertEqual(summary["deleted"], 0)
        self.assertEqual(summary["unchanged"], 1)

    def test_sync_updates_existing_example_when_payload_changes(self) -> None:
        local = _example("ex-1", expected_value=42.0)
        remote = _example("ex-1", expected_value=10.0)
        remote_id = _remote_example_id("dataset-1", "ex-1")
        client = FakeClient(
            dataset=FakeDataset(id="dataset-1", name="evals"),
            examples=[
                FakeExample(
                    id=remote_id,
                    inputs=remote["inputs"],
                    outputs=remote["outputs"],
                    metadata={**remote["metadata"], "local_example_id": "ex-1"},
                )
            ],
        )

        _, summary = sync_langsmith_dataset(client, "evals", [local])

        self.assertEqual(len(client.updated_examples), 1)
        self.assertEqual(client.updated_examples[0]["outputs"]["expected_value"], 42.0)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["unchanged"], 0)

    def test_sync_creates_missing_examples_and_deletes_stale_examples(self) -> None:
        kept = _example("ex-keep")
        new_local = _example("ex-new")
        stale_remote = _example("ex-stale")
        client = FakeClient(
            dataset=FakeDataset(id="dataset-1", name="evals"),
            examples=[
                FakeExample(
                    id=_remote_example_id("dataset-1", "ex-keep"),
                    inputs=kept["inputs"],
                    outputs=kept["outputs"],
                    metadata={**kept["metadata"], "local_example_id": "ex-keep"},
                ),
                FakeExample(
                    id=_remote_example_id("dataset-1", "ex-stale"),
                    inputs=stale_remote["inputs"],
                    outputs=stale_remote["outputs"],
                    metadata={**stale_remote["metadata"], "local_example_id": "ex-stale"},
                ),
            ],
        )

        _, summary = sync_langsmith_dataset(client, "evals", [kept, new_local])

        self.assertEqual(len(client.created_single_examples), 1)
        self.assertEqual(client.created_single_examples[0]["id"], _remote_example_id("dataset-1", "ex-new"))
        self.assertEqual(client.deleted_example_ids, [_remote_example_id("dataset-1", "ex-stale")])
        self.assertEqual(summary["created"], 1)
        self.assertEqual(summary["deleted"], 1)
        self.assertEqual(summary["unchanged"], 1)

    def test_sync_falls_back_to_update_when_create_hits_conflict(self) -> None:
        local = _example("ex-1", expected_value=42.0)
        client = FakeClient(dataset=FakeDataset(id="dataset-1", name="evals"))
        remote_id = _remote_example_id("dataset-1", "ex-1")
        client.conflict_on_create_ids.add(remote_id)

        _, summary = sync_langsmith_dataset(client, "evals", [local])

        self.assertEqual(client.created_single_examples, [])
        self.assertEqual(len(client.updated_single_examples), 1)
        self.assertEqual(client.updated_single_examples[0]["id"], remote_id)
        self.assertEqual(client.updated_single_examples[0]["outputs"]["expected_value"], 42.0)
        self.assertEqual(summary["created"], 0)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["unchanged"], 0)

    def test_sync_rejects_duplicate_local_ids(self) -> None:
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "Duplicate eval dataset id detected"):
            sync_langsmith_dataset(client, "evals", [_example("ex-1"), _example("ex-1")])

    def test_expected_file_behavior_evaluator_scores_alignment_open_only_and_miss(self) -> None:
        example = FakeExample(
            id="ex-1",
            inputs={},
            outputs={"correct_any_of_files": ["Acme 2024 10-K.pdf"]},
            metadata={"correct_any_of_files": ["Acme 2024 10-K.pdf"]},
        )

        aligned = expected_file_behavior_evaluator(
            FakeRun(
                id="run-aligned",
                outputs={
                    "response": {
                        "citations": [{"source_label": "Acme 2024 10-K.pdf"}],
                    },
                    "telemetry": {
                        "opened_files": ["/tmp/Acme 2024 10-K.pdf"],
                        "primary_candidate_file": "/tmp/Acme 2024 10-K.pdf",
                        "primary_file_used": True,
                    },
                }
            ),
            example,
        )
        opened_only = expected_file_behavior_evaluator(
            FakeRun(
                id="run-opened-only",
                outputs={
                    "response": {"citations": []},
                    "telemetry": {
                        "opened_files": ["/tmp/Acme 2024 10-K.pdf"],
                        "primary_candidate_file": "/tmp/Other file.pdf",
                        "primary_file_used": False,
                    },
                }
            ),
            example,
        )
        miss = expected_file_behavior_evaluator(
            FakeRun(
                id="run-miss",
                outputs={
                    "response": {"citations": []},
                    "telemetry": {
                        "opened_files": ["/tmp/Other file.pdf"],
                        "primary_candidate_file": "/tmp/Other file.pdf",
                        "primary_file_used": False,
                    },
                }
            ),
            example,
        )

        self.assertEqual(aligned.key, "expected_file_behavior")
        self.assertEqual(aligned.score, 1.0)
        self.assertEqual(opened_only.score, 0.5)
        self.assertEqual(miss.score, 0.0)


if __name__ == "__main__":
    unittest.main()
