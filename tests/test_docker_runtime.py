from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gtm_diligence_assistant.docker_runtime import prepare_runtime_indexes, run_docker_runtime
from gtm_diligence_assistant.models import DiligenceResponse


def _write_dataset(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class FakeWorkflow:
    def __init__(self) -> None:
        self.requests = []

    def run_request_with_trace(self, request):
        self.requests.append(request)
        return (
            DiligenceResponse(
                final_answer="42",
                answer_kind="number",
                explanation="Test response.",
                citations=[],
                confidence=0.5,
                needs_human_review=False,
                errors=[],
            ),
            {
                "opened_files": [],
                "primary_candidate_file": None,
                "primary_file_used": False,
                "retrieval_stop_reason": "coverage_complete",
            },
        )


class RecordingManager:
    init_calls: list[tuple[Path, Path, object]] = []
    build_calls: list[tuple[Path, bool]] = []
    behavior_by_path: dict[str, str] = {}

    def __init__(self, embedding_model, dataroom_root, cache_dir) -> None:  # noqa: ANN001
        self.embedding_model = embedding_model
        self.dataroom_root = Path(dataroom_root)
        self.cache_dir = Path(cache_dir)
        type(self).init_calls.append((self.dataroom_root, self.cache_dir, embedding_model))

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.build_calls = []
        cls.behavior_by_path = {}

    def build_fy_index(self, fy_dir, force: bool = False):  # noqa: ANN001
        fy_path = Path(fy_dir)
        type(self).build_calls.append((fy_path, force))
        behavior = type(self).behavior_by_path.get(str(fy_path), "rebuilt")
        if behavior == "error":
            raise RuntimeError(f"failed to index {fy_path.name}")
        return {
            "fy_dir": str(fy_path),
            "index_path": str(self.cache_dir / fy_path.name / "index.json"),
            "manifest_path": str(self.cache_dir / fy_path.name / "manifest.json"),
            "rebuilt": behavior == "rebuilt",
            "document_count": 3,
        }


class DockerRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingManager.reset()

    def test_prepare_runtime_indexes_builds_only_requested_fy_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataroom_root = root / "dataroom"
            cache_dir = root / "vector_indexes"
            for folder in (
                dataroom_root / "Acme" / "FY 2024",
                dataroom_root / "Acme" / "FY 2023",
                dataroom_root / "Beta" / "FY 2024",
            ):
                folder.mkdir(parents=True)

            dataset_path = root / "dataset.jsonl"
            _write_dataset(
                dataset_path,
                [
                    {
                        "id": "1",
                        "inputs": {
                            "question": "Question 1",
                            "request_id": "1",
                        },
                        "metadata": {"qid": 5, "company": "Acme", "fiscal_year": 2024},
                    },
                    {
                        "id": "2",
                        "inputs": {
                            "question": "Question 2",
                            "request_id": "2",
                        },
                        "metadata": {"qid": 6, "company": "Beta", "fiscal_year": 2024},
                    },
                    {
                        "id": "3",
                        "inputs": {
                            "question": "Duplicate pair",
                            "request_id": "3",
                        },
                        "metadata": {"qid": 7, "company": "Acme", "fiscal_year": 2024},
                    },
                ],
            )

            _, summary = prepare_runtime_indexes(
                dataset_jsonl=dataset_path,
                dataroom_root=dataroom_root,
                vector_index_cache_dir=cache_dir,
                create_embedding_model_fn=lambda provider: object(),
                manager_cls=RecordingManager,
            )

            built_paths = [str(path) for path, _ in RecordingManager.build_calls]
            self.assertEqual(len(built_paths), 2)
            self.assertIn(str(dataroom_root / "Acme" / "FY 2024"), built_paths)
            self.assertIn(str(dataroom_root / "Beta" / "FY 2024"), built_paths)
            self.assertNotIn(str(dataroom_root / "Acme" / "FY 2023"), built_paths)
            self.assertTrue(summary["embeddings_available"])
            self.assertEqual(len(summary["requested_fy_folders"]), 2)

    def test_run_docker_runtime_passes_runtime_paths_to_manager_and_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataroom_root = root / "data"
            cache_dir = root / "outputs" / "vector_indexes"
            out_jsonl = root / "outputs" / "results.jsonl"
            out_summary = root / "outputs" / "summary.json"
            (dataroom_root / "Acme" / "FY 2024").mkdir(parents=True)

            dataset_path = root / "dataset.jsonl"
            _write_dataset(
                dataset_path,
                [
                    {
                        "id": "1",
                        "inputs": {
                            "question": "Question 1",
                            "request_id": "1",
                        },
                        "metadata": {"qid": 5, "company": "Acme", "fiscal_year": 2024},
                        "outputs": {"expected_kind": "number", "expected_value": 42},
                    }
                ],
            )

            workflow = FakeWorkflow()
            workflow_args: dict[str, object] = {}

            def workflow_factory(runtime_root: Path, runtime_cache_dir: Path, embedding_model):  # noqa: ANN001
                workflow_args["dataroom_root"] = runtime_root
                workflow_args["cache_dir"] = runtime_cache_dir
                workflow_args["embedding_model"] = embedding_model
                return workflow

            summary = run_docker_runtime(
                dataset_jsonl=dataset_path,
                dataroom_root=dataroom_root,
                vector_index_cache_dir=cache_dir,
                out_jsonl=out_jsonl,
                out_summary_json=out_summary,
                create_embedding_model_fn=lambda provider: "embedding-sentinel",
                manager_cls=RecordingManager,
                workflow_factory=workflow_factory,
            )

            self.assertEqual(RecordingManager.init_calls[0][0], dataroom_root)
            self.assertEqual(RecordingManager.init_calls[0][1], cache_dir)
            self.assertEqual(workflow_args["dataroom_root"], dataroom_root)
            self.assertEqual(workflow_args["cache_dir"], cache_dir)
            self.assertEqual(workflow_args["embedding_model"], "embedding-sentinel")
            self.assertTrue(out_summary.exists())
            self.assertIn("index_prep", summary)
            self.assertTrue(summary["index_prep"]["embeddings_available"])

    def test_run_docker_runtime_degrades_cleanly_when_embeddings_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataroom_root = root / "data"
            cache_dir = root / "outputs" / "vector_indexes"
            out_jsonl = root / "outputs" / "results.jsonl"
            out_summary = root / "outputs" / "summary.json"
            (dataroom_root / "Acme" / "FY 2024").mkdir(parents=True)

            dataset_path = root / "dataset.jsonl"
            _write_dataset(
                dataset_path,
                [
                    {
                        "id": "1",
                        "inputs": {
                            "question": "Question 1",
                            "request_id": "1",
                        },
                        "metadata": {"qid": 5, "company": "Acme", "fiscal_year": 2024},
                    }
                ],
            )

            workflow = FakeWorkflow()
            captured_embedding_models: list[object | None] = []

            def workflow_factory(runtime_root: Path, runtime_cache_dir: Path, embedding_model):  # noqa: ANN001
                captured_embedding_models.append(embedding_model)
                return workflow

            summary = run_docker_runtime(
                dataset_jsonl=dataset_path,
                dataroom_root=dataroom_root,
                vector_index_cache_dir=cache_dir,
                out_jsonl=out_jsonl,
                out_summary_json=out_summary,
                create_embedding_model_fn=lambda provider: (_ for _ in ()).throw(RuntimeError("embed boom")),
                manager_cls=RecordingManager,
                workflow_factory=workflow_factory,
            )

            self.assertEqual(captured_embedding_models, [None])
            self.assertFalse(summary["index_prep"]["embeddings_available"])
            self.assertIn("embedding_error", summary["index_prep"])
            self.assertTrue(out_jsonl.exists())
            self.assertFalse(RecordingManager.init_calls)

    def test_run_docker_runtime_records_partial_index_failures_without_aborting_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataroom_root = root / "data"
            cache_dir = root / "outputs" / "vector_indexes"
            out_jsonl = root / "outputs" / "results.jsonl"
            out_summary = root / "outputs" / "summary.json"
            acme_fy = dataroom_root / "Acme" / "FY 2024"
            beta_fy = dataroom_root / "Beta" / "FY 2024"
            acme_fy.mkdir(parents=True)
            beta_fy.mkdir(parents=True)

            RecordingManager.behavior_by_path[str(beta_fy)] = "error"

            dataset_path = root / "dataset.jsonl"
            _write_dataset(
                dataset_path,
                [
                    {
                        "id": "1",
                        "inputs": {
                            "question": "Question 1",
                            "request_id": "1",
                        },
                        "metadata": {"qid": 5, "company": "Acme", "fiscal_year": 2024},
                    },
                    {
                        "id": "2",
                        "inputs": {
                            "question": "Question 2",
                            "request_id": "2",
                        },
                        "metadata": {"qid": 6, "company": "Beta", "fiscal_year": 2024},
                    },
                ],
            )

            workflow = FakeWorkflow()
            summary = run_docker_runtime(
                dataset_jsonl=dataset_path,
                dataroom_root=dataroom_root,
                vector_index_cache_dir=cache_dir,
                out_jsonl=out_jsonl,
                out_summary_json=out_summary,
                create_embedding_model_fn=lambda provider: object(),
                manager_cls=RecordingManager,
                workflow_factory=lambda runtime_root, runtime_cache_dir, embedding_model: workflow,
            )

            self.assertEqual(len(workflow.requests), 2)
            self.assertEqual(summary["total_examples"], 2)
            self.assertEqual(len(summary["index_prep"]["indexed_folders"]), 1)
            self.assertEqual(len(summary["index_prep"]["failed_folders"]), 1)
            self.assertIn("Beta", summary["index_prep"]["failed_folders"][0]["fy_dir"])


if __name__ == "__main__":
    unittest.main()
