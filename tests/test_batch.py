from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gtm_diligence_assistant.batch import run_batch
from gtm_diligence_assistant.models import DiligenceResponse


class FakeWorkflow:
    def __init__(self) -> None:
        self.requests = []

    def run_request_with_trace(self, request):
        self.requests.append(request)
        return (
            DiligenceResponse(
                final_answer="5900",
                answer_kind="number",
                explanation="Net debt equals debt plus lease liabilities minus cash.",
                citations=[],
                confidence=0.5,
                needs_human_review=False,
                errors=[],
            ),
            {
                "opened_files": ["/tmp/Acme 2024 10-K.pdf"],
                "primary_candidate_file": "/tmp/Acme 2024 10-K.pdf",
                "primary_file_used": True,
                "retrieval_stop_reason": "coverage_complete",
            },
        )


class BatchTests(unittest.TestCase):
    def test_run_batch_uses_only_inputs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.jsonl"
            out_jsonl = Path(tmpdir) / "results.jsonl"
            out_summary = Path(tmpdir) / "summary.json"
            record = {
                "id": "example-1",
                "inputs": {
                    "question": "What is Acme's net debt?",
                    "request_id": "example-1",
                },
                "outputs": {
                    "expected_kind": "number",
                    "expected_value": 5900,
                    "correct_any_of_files": ["Acme 2024 10-K.pdf"],
                },
                "metadata": {
                    "qid": 6,
                    "company": "Acme",
                    "fiscal_year": 2024,
                    "correct_any_of_files": ["Acme 2024 10-K.pdf"],
                },
            }
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            workflow = FakeWorkflow()
            summary = run_batch(
                dataset_jsonl=dataset_path,
                dataroom_root=tmpdir,
                out_jsonl=out_jsonl,
                out_summary_json=out_summary,
                workflow=workflow,
                index_prep_summary={"embeddings_available": True, "requested_fy_folders": ["/data/Acme/FY 2024"]},
            )

            self.assertEqual(len(workflow.requests), 1)
            self.assertNotIn("outputs", workflow.requests[0].model_dump(mode="json"))
            self.assertTrue(out_jsonl.exists())
            self.assertTrue(out_summary.exists())

            result_row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
            self.assertIsNone(result_row["request"]["company"])
            self.assertTrue(result_row["evaluation"]["numeric_within_tolerance"])
            self.assertEqual(result_row["evaluation"]["expected_file_behavior"], 1.0)
            self.assertEqual(summary["total_examples"], 1)
            self.assertEqual(summary["expected_file_behavior_mean"], 1.0)
            self.assertIn("index_prep", summary)
            self.assertTrue(summary["index_prep"]["embeddings_available"])


if __name__ == "__main__":
    unittest.main()
