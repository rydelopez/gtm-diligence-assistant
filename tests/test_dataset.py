from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gtm_diligence_assistant.dataset import load_request_from_dataset, request_from_dataset_record


class DatasetTests(unittest.TestCase):
    def test_request_from_dataset_record_uses_only_inputs(self) -> None:
        record = {
            "id": "example-1",
            "inputs": {
                "question": "What is net debt?",
                "request_id": "example-1",
            },
            "outputs": {
                "expected_kind": "number",
                "expected_value": 100,
                "correct_company_folder": "/Acme",
            },
            "metadata": {"qid": 1, "company": "Acme", "fiscal_year": 2024, "correct_fy_folder": "/Acme/FY 2024"},
        }
        request = request_from_dataset_record(record)
        self.assertEqual(request.request_id, "example-1")
        self.assertIsNone(request.qid)
        self.assertIsNone(request.company)
        self.assertIsNone(request.fiscal_year)

    def test_load_request_from_dataset_finds_qid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "examples.jsonl"
            record = {
                "id": "example-2",
                "inputs": {
                    "question": "What is Adobe's net debt?",
                    "request_id": "example-2",
                },
                "outputs": {"expected_kind": "number", "expected_value": -1830000000},
                "metadata": {"qid": 9, "company": "Adobe", "fiscal_year": 2024},
            }
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            request = load_request_from_dataset(dataset_path, 9)
            self.assertEqual(request.question, "What is Adobe's net debt?")
            self.assertIsNone(request.company)


if __name__ == "__main__":
    unittest.main()
