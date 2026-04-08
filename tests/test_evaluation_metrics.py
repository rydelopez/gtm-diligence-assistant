from __future__ import annotations

import unittest

from gtm_diligence_assistant.evaluation_metrics import compute_file_metrics, expected_files_from_record
from gtm_diligence_assistant.models import Citation, DiligenceResponse


class EvaluationMetricsTests(unittest.TestCase):
    def test_expected_files_are_grader_only_metadata(self) -> None:
        record = {
            "inputs": {"question": "What is net debt?"},
            "outputs": {"correct_any_of_files": ["Acme 2024 10-K.pdf"]},
            "metadata": {"correct_any_of_files": ["Acme annual report.pdf"]},
        }
        self.assertEqual(
            expected_files_from_record(record),
            ["Acme 2024 10-K.pdf", "Acme annual report.pdf"],
        )

    def test_compute_file_metrics_scores_full_alignment_as_one(self) -> None:
        response = DiligenceResponse(
            final_answer="10",
            answer_kind="number",
            explanation="Used the 10-K.",
            citations=[
                Citation(
                    citation_id="local_pdf:acme:1",
                    source_type="local_pdf",
                    source_label="Acme 2024 10-K.pdf",
                    source_path="/tmp/Acme 2024 10-K.pdf",
                    page_number=1,
                )
            ],
            confidence=0.5,
            needs_human_review=False,
            errors=[],
        )
        metrics = compute_file_metrics(
            response,
            {
                "opened_files": ["/tmp/Acme 2024 10-K.pdf"],
                "primary_candidate_file": "/tmp/Acme 2024 10-K.pdf",
                "primary_file_used": True,
            },
            ["Acme 2024 10-K.pdf"],
        )
        self.assertTrue(metrics["opened_expected_file"])
        self.assertTrue(metrics["primary_file_alignment"])
        self.assertEqual(metrics["expected_file_behavior"], 1.0)

    def test_compute_file_metrics_scores_opened_without_alignment_as_half(self) -> None:
        response = DiligenceResponse(
            final_answer="10",
            answer_kind="number",
            explanation="Opened a candidate file but cited something else.",
            citations=[],
            confidence=0.5,
            needs_human_review=False,
            errors=[],
        )
        metrics = compute_file_metrics(
            response,
            {
                "opened_files": ["/tmp/Acme 2024 10-K.pdf"],
                "primary_candidate_file": "/tmp/Other file.pdf",
                "primary_file_used": False,
            },
            ["Acme 2024 10-K.pdf"],
        )
        self.assertTrue(metrics["opened_expected_file"])
        self.assertFalse(metrics["primary_file_alignment"])
        self.assertEqual(metrics["expected_file_behavior"], 0.5)

    def test_compute_file_metrics_scores_non_match_as_zero(self) -> None:
        response = DiligenceResponse(
            final_answer="10",
            answer_kind="number",
            explanation="No aligned file behavior.",
            citations=[],
            confidence=0.5,
            needs_human_review=False,
            errors=[],
        )
        metrics = compute_file_metrics(
            response,
            {
                "opened_files": ["/tmp/Other file.pdf"],
                "primary_candidate_file": "/tmp/Other file.pdf",
                "primary_file_used": False,
            },
            ["Acme 2024 10-K.pdf"],
        )
        self.assertFalse(metrics["opened_expected_file"])
        self.assertFalse(metrics["primary_file_alignment"])
        self.assertEqual(metrics["expected_file_behavior"], 0.0)


if __name__ == "__main__":
    unittest.main()
