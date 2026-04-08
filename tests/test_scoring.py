from __future__ import annotations

import unittest

from gtm_diligence_assistant.scoring import evaluate_numeric_answer, parse_answer_value


class ScoringTests(unittest.TestCase):
    def test_parse_answer_value_handles_percent_strings(self) -> None:
        self.assertEqual(parse_answer_value("percent", "28.21%"), 28.21)

    def test_parse_answer_value_handles_number_strings(self) -> None:
        self.assertEqual(parse_answer_value("number", "7682000000"), 7682000000.0)

    def test_evaluate_numeric_answer_marks_exact_match(self) -> None:
        result = evaluate_numeric_answer(
            answer_kind="number",
            final_answer="7682000000",
            expected_kind="number",
            expected_value=7682000000.0,
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["tier"], "exact")
        self.assertTrue(result["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
