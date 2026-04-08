from __future__ import annotations

import unittest

from gtm_diligence_assistant.models import EvidenceChunk, ReasonedAnswer, ReasonedOperand
from gtm_diligence_assistant.numeric_validation import validate_reasoned_answer


def _chunk(
    citation_id: str,
    source_type: str,
    source_label: str,
    page_number: int,
    snippet: str,
    score: float = 1.0,
) -> EvidenceChunk:
    return EvidenceChunk(
        citation_id=citation_id,
        source_type=source_type,
        source_label=source_label,
        source_path=source_label,
        page_number=page_number,
        snippet=snippet,
        score=score,
    )


class NumericValidationTests(unittest.TestCase):
    def test_addition_and_subtraction_formula_recomputes_correctly(self) -> None:
        evidence = [
            _chunk("local_pdf:acme:1", "local_pdf", "Acme.pdf", 1, "Debt 100 cash 20 lease 10"),
        ]
        answer = ReasonedAnswer(
            answer_kind="number",
            proposed_answer="90",
            formula="debt + lease - cash",
            operands=[
                ReasonedOperand(name="debt", value=100, citation_id="local_pdf:acme:1"),
                ReasonedOperand(name="lease", value=10, citation_id="local_pdf:acme:1"),
                ReasonedOperand(name="cash", value=20, citation_id="local_pdf:acme:1"),
            ],
            explanation="Used debt plus lease minus cash.",
            citation_ids=["local_pdf:acme:1"],
            assumptions=[],
            confidence=0.8,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertTrue(result.matches_proposed_answer)
        self.assertEqual(result.recomputed_answer, "90")
        self.assertEqual(result.issues, [])

    def test_percent_output_format_is_validated(self) -> None:
        evidence = [
            _chunk("local_pdf:prod:1", "local_pdf", "prod.pdf", 1, "growth rate 3.3"),
        ]
        answer = ReasonedAnswer(
            answer_kind="percent",
            proposed_answer="3.30%",
            formula="growth_rate",
            operands=[
                ReasonedOperand(name="growth_rate", kind="percent", value=3.3, citation_id="local_pdf:prod:1"),
            ],
            explanation="The growth rate is 3.30%.",
            citation_ids=["local_pdf:prod:1"],
            assumptions=[],
            confidence=0.7,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertTrue(result.format_ok)
        self.assertTrue(result.matches_proposed_answer)
        self.assertEqual(result.recomputed_answer, "3.30%")

    def test_malformed_formula_fails_safely(self) -> None:
        evidence = [
            _chunk("local_pdf:acme:1", "local_pdf", "Acme.pdf", 1, "Debt 100"),
        ]
        answer = ReasonedAnswer(
            answer_kind="number",
            proposed_answer="100",
            formula="debt +",
            operands=[
                ReasonedOperand(name="debt", value=100, citation_id="local_pdf:acme:1"),
            ],
            explanation="Malformed formula.",
            citation_ids=["local_pdf:acme:1"],
            assumptions=[],
            confidence=0.2,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertFalse(result.matches_proposed_answer)
        self.assertIn("Formula is malformed.", result.issues)

    def test_unknown_operand_reference_fails_safely(self) -> None:
        evidence = [
            _chunk("local_pdf:acme:1", "local_pdf", "Acme.pdf", 1, "Debt 100"),
        ]
        answer = ReasonedAnswer(
            answer_kind="number",
            proposed_answer="100",
            formula="debt + cash",
            operands=[
                ReasonedOperand(name="debt", value=100, citation_id="local_pdf:acme:1"),
            ],
            explanation="Unknown cash operand.",
            citation_ids=["local_pdf:acme:1"],
            assumptions=[],
            confidence=0.2,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertFalse(result.matches_proposed_answer)
        self.assertIn("Formula references unknown operand 'cash'.", result.issues)

    def test_incomplete_unknown_answer_validates_cleanly(self) -> None:
        evidence = [
            _chunk("local_pdf:acme:1", "local_pdf", "Acme.pdf", 1, "Cash 20 but long-term debt not shown"),
        ]
        answer = ReasonedAnswer(
            answer_kind="unknown",
            proposed_answer="",
            formula="",
            operands=[],
            explanation="Cash is present, but long-term debt is missing from the evidence.",
            citation_ids=["local_pdf:acme:1"],
            assumptions=[],
            completion_status="incomplete",
            missing_operands=["long_term_debt"],
            confidence=0.4,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertFalse(result.coverage_complete)
        self.assertEqual(result.issues, [])

    def test_incomplete_unknown_answer_requires_missing_operands(self) -> None:
        evidence = [
            _chunk("local_pdf:acme:1", "local_pdf", "Acme.pdf", 1, "Cash 20"),
        ]
        answer = ReasonedAnswer(
            answer_kind="unknown",
            proposed_answer="",
            formula="",
            operands=[],
            explanation="Evidence is incomplete.",
            citation_ids=["local_pdf:acme:1"],
            assumptions=[],
            completion_status="incomplete",
            missing_operands=[],
            confidence=0.2,
        )
        result = validate_reasoned_answer(answer, evidence)
        self.assertIn("Incomplete answers must list missing_operands.", result.issues)


if __name__ == "__main__":
    unittest.main()
