from __future__ import annotations

import unittest

from gtm_diligence_assistant.task_planning import build_local_queries, expand_missing_operand_queries


class TaskPlanningTests(unittest.TestCase):
    def test_build_local_queries_adds_finance_aliases(self) -> None:
        queries = build_local_queries(
            task_type="net_debt",
            question="What is Acme's net debt?",
            metrics=["cash and cash equivalents", "long-term debt"],
        )
        combined = " || ".join(queries)
        self.assertIn("cash equivalents", combined)
        self.assertIn("noncurrent borrowings", combined)

    def test_expand_missing_operand_queries_adds_lexical_variants(self) -> None:
        queries = expand_missing_operand_queries(["long_term_debt", "lease_liabilities"])
        combined = " || ".join(queries)
        self.assertIn("noncurrent borrowings", combined)
        self.assertIn("lease obligations", combined)


if __name__ == "__main__":
    unittest.main()
