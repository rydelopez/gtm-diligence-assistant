from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf
from langchain_core.embeddings import Embeddings

from gtm_diligence_assistant.models import (
    CoverageAssessment,
    DiligenceRequest,
    EvidenceChunk,
    IntakeAnalysis,
    ReasonedAnswer,
    ReasonedOperand,
    RetrievalPlan,
    ValidationResult,
)
from gtm_diligence_assistant.vector_index import LocalVectorIndexManager
from gtm_diligence_assistant.workflow import DiligenceWorkflow


def _create_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeStructuredRunnable:
    def __init__(self, schema, outputs):
        self.schema = schema
        self.outputs = list(outputs)

    def invoke(self, _prompt):
        if not self.outputs:
            raise AssertionError(f"No fake outputs left for schema {self.schema.__name__}")
        payload = self.outputs.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return self.schema.model_validate(payload)


class FakeChatModel:
    def __init__(self, schema_outputs):
        self.schema_outputs = {key: list(value) for key, value in schema_outputs.items()}

    def with_structured_output(self, schema):
        return FakeStructuredRunnable(schema, self.schema_outputs.get(schema.__name__, []))


class KeywordEmbeddings(Embeddings):
    def _encode(self, text: str) -> list[float]:
        lower = text.lower()
        groups = [
            ("debt", "borrowings", "notes", "loan"),
            ("cash", "equivalents", "investments", "securities"),
            ("lease", "obligations", "liabilities"),
            ("revenue", "sales"),
            ("equity", "shareholders", "stockholders"),
            ("income", "earnings", "profit"),
        ]
        return [float(sum(lower.count(term) for term in group)) for group in groups]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def _state_for_direct_call(request: DiligenceRequest) -> dict:
    return {
        "request": request,
        "intake": None,
        "retrieval_plan": None,
        "coverage_assessment": None,
        "evidence": [],
        "evidence_pool": [],
        "citations": [],
        "reasoned_answer": None,
        "validation_result": None,
        "final_response": None,
        "attempt_count": 0,
        "retrieval_iteration": 0,
        "empty_retrieval_pass_count": 0,
        "last_retrieval_added_count": 0,
        "retrieval_stop_reason": None,
        "pages_scanned_count": 0,
        "pages_opened_count": 0,
        "exact_scan_match_count": 0,
        "vector_hits_count": 0,
        "vector_index_used": False,
        "vector_primary_hit_rank": None,
        "vector_retrieval_queries": [],
        "status": "queued",
        "errors": [],
        "seen_query_fingerprints": [],
        "searched_file_query_pairs": [],
        "opened_files": [],
        "deep_read_files": [],
        "full_text_fallback_used": False,
        "coverage_notes": [],
    }


class WorkflowTests(unittest.TestCase):
    def test_intake_infers_company_and_fiscal_year_from_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            (dataroom_root / "Adobe" / "FY 2024").mkdir(parents=True)
            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": None,
                            "fiscal_year": None,
                            "task_type": "net_debt",
                            "required_metrics": ["debt"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [],
                    "ReasonedAnswer": [],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            result = workflow.intake(
                _state_for_direct_call(
                    DiligenceRequest(
                        question="What is Adobe's net debt as of FY 2024?",
                        request_id="inference-test",
                    )
                )
            )

            self.assertEqual(result["status"], "intake_complete")
            self.assertEqual(result["request"].company, "Adobe")
            self.assertEqual(result["request"].fiscal_year, 2024)
            self.assertEqual(result["intake"].company, "Adobe")
            self.assertEqual(result["intake"].fiscal_year, 2024)

    def test_intake_still_short_circuits_when_question_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            (dataroom_root / "Adobe" / "FY 2024").mkdir(parents=True)
            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": None,
                            "fiscal_year": None,
                            "task_type": "unknown",
                            "required_metrics": ["net debt"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [],
                    "ReasonedAnswer": [],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            result = workflow.intake(
                _state_for_direct_call(
                    DiligenceRequest(
                        question="What is net debt?",
                        request_id="ambiguous-test",
                    )
                )
            )

            self.assertEqual(result["status"], "missing_fields")
            self.assertIn("company", result["intake"].missing_fields)
            self.assertIn("fiscal_year", result["intake"].missing_fields)

    def test_explicit_request_identity_wins_over_question_inference(self) -> None:
        workflow = DiligenceWorkflow(FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []}))
        company, fiscal_year = workflow.infer_request_identity(
            DiligenceRequest(
                question="What is Adobe's net debt as of FY 2023?",
                company="Acme",
                fiscal_year=2024,
                request_id="explicit-identity-test",
            )
        )
        self.assertEqual(company, "Acme")
        self.assertEqual(fiscal_year, 2024)

    def test_missing_fields_short_circuits_to_human_review(self) -> None:
        fake_model = FakeChatModel(
            {
                "IntakeAnalysis": [
                    {
                        "company": None,
                        "fiscal_year": None,
                        "task_type": "unknown",
                        "required_metrics": ["net debt"],
                        "missing_fields": ["company", "fiscal_year"],
                        "notes": [],
                    }
                ],
                "CoverageAssessment": [],
                "ReasonedAnswer": [],
            }
        )
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        response = workflow.run_request(DiligenceRequest(question="What is net debt?"))
        self.assertTrue(response.needs_human_review)
        self.assertIn("Missing required fields", " ".join(response.errors))

    def test_intake_ignores_retrievable_metric_missing_fields(self) -> None:
        fake_model = FakeChatModel(
            {
                "IntakeAnalysis": [
                    {
                        "company": "Adobe",
                        "fiscal_year": 2024,
                        "task_type": "net_debt",
                        "required_metrics": ["debt"],
                        "missing_fields": [
                            "Cash and cash equivalents from the FY 2024 10-K",
                            "Long-term debt amount from the FY 2024 10-K",
                        ],
                        "notes": [],
                    }
                ],
                "CoverageAssessment": [],
                "ReasonedAnswer": [],
            }
        )
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        state = _state_for_direct_call(
            DiligenceRequest(
                question="What is Adobe's net debt as of FY 2024?",
                company="Adobe",
                fiscal_year=2024,
                request_id="intake-filter-test",
            )
        )
        result = workflow.intake(state)
        self.assertEqual(result["status"], "intake_complete")
        intake = result["intake"]
        self.assertEqual(intake.missing_fields, [])
        self.assertIn("Cash and cash equivalents from the FY 2024 10-K", intake.required_metrics)
        self.assertIn("Long-term debt amount from the FY 2024 10-K", intake.required_metrics)

    def test_local_only_workflow_uses_reasoning_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(
                pdf_path,
                [
                    (
                        "ACME INC. CONSOLIDATED BALANCE SHEETS (In millions)\n"
                        "Cash and cash equivalents $ 300\n"
                        "Current portion of long-term debt, net $ 1,000\n"
                        "Long-term debt, net of current portion $ 5,000\n"
                    ),
                    "Total operating lease liabilities $ 200\n",
                ],
            )
            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["debt", "cash", "operating lease liabilities"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "found_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["The current evidence is sufficient."],
                        }
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "5900000000",
                            "formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "operands": [
                                {"name": "current_debt", "value": 1000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "long_term_debt", "value": 5000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "lease_liabilities", "value": 200000000, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                                {"name": "cash", "value": 300000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "Net debt is current debt plus long-term debt plus lease liabilities minus cash.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1", "local_pdf:Acme 2024 10-K:2"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.89,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="local-only-test",
                )
            )
            self.assertFalse(response.needs_human_review)
            self.assertEqual(response.final_answer, "5900000000")
            self.assertGreaterEqual(len(response.citations), 2)
            self.assertIn("Net debt", response.explanation)

            _, telemetry = workflow.run_request_with_trace(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="local-only-trace-test",
                )
            )
            self.assertEqual(Path(telemetry["primary_candidate_file"]).name, "Acme 2024 10-K.pdf")
            self.assertTrue(telemetry["primary_file_used"])

    def test_vector_retrieval_loads_precomputed_index_before_exact_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            cache_dir = Path(tmpdir) / "vector_indexes"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            primary_pdf = fy_dir / "Acme 2024 10-K.pdf"
            secondary_pdf = fy_dir / "Acme appendix.pdf"
            _create_pdf(
                primary_pdf,
                [
                    "Company overview.",
                    "Borrowings were $ 5,000 and cash and cash equivalents were $ 300.",
                ],
            )
            _create_pdf(secondary_pdf, ["Borrowings were $ 99, but this appendix should not outrank the 10-K."])

            embeddings = KeywordEmbeddings()
            LocalVectorIndexManager(
                embedding_model=embeddings,
                dataroom_root=dataroom_root,
                cache_dir=cache_dir,
            ).build_fy_index(fy_dir)

            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["borrowings", "cash"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "borrowings - cash",
                            "required_operands": ["borrowings", "cash"],
                            "found_operands": ["borrowings", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["The primary filing provides both operands."],
                        }
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "4700",
                            "formula": "borrowings - cash",
                            "operands": [
                                {"name": "borrowings", "value": 5000, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                                {"name": "cash", "value": 300, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                            ],
                            "explanation": "The primary filing provides both borrowings and cash on the same page.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:2"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.88,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(
                fake_model,
                dataroom_root=dataroom_root,
                embedding_model=embeddings,
                vector_index_cache_dir=cache_dir,
            )
            response, telemetry = workflow.run_request_with_trace(
                DiligenceRequest(
                    question="What are borrowings minus cash for Acme FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="vector-primary-test",
                )
            )

            self.assertFalse(response.needs_human_review)
            self.assertEqual(response.final_answer, "4700")
            self.assertTrue(telemetry["vector_index_used"])
            self.assertGreaterEqual(telemetry["vector_hits_count"], 1)
            self.assertEqual(telemetry["vector_primary_hit_rank"], 1)
            self.assertEqual(telemetry["exact_scan_match_count"], 0)
            self.assertIn("What are borrowings minus cash for Acme FY 2024?", telemetry["vector_retrieval_queries"])

    def test_local_open_loop_second_pass_targets_missing_operands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            preferred_pdf = fy_dir / "Acme 2024 10-K.pdf"
            secondary_pdf = fy_dir / "Acme debt note.pdf"
            _create_pdf(preferred_pdf, ["Cash and cash equivalents $ 300\nCurrent portion of long-term debt $ 1,000"])
            _create_pdf(secondary_pdf, ["Noncurrent borrowings were $ 5,000 and future lease obligations were $ 200."])

            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["debt", "cash", "lease obligations"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "found_operands": ["current_debt", "cash"],
                            "missing_operands": ["long_term_debt", "lease_liabilities"],
                            "follow_up_local_queries": ["noncurrent borrowings future lease obligations"],
                            "enough_evidence_to_answer": False,
                            "reasoning_notes": ["Need a second pass to find long-term debt and lease obligations."],
                        },
                        {
                            "candidate_formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "found_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["The combined evidence now supports the full formula."],
                        },
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "5900000000",
                            "formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "operands": [
                                {"name": "current_debt", "value": 1000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "long_term_debt", "value": 5000000000, "citation_id": "local_pdf:Acme debt note:1"},
                                {"name": "lease_liabilities", "value": 200000000, "citation_id": "local_pdf:Acme debt note:1"},
                                {"name": "cash", "value": 300000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "The second retrieval pass found noncurrent borrowings and lease obligations, completing the formula.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1", "local_pdf:Acme debt note:1"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.86,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="local-open-loop-test",
                )
            )
            self.assertFalse(response.needs_human_review)
            self.assertEqual(response.final_answer, "5900000000")
            self.assertEqual({citation.source_label for citation in response.citations}, {"Acme 2024 10-K.pdf", "Acme debt note.pdf"})

    def test_primary_file_is_searched_before_secondary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            primary_pdf = fy_dir / "Acme 2024 10-K.pdf"
            secondary_pdf = fy_dir / "Acme appendix.pdf"
            _create_pdf(primary_pdf, ["Cash and cash equivalents $ 300\nLong-term debt $ 5,000"])
            _create_pdf(secondary_pdf, ["This file should not be needed."])

            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["cash", "long-term debt"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "long_term_debt - cash",
                            "required_operands": ["long_term_debt", "cash"],
                            "found_operands": ["long_term_debt", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["Primary filing is sufficient."],
                        }
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "4700",
                            "formula": "long_term_debt - cash",
                            "operands": [
                                {"name": "long_term_debt", "value": 5000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "cash", "value": 300, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "Both operands came from the primary 10-K.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.9,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            _, telemetry = workflow.run_request_with_trace(
                DiligenceRequest(
                    question="What is Acme's long-term debt minus cash?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="primary-first-test",
                )
            )
            opened_names = [Path(path).name for path in telemetry["opened_files"]]
            self.assertIn("Acme 2024 10-K.pdf", opened_names)
            self.assertNotIn("Acme appendix.pdf", opened_names)

    def test_primary_file_full_text_fallback_adds_evidence_when_search_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            primary_pdf = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(primary_pdf, ["Noncurrent borrowings were $ 5,000 and cash was $ 300."])

            fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            state = _state_for_direct_call(
                DiligenceRequest(question="What is covenant leverage?", company="Acme", fiscal_year=2024)
            )
            state["retrieval_plan"] = RetrievalPlan(
                candidate_files=[str(primary_pdf)],
                primary_candidate_file=str(primary_pdf),
                secondary_candidate_files=[],
                search_queries=["term loan covenant"],
                active_local_queries=["term loan covenant"],
            )

            result = workflow.retrieve_local_evidence(state)
            self.assertTrue(result["full_text_fallback_used"])
            self.assertTrue(result["evidence_pool"])
            self.assertIn(str(primary_pdf), result["opened_files"])

    def test_retrieve_local_evidence_finds_adobe_debt_note_from_primary_file(self) -> None:
        pdf_path = Path("dataroom/Adobe/FY 2024/adbe-10k-fy24-final.pdf")
        if not pdf_path.exists():
            self.skipTest("Adobe 10-K fixture not present.")

        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        state = _state_for_direct_call(
            DiligenceRequest(
                question="What is Adobe's net debt as of the FY 2024 10-K?",
                company="Adobe",
                fiscal_year=2024,
            )
        )
        state["intake"] = IntakeAnalysis(
            company="Adobe",
            fiscal_year=2024,
            task_type="net_debt",
            required_metrics=["total debt", "cash and cash equivalents", "operating lease liabilities"],
            missing_fields=[],
            notes=[],
        )
        state["retrieval_plan"] = RetrievalPlan(
            candidate_files=[str(pdf_path)],
            primary_candidate_file=str(pdf_path),
            secondary_candidate_files=[],
            search_queries=["Adobe net debt"],
            active_local_queries=["Adobe net debt"],
        )

        result = workflow.retrieve_local_evidence(state)
        self.assertIn("local_pdf:adbe-10k-fy24-final:88", {chunk.citation_id for chunk in result["evidence_pool"]})

    def test_incomplete_coverage_returns_partial_unknown_with_missing_operands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(pdf_path, ["Cash and cash equivalents $ 300\nCurrent portion of long-term debt $ 1,000"])

            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["debt", "cash"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "(current_debt + long_term_debt) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "cash"],
                            "found_operands": ["current_debt", "cash"],
                            "missing_operands": ["long_term_debt"],
                            "follow_up_local_queries": ["noncurrent borrowings"],
                            "enough_evidence_to_answer": False,
                            "reasoning_notes": ["Long-term debt is still missing."],
                        },
                        {
                            "candidate_formula": "(current_debt + long_term_debt) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "cash"],
                            "found_operands": ["current_debt", "cash"],
                            "missing_operands": ["long_term_debt"],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": False,
                            "reasoning_notes": ["The second pass still did not find long-term debt."],
                        },
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "unknown",
                            "proposed_answer": "",
                            "formula": "",
                            "operands": [],
                            "explanation": "Cash and current debt were found, but long-term debt is still missing from the evidence.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1"],
                            "assumptions": [],
                            "completion_status": "incomplete",
                            "missing_operands": ["long_term_debt"],
                            "confidence": 0.35,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="incomplete-coverage-test",
                )
            )
            self.assertTrue(response.needs_human_review)
            self.assertIsNone(response.final_answer)
            self.assertEqual(response.answer_kind, "unknown")
            self.assertIn("Missing operands", response.explanation)

    def test_graph_handles_non_net_debt_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(pdf_path, ["Revenue was $ 100 and cost of revenue was $ 40."])

            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "unknown",
                            "required_metrics": ["revenue", "cost of revenue"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "revenue - cost_of_revenue",
                            "required_operands": ["revenue", "cost_of_revenue"],
                            "found_operands": ["revenue", "cost_of_revenue"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["The evidence supports a simple subtraction formula."],
                        }
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "60",
                            "formula": "revenue - cost_of_revenue",
                            "operands": [
                                {"name": "revenue", "value": 100, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "cost_of_revenue", "value": 40, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "The answer uses a general subtraction formula, not a net-debt-specific rule.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.88,
                        }
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is revenue minus cost of revenue for FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="generic-formula-test",
                )
            )
            self.assertFalse(response.needs_human_review)
            self.assertEqual(response.final_answer, "60")

    def test_arithmetic_mismatch_triggers_one_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(pdf_path, ["Cash and cash equivalents $ 300\nCurrent portion of long-term debt $ 1,000\nLong-term debt $ 5,000", "Total operating lease liabilities $ 200"])
            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["debt", "cash", "operating lease liabilities"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "found_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["Evidence is sufficient."],
                        }
                    ],
                    "ReasonedAnswer": [
                        {
                            "answer_kind": "number",
                            "proposed_answer": "5800000000",
                            "formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "operands": [
                                {"name": "current_debt", "value": 1000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "long_term_debt", "value": 5000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "lease_liabilities", "value": 200000000, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                                {"name": "cash", "value": 300000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "First pass had the wrong arithmetic.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1", "local_pdf:Acme 2024 10-K:2"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.6,
                        },
                        {
                            "answer_kind": "number",
                            "proposed_answer": "5900000000",
                            "formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "operands": [
                                {"name": "current_debt", "value": 1000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "long_term_debt", "value": 5000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                                {"name": "lease_liabilities", "value": 200000000, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                                {"name": "cash", "value": 300000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                            ],
                            "explanation": "Second pass fixed the arithmetic.",
                            "citation_ids": ["local_pdf:Acme 2024 10-K:1", "local_pdf:Acme 2024 10-K:2"],
                            "assumptions": [],
                            "completion_status": "complete",
                            "missing_operands": [],
                            "confidence": 0.8,
                        },
                    ],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="retry-test",
                )
            )
            self.assertFalse(response.needs_human_review)
            self.assertEqual(response.final_answer, "5900000000")

    def test_repeated_mismatch_escalates_to_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(pdf_path, ["Cash and cash equivalents $ 300\nCurrent portion of long-term debt $ 1,000\nLong-term debt $ 5,000", "Total operating lease liabilities $ 200"])
            bad_reasoning = {
                "answer_kind": "number",
                "proposed_answer": "5800000000",
                "formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                "operands": [
                    {"name": "current_debt", "value": 1000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                    {"name": "long_term_debt", "value": 5000000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                    {"name": "lease_liabilities", "value": 200000000, "citation_id": "local_pdf:Acme 2024 10-K:2"},
                    {"name": "cash", "value": 300000000, "citation_id": "local_pdf:Acme 2024 10-K:1"},
                ],
                "explanation": "Still wrong.",
                "citation_ids": ["local_pdf:Acme 2024 10-K:1", "local_pdf:Acme 2024 10-K:2"],
                "assumptions": [],
                "completion_status": "complete",
                "missing_operands": [],
                "confidence": 0.5,
            }
            fake_model = FakeChatModel(
                {
                    "IntakeAnalysis": [
                        {
                            "company": "Acme",
                            "fiscal_year": 2024,
                            "task_type": "net_debt",
                            "required_metrics": ["debt", "cash", "operating lease liabilities"],
                            "missing_fields": [],
                            "notes": [],
                        }
                    ],
                    "CoverageAssessment": [
                        {
                            "candidate_formula": "(current_debt + long_term_debt + lease_liabilities) - cash",
                            "required_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "found_operands": ["current_debt", "long_term_debt", "lease_liabilities", "cash"],
                            "missing_operands": [],
                            "follow_up_local_queries": [],
                            "enough_evidence_to_answer": True,
                            "reasoning_notes": ["Evidence is sufficient."],
                        }
                    ],
                    "ReasonedAnswer": [bad_reasoning, bad_reasoning],
                }
            )
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            response = workflow.run_request(
                DiligenceRequest(
                    question="What is Acme's net debt as of FY 2024?",
                    company="Acme",
                    fiscal_year=2024,
                    request_id="repeat-mismatch-test",
                )
            )
            self.assertTrue(response.needs_human_review)
            self.assertIn("Proposed answer does not match the recomputed formula result", response.explanation)

    def test_verify_answer_retries_when_citations_are_invalid(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        verify_state = {
            "request": DiligenceRequest(question="What is the answer?", company="Acme", fiscal_year=2024, request_id="verify-test"),
            "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=[], missing_fields=[], notes=[]),
            "retrieval_plan": None,
            "coverage_assessment": CoverageAssessment(),
            "evidence": [],
            "evidence_pool": [],
            "citations": [],
            "reasoned_answer": ReasonedAnswer(
                answer_kind="number",
                proposed_answer="10",
                formula="value",
                operands=[ReasonedOperand(name="value", value=10, citation_id="local_pdf:missing:2")],
                explanation="First draft with the wrong citation.",
                citation_ids=["local_pdf:missing:2"],
                assumptions=[],
                completion_status="complete",
                missing_operands=[],
                confidence=0.72,
            ),
            "validation_result": ValidationResult(
                recomputed_answer="10",
                matches_proposed_answer=True,
                format_ok=True,
                citation_ids_valid=False,
                coverage_complete=True,
                issues=["Operand citation 'local_pdf:missing:2' was not found in retrieved evidence."],
            ),
            "final_response": None,
            "attempt_count": 0,
            "retrieval_iteration": 1,
            "empty_retrieval_pass_count": 0,
            "last_retrieval_added_count": 0,
            "retrieval_stop_reason": None,
            "status": "reasoning_complete",
            "errors": [],
            "seen_query_fingerprints": [],
            "searched_file_query_pairs": [],
            "opened_files": [],
            "deep_read_files": [],
            "full_text_fallback_used": False,
            "coverage_notes": [],
        }

        first_pass = workflow.verify_answer(verify_state)
        self.assertEqual(first_pass["status"], "retry_reasoning")
        self.assertEqual(first_pass["attempt_count"], 1)

    def test_reasoning_prompt_uses_capped_evidence_pack(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        captured: dict[str, str] = {}

        def _invoke(prompt: str) -> ReasonedAnswer:
            captured["prompt"] = prompt
            return ReasonedAnswer(
                answer_kind="number",
                proposed_answer="10",
                formula="value",
                operands=[ReasonedOperand(name="value", value=10, citation_id="local_pdf:doc:1")],
                explanation="Used the first chunk.",
                citation_ids=["local_pdf:doc:1"],
                assumptions=[],
                completion_status="complete",
                missing_operands=[],
                confidence=0.6,
            )

        workflow.reasoning_llm = mock.Mock(invoke=_invoke)
        evidence = [
            EvidenceChunk(
                citation_id=f"local_pdf:doc:{index}",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=index,
                snippet=f"Snippet {index}",
                score=1.0,
            )
            for index in range(1, 11)
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is the answer?", company="Acme", fiscal_year=2024, request_id="prompt-cap-test")
        )
        state["evidence"] = evidence
        state["evidence_pool"] = evidence
        state["citations"] = [chunk.to_citation() for chunk in evidence]
        state["status"] = "coverage_ready"

        workflow.reason_with_evidence(state)
        prompt = captured["prompt"]
        self.assertIn("local_pdf:doc:1", prompt)
        self.assertIn("local_pdf:doc:10", prompt)

    def test_assess_evidence_coverage_routes_with_deduped_follow_up_queries(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        workflow.coverage_llm = mock.Mock(
            invoke=mock.Mock(
                return_value=CoverageAssessment(
                    candidate_formula="value_a + value_b",
                    required_operands=["value_a", "value_b"],
                    found_operands=["value_a"],
                    missing_operands=["value_b"],
                    follow_up_local_queries=["value_b note", "value_b note", "duplicate query"],
                    enough_evidence_to_answer=False,
                    reasoning_notes=["Need another local search for value_b."],
                )
            )
        )
        evidence = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="value_a is 10",
                score=1.0,
            )
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is value_a plus value_b?", company="Acme", fiscal_year=2024)
        )
        state.update(
            {
                "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=["value_a", "value_b"], missing_fields=[], notes=[]),
                "retrieval_plan": RetrievalPlan(
                    candidate_files=["doc.pdf"],
                    search_queries=["value_a", "value_b"],
                    active_local_queries=["value_a", "value_b"],
                ),
                "evidence": evidence,
                "evidence_pool": evidence,
                "citations": [chunk.to_citation() for chunk in evidence],
                "last_retrieval_added_count": 1,
                "seen_query_fingerprints": [workflow._query_fingerprint("duplicate query")],
            }
        )

        result = workflow.assess_evidence_coverage(state)
        self.assertEqual(result["status"], "needs_more_evidence")
        self.assertEqual(result["retrieval_iteration"], 1)
        self.assertIn("value_b note", result["retrieval_plan"].active_local_queries)

    def test_assess_evidence_coverage_stops_when_file_query_pairs_are_exhausted(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        workflow.coverage_llm = mock.Mock(
            invoke=mock.Mock(
                return_value=CoverageAssessment(
                    candidate_formula="value_a + value_b",
                    required_operands=["value_a", "value_b"],
                    found_operands=["value_a"],
                    missing_operands=["value_b"],
                    follow_up_local_queries=["value_b note"],
                    enough_evidence_to_answer=False,
                    reasoning_notes=["Still missing value_b."],
                )
            )
        )
        evidence = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="value_a is 10",
                score=1.0,
            )
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is value_a plus value_b?", company="Acme", fiscal_year=2024)
        )
        state.update(
            {
                "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=["value_a", "value_b"], missing_fields=[], notes=[]),
                "retrieval_plan": RetrievalPlan(
                    candidate_files=["doc.pdf"],
                    search_queries=["value_a", "value_b"],
                    active_local_queries=["value_b note"],
                ),
                "evidence": evidence,
                "evidence_pool": evidence,
                "citations": [chunk.to_citation() for chunk in evidence],
                "last_retrieval_added_count": 1,
                "seen_query_fingerprints": [
                    workflow._query_fingerprint("value_b note"),
                    workflow._query_fingerprint("value b"),
                ],
                "searched_file_query_pairs": [workflow._file_query_pair_key("doc.pdf", "value_b note")],
            }
        )

        result = workflow.assess_evidence_coverage(state)
        self.assertEqual(result["status"], "coverage_ready")
        self.assertIn("all file/query combinations were exhausted", " ".join(result["coverage_notes"]))

    def test_assess_evidence_coverage_continues_before_empty_pass_budget_is_reached(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        workflow.coverage_llm = mock.Mock(
            invoke=mock.Mock(
                return_value=CoverageAssessment(
                    candidate_formula="value_a + value_b",
                    required_operands=["value_a", "value_b"],
                    found_operands=["value_a"],
                    missing_operands=["value_b"],
                    follow_up_local_queries=["value_b note"],
                    enough_evidence_to_answer=False,
                    reasoning_notes=["Still missing value_b."],
                )
            )
        )
        evidence = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="value_a is 10",
                score=1.0,
            )
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is value_a plus value_b?", company="Acme", fiscal_year=2024)
        )
        state.update(
            {
                "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=["value_a", "value_b"], missing_fields=[], notes=[]),
                "retrieval_plan": RetrievalPlan(
                    candidate_files=["doc.pdf", "support.pdf"],
                    search_queries=["value_b note"],
                    active_local_queries=["value_b note"],
                ),
                "evidence": evidence,
                "evidence_pool": evidence,
                "citations": [chunk.to_citation() for chunk in evidence],
                "last_retrieval_added_count": 0,
                "empty_retrieval_pass_count": 3,
            }
        )

        result = workflow.assess_evidence_coverage(state)
        self.assertEqual(result["status"], "needs_more_evidence")
        self.assertIsNone(result["retrieval_stop_reason"])
        self.assertEqual(result["retrieval_iteration"], 1)

    def test_assess_evidence_coverage_stops_once_empty_pass_budget_is_reached(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        workflow.coverage_llm = mock.Mock(
            invoke=mock.Mock(
                return_value=CoverageAssessment(
                    candidate_formula="value_a + value_b",
                    required_operands=["value_a", "value_b"],
                    found_operands=["value_a"],
                    missing_operands=["value_b"],
                    follow_up_local_queries=["value_b note"],
                    enough_evidence_to_answer=False,
                    reasoning_notes=["Still missing value_b."],
                )
            )
        )
        evidence = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="value_a is 10",
                score=1.0,
            )
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is value_a plus value_b?", company="Acme", fiscal_year=2024)
        )
        state.update(
            {
                "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=["value_a", "value_b"], missing_fields=[], notes=[]),
                "retrieval_plan": RetrievalPlan(
                    candidate_files=["doc.pdf", "support.pdf"],
                    search_queries=["value_b note"],
                    active_local_queries=["value_b note"],
                ),
                "evidence": evidence,
                "evidence_pool": evidence,
                "citations": [chunk.to_citation() for chunk in evidence],
                "last_retrieval_added_count": 0,
                "empty_retrieval_pass_count": 4,
            }
        )

        result = workflow.assess_evidence_coverage(state)
        self.assertEqual(result["status"], "coverage_ready")
        self.assertEqual(result["retrieval_stop_reason"], "empty_pass_budget_reached")
        self.assertIn("empty-pass budget of 4", " ".join(result["coverage_notes"]))

    def test_assess_evidence_coverage_stops_when_total_pass_budget_is_reached(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        workflow.coverage_llm = mock.Mock(
            invoke=mock.Mock(
                return_value=CoverageAssessment(
                    candidate_formula="value_a + value_b",
                    required_operands=["value_a", "value_b"],
                    found_operands=["value_a"],
                    missing_operands=["value_b"],
                    follow_up_local_queries=["value_b note"],
                    enough_evidence_to_answer=False,
                    reasoning_notes=["Still missing value_b."],
                )
            )
        )
        evidence = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="value_a is 10",
                score=1.0,
            )
        ]
        state = _state_for_direct_call(
            DiligenceRequest(question="What is value_a plus value_b?", company="Acme", fiscal_year=2024)
        )
        state.update(
            {
                "intake": IntakeAnalysis(company="Acme", fiscal_year=2024, task_type="unknown", required_metrics=["value_a", "value_b"], missing_fields=[], notes=[]),
                "retrieval_plan": RetrievalPlan(
                    candidate_files=["doc.pdf", "support.pdf"],
                    search_queries=["value_b note"],
                    active_local_queries=["value_b note"],
                ),
                "evidence": evidence,
                "evidence_pool": evidence,
                "citations": [chunk.to_citation() for chunk in evidence],
                "retrieval_iteration": 9,
                "empty_retrieval_pass_count": 0,
                "last_retrieval_added_count": 0,
            }
        )

        result = workflow.assess_evidence_coverage(state)
        self.assertEqual(result["status"], "coverage_ready")
        self.assertEqual(result["retrieval_stop_reason"], "pass_budget_reached")
        self.assertIn("loop budget of 10 passes", " ".join(result["coverage_notes"]))

    def test_select_prompt_evidence_prefers_operand_coverage(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        evidence_pool = [
            EvidenceChunk(
                citation_id="local_pdf:doc:1",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=1,
                snippet="General overview page",
                score=10.0,
            ),
            EvidenceChunk(
                citation_id="local_pdf:doc:2",
                source_type="local_pdf",
                source_label="doc.pdf",
                source_path="doc.pdf",
                page_number=2,
                snippet="Long-term borrowings were 5,000",
                score=0.2,
            ),
        ]
        selected = workflow._select_prompt_evidence(
            evidence_pool,
            CoverageAssessment(
                candidate_formula="debt - cash",
                required_operands=["long_term_borrowings", "cash"],
                found_operands=["long_term_borrowings"],
                missing_operands=["cash"],
                follow_up_local_queries=[],
                enough_evidence_to_answer=False,
                reasoning_notes=[],
            ),
            limit=1,
        )
        self.assertEqual([chunk.citation_id for chunk in selected], ["local_pdf:doc:2"])

    def test_select_prompt_evidence_prefers_primary_file_when_operands_overlap(self) -> None:
        fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
        workflow = DiligenceWorkflow(fake_model, dataroom_root="dataroom")
        evidence_pool = [
            EvidenceChunk(
                citation_id="local_pdf:secondary:4",
                source_type="local_pdf",
                source_label="appendix.pdf",
                source_path="/tmp/appendix.pdf",
                page_number=4,
                snippet="cash and cash equivalents were 300",
                score=1.0,
            ),
            EvidenceChunk(
                citation_id="local_pdf:primary:7",
                source_type="local_pdf",
                source_label="10k.pdf",
                source_path="/tmp/10k.pdf",
                page_number=7,
                snippet="cash and cash equivalents were 300",
                score=0.8,
            ),
        ]
        selected = workflow._select_prompt_evidence(
            evidence_pool,
            CoverageAssessment(
                candidate_formula="cash",
                required_operands=["cash_and_cash_equivalents"],
                found_operands=["cash_and_cash_equivalents"],
                missing_operands=[],
                follow_up_local_queries=[],
                enough_evidence_to_answer=True,
                reasoning_notes=[],
            ),
            plan=RetrievalPlan(
                candidate_files=["/tmp/10k.pdf", "/tmp/appendix.pdf"],
                primary_candidate_file="/tmp/10k.pdf",
                secondary_candidate_files=["/tmp/appendix.pdf"],
                search_queries=[],
                active_local_queries=[],
            ),
            limit=1,
        )
        self.assertEqual([chunk.citation_id for chunk in selected], ["local_pdf:primary:7"])

    def test_plan_sources_includes_all_ranked_files_without_grading_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            preferred_pdf = fy_dir / "Acme 2024 10-K.pdf"
            second_pdf = fy_dir / "Acme annual report.pdf"
            third_pdf = fy_dir / "Acme debt appendix.pdf"
            fourth_pdf = fy_dir / "Acme investor deck.pdf"
            for pdf in [preferred_pdf, second_pdf, third_pdf, fourth_pdf]:
                _create_pdf(pdf, ["Example page"])

            fake_model = FakeChatModel({"IntakeAnalysis": [], "CoverageAssessment": [], "ReasonedAnswer": []})
            workflow = DiligenceWorkflow(fake_model, dataroom_root=dataroom_root)
            state = _state_for_direct_call(
                DiligenceRequest(question="What is Acme's net debt?", company="Acme", fiscal_year=2024, request_id="plan-test")
            )
            state["intake"] = IntakeAnalysis(
                company="Acme",
                fiscal_year=2024,
                task_type="net_debt",
                required_metrics=["debt", "cash"],
                missing_fields=[],
                notes=[],
            )

            result = workflow.plan_sources(state)
            candidate_names = [Path(path).name for path in result["retrieval_plan"].candidate_files]
            self.assertEqual(candidate_names[0], "Acme 2024 10-K.pdf")
            self.assertEqual(len(candidate_names), 4)
            self.assertEqual(Path(result["retrieval_plan"].primary_candidate_file).name, "Acme 2024 10-K.pdf")
            self.assertEqual(
                [Path(path).name for path in result["retrieval_plan"].secondary_candidate_files],
                ["Acme annual report.pdf", "Acme debt appendix.pdf", "Acme investor deck.pdf"],
            )


if __name__ == "__main__":
    unittest.main()
