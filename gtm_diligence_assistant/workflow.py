from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from typing_extensions import NotRequired, TypedDict

from .models import (
    Citation,
    CoverageAssessment,
    DiligenceRequest,
    DiligenceResponse,
    EvidenceChunk,
    IntakeAnalysis,
    ReasonedAnswer,
    RetrievalPlan,
    ValidationResult,
)
from .numeric_validation import validate_reasoned_answer
from .scoring import parse_answer_value
from .task_planning import (
    build_local_queries,
    expand_search_terms,
    expand_missing_operand_queries,
    merged_required_metrics,
    normalize_task_type,
    request_identity_missing_fields,
)
from .tools import get_full_pdf_text, read_pdf_pages, scan_pdf_pages, search_document_pages
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR, LocalVectorIndexManager


# Hardcoded retrieval budgets are slightly higher to avoid premature stops on harder
# local-only cases that need a few more passes before genuinely new evidence appears.
MAX_RETRIEVAL_PASSES = 10
MAX_EMPTY_PASSES = 4
PRIMARY_SEARCH_TOP_K = 5
PROMPT_EVIDENCE_LIMIT = 12
ASSESSMENT_EVIDENCE_LIMIT = 12
PRIMARY_PAGE_WINDOW = (1, 2)
RECURSION_LIMIT = 150
FULL_TEXT_FALLBACK_FILES = 1
SECONDARY_HEURISTIC_TOP_K = 8
VECTOR_SEARCH_TOP_K = 8
VECTOR_QUERY_LIMIT = 10
VECTOR_HIT_LIMIT = 12
COMPANY_NAME_STOP_WORDS = {
    "and",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "plc",
    "the",
}


class DiligenceState(TypedDict):
    request: DiligenceRequest
    intake: IntakeAnalysis | None
    retrieval_plan: RetrievalPlan | None
    coverage_assessment: CoverageAssessment | None
    evidence: list[EvidenceChunk]
    evidence_pool: list[EvidenceChunk]
    citations: list[Citation]
    reasoned_answer: ReasonedAnswer | None
    validation_result: ValidationResult | None
    final_response: DiligenceResponse | None
    attempt_count: int
    retrieval_iteration: int
    empty_retrieval_pass_count: int
    last_retrieval_added_count: int
    retrieval_stop_reason: str | None
    pages_scanned_count: int
    pages_opened_count: int
    exact_scan_match_count: int
    vector_hits_count: int
    vector_index_used: bool
    vector_primary_hit_rank: int | None
    vector_retrieval_queries: list[str]
    status: str
    errors: list[str]
    seen_query_fingerprints: list[str]
    searched_file_query_pairs: list[str]
    opened_files: list[str]
    deep_read_files: list[str]
    full_text_fallback_used: bool
    coverage_notes: list[str]
    verification_notes: NotRequired[list[str]]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _unique_errors(errors: list[str]) -> list[str]:
    seen: list[str] = []
    for error in errors:
        if error and error not in seen:
            seen.append(error)
    return seen


class DiligenceWorkflow:
    def __init__(
        self,
        llm: Any,
        dataroom_root: str | Path = "dataroom",
        embedding_model: Any | None = None,
        vector_index_cache_dir: str | Path = DEFAULT_VECTOR_INDEX_CACHE_DIR,
    ) -> None:
        self.llm = llm
        self.intake_llm = llm.with_structured_output(IntakeAnalysis)
        self.coverage_llm = llm.with_structured_output(CoverageAssessment)
        self.reasoning_llm = llm.with_structured_output(ReasonedAnswer)
        self.dataroom_root = Path(dataroom_root)
        self.embedding_model = embedding_model
        self.vector_index_cache_dir = Path(vector_index_cache_dir)
        self.index_manager = (
            LocalVectorIndexManager(
                embedding_model=embedding_model,
                dataroom_root=self.dataroom_root,
                cache_dir=self.vector_index_cache_dir,
            )
            if embedding_model is not None
            else None
        )
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph().compile(
            checkpointer=self.checkpointer,
            name="gtm_diligence_assistant_local_only_v1",
        )
        self._runner = self._build_runner()

    def _company_tokens(self, value: str) -> set[str]:
        tokens = {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}
        filtered = {token for token in tokens if token not in COMPANY_NAME_STOP_WORDS}
        return filtered or tokens

    def _infer_company_from_question(self, question: str) -> str | None:
        if not question.strip() or not self.dataroom_root.exists():
            return None

        normalized_question = _normalize_name(question)
        candidates = [path for path in self.dataroom_root.iterdir() if path.is_dir()]
        substring_matches = [
            candidate.name
            for candidate in candidates
            if _normalize_name(candidate.name) and _normalize_name(candidate.name) in normalized_question
        ]
        if substring_matches:
            return sorted(substring_matches, key=lambda value: (-len(_normalize_name(value)), value))[0]

        question_tokens = {token for token in re.findall(r"[a-z0-9]+", question.lower()) if token}
        best_name: str | None = None
        best_score: tuple[float, int, int] | None = None
        for candidate in candidates:
            candidate_tokens = self._company_tokens(candidate.name)
            overlap = question_tokens & candidate_tokens
            if not overlap:
                continue
            score = (
                len(overlap) / max(len(candidate_tokens), 1),
                len(overlap),
                len(_normalize_name(candidate.name)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_name = candidate.name

        if best_name is not None and best_score is not None:
            coverage, overlap_count, _ = best_score
            if coverage >= 0.6 or overlap_count >= 2:
                return best_name
        return None

    def infer_request_identity(self, request: DiligenceRequest) -> tuple[str | None, int | None]:
        company = request.company or self._infer_company_from_question(request.question)
        fiscal_year = request.fiscal_year or self._infer_fiscal_year(request.question)
        return company, fiscal_year

    def _request_with_inferred_identity(self, request: DiligenceRequest) -> DiligenceRequest:
        company, fiscal_year = self.infer_request_identity(request)
        if company == request.company and fiscal_year == request.fiscal_year:
            return request
        return request.model_copy(update={"company": company, "fiscal_year": fiscal_year})

    def _build_runner(self):
        @traceable(name="run_diligence_request", run_type="chain")
        def _runner(payload: dict[str, Any]) -> dict[str, Any]:
            request = DiligenceRequest.model_validate(payload)
            response, trace_metadata = self._invoke_graph(request)
            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.add_tags(
                    [
                        "retrieval_mode:local_only",
                        "reasoning_mode:hybrid",
                        f"validation_outcome:{trace_metadata['validation_outcome']}",
                        f"coverage_complete:{trace_metadata['coverage_complete']}",
                        f"retrieval_iterations:{trace_metadata['retrieval_pass_count']}",
                        f"missing_operands:{trace_metadata['missing_operand_count']}",
                    ]
                )
                run_tree.add_metadata(trace_metadata)
            return {
                "response": response.model_dump(mode="json"),
                "telemetry": trace_metadata,
            }

        return _runner

    def run_request(self, request: DiligenceRequest) -> DiligenceResponse:
        payload = request.model_dump(mode="json")
        result = self._runner(payload)
        return DiligenceResponse.model_validate(result["response"])

    def run_request_with_trace(self, request: DiligenceRequest) -> tuple[DiligenceResponse, dict[str, Any]]:
        payload = request.model_dump(mode="json")
        result = self._runner(payload)
        return DiligenceResponse.model_validate(result["response"]), dict(result["telemetry"])

    def _invoke_graph(self, request: DiligenceRequest) -> tuple[DiligenceResponse, dict[str, Any]]:
        request = self._request_with_inferred_identity(request)
        initial_state: DiligenceState = {
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
        task_type = normalize_task_type(request.question)
        config = {
            "configurable": {"thread_id": request.request_id},
            "tags": [
                f"qid:{request.qid or 'adhoc'}",
                f"company:{(request.company or 'unknown').replace(' ', '_')}",
                f"task_type:{task_type}",
                "retrieval_mode:local_only",
                "reasoning_mode:hybrid",
                "workflow_version:local_only_v1",
            ],
            "metadata": {
                "qid": request.qid,
                "company": request.company,
                "task_type": task_type,
                "retrieval_mode": "local_only",
                "reasoning_mode": "hybrid",
                "validation_outcome": "pending",
                "coverage_complete": False,
                "missing_operand_count": 0,
                "retrieval_pass_count": 0,
                "empty_pass_count": 0,
                "retrieval_stop_reason": "not_started",
                "pages_scanned_count": 0,
                "pages_opened_count": 0,
                "exact_scan_match_count": 0,
                "vector_hits_count": 0,
                "vector_index_used": False,
                "vector_primary_hit_rank": None,
                "vector_retrieval_queries": [],
                "opened_files": [],
                "primary_candidate_file": None,
                "primary_file_used": False,
                "full_text_fallback_used": False,
                "workflow_version": "local_only_v1",
            },
            "recursion_limit": RECURSION_LIMIT,
        }

        try:
            result = self.graph.invoke(initial_state, config=config)
        except Exception as exc:
            response = DiligenceResponse(
                final_answer=None,
                answer_kind="unknown",
                explanation=(
                    "The workflow hit an unrecoverable error before it could verify the answer. "
                    "Escalating this request for human review is safer than guessing."
                ),
                citations=[],
                confidence=0.0,
                needs_human_review=True,
                errors=[str(exc)],
            )
            return response, {
                "qid": request.qid,
                "company": request.company,
                "task_type": task_type,
                "retrieval_mode": "local_only",
                "reasoning_mode": "hybrid",
                "validation_outcome": "human_review",
                "coverage_complete": False,
                "missing_operand_count": 0,
                "retrieval_pass_count": 0,
                "empty_pass_count": 0,
                "retrieval_stop_reason": "graph_error",
                "pages_scanned_count": 0,
                "pages_opened_count": 0,
                "exact_scan_match_count": 0,
                "vector_hits_count": 0,
                "vector_index_used": False,
                "vector_primary_hit_rank": None,
                "vector_retrieval_queries": [],
                "opened_files": [],
                "primary_candidate_file": None,
                "primary_file_used": False,
                "full_text_fallback_used": False,
                "workflow_version": "local_only_v1",
            }

        final_response = result.get("final_response")
        if isinstance(final_response, DiligenceResponse):
            response = final_response
        elif isinstance(final_response, dict):
            response = DiligenceResponse.model_validate(final_response)
        else:
            response = DiligenceResponse(
                final_answer=None,
                answer_kind="unknown",
                explanation="The workflow finished without producing a verified response.",
                citations=[],
                confidence=0.0,
                needs_human_review=True,
                errors=["Missing final response in graph state."],
            )

        assessment = result.get("coverage_assessment")
        trace_metadata = {
            "qid": request.qid,
            "company": request.company,
            "task_type": result.get("intake").task_type if result.get("intake") else task_type,
            "retrieval_mode": "local_only",
            "reasoning_mode": "hybrid",
            "validation_outcome": self._validation_outcome(result, response),
            "coverage_complete": bool(
                assessment.enough_evidence_to_answer
                if isinstance(assessment, CoverageAssessment)
                else (assessment or {}).get("enough_evidence_to_answer", False)
            ),
            "missing_operand_count": len(
                assessment.missing_operands
                if isinstance(assessment, CoverageAssessment)
                else (assessment or {}).get("missing_operands", [])
            ),
            "retrieval_pass_count": int(result.get("retrieval_iteration", 0)),
            "empty_pass_count": int(result.get("empty_retrieval_pass_count", 0)),
            "retrieval_stop_reason": result.get("retrieval_stop_reason") or "coverage_complete",
            "pages_scanned_count": int(result.get("pages_scanned_count", 0)),
            "pages_opened_count": int(result.get("pages_opened_count", 0)),
            "exact_scan_match_count": int(result.get("exact_scan_match_count", 0)),
            "vector_hits_count": int(result.get("vector_hits_count", 0)),
            "vector_index_used": bool(result.get("vector_index_used", False)),
            "vector_primary_hit_rank": result.get("vector_primary_hit_rank"),
            "vector_retrieval_queries": list(result.get("vector_retrieval_queries", [])),
            "opened_files": list(result.get("opened_files", [])),
            "primary_candidate_file": (
                result.get("retrieval_plan").primary_candidate_file
                if isinstance(result.get("retrieval_plan"), RetrievalPlan)
                else (result.get("retrieval_plan") or {}).get("primary_candidate_file")
            ),
            "primary_file_used": self._primary_file_used(result, response),
            "full_text_fallback_used": bool(result.get("full_text_fallback_used", False)),
            "workflow_version": "local_only_v1",
        }
        return response, trace_metadata

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(DiligenceState)
        graph.add_node("intake", self.intake)
        graph.add_node("plan_sources", self.plan_sources)
        graph.add_node("retrieve_local_evidence", self.retrieve_local_evidence)
        graph.add_node("assess_evidence_coverage", self.assess_evidence_coverage)
        graph.add_node("reason_with_evidence", self.reason_with_evidence)
        graph.add_node("validate_numeric_answer", self.validate_numeric_answer)
        graph.add_node("verify_answer", self.verify_answer)
        graph.add_node("respond", self.respond)

        graph.add_edge(START, "intake")
        graph.add_conditional_edges(
            "intake",
            self._route_after_intake,
            {"plan_sources": "plan_sources", "respond": "respond"},
        )
        graph.add_edge("plan_sources", "retrieve_local_evidence")
        graph.add_edge("retrieve_local_evidence", "assess_evidence_coverage")
        graph.add_conditional_edges(
            "assess_evidence_coverage",
            self._route_after_coverage_assessment,
            {"retrieve_local_evidence": "retrieve_local_evidence", "reason_with_evidence": "reason_with_evidence"},
        )
        graph.add_edge("reason_with_evidence", "validate_numeric_answer")
        graph.add_edge("validate_numeric_answer", "verify_answer")
        graph.add_conditional_edges(
            "verify_answer",
            self._route_after_verify,
            {"reason_with_evidence": "reason_with_evidence", "respond": "respond"},
        )
        graph.add_edge("respond", END)
        return graph

    def _route_after_intake(self, state: DiligenceState) -> str:
        return "respond" if state["status"] == "missing_fields" else "plan_sources"

    def _route_after_coverage_assessment(self, state: DiligenceState) -> str:
        return "retrieve_local_evidence" if state["status"] == "needs_more_evidence" else "reason_with_evidence"

    def _route_after_verify(self, state: DiligenceState) -> str:
        return "reason_with_evidence" if state["status"] == "retry_reasoning" else "respond"

    def _validation_outcome(self, state: dict[str, Any], response: DiligenceResponse) -> str:
        validation = state.get("validation_result")
        if response.needs_human_review:
            return "human_review"
        if state.get("attempt_count", 0) > 0:
            return "retry"
        if isinstance(validation, ValidationResult) and not validation.issues:
            return "pass"
        if isinstance(validation, dict) and not validation.get("issues"):
            return "pass"
        return "pass"

    def _primary_file_used(self, state: dict[str, Any], response: DiligenceResponse) -> bool:
        plan = state.get("retrieval_plan")
        if isinstance(plan, RetrievalPlan):
            primary_file = plan.primary_candidate_file
        else:
            primary_file = (plan or {}).get("primary_candidate_file")
        if not primary_file:
            return False
        return any(citation.source_path == primary_file for citation in response.citations)

    def _query_fingerprint(self, query: str) -> str:
        return " ".join(query.lower().split())

    def _file_query_pair_key(self, file_path: str, query: str) -> str:
        return f"{file_path}::{self._query_fingerprint(query)}"

    def _dedupe_active_queries(self, queries: list[str], seen_fingerprints: list[str] | None = None) -> list[str]:
        seen = set(seen_fingerprints or [])
        deduped: list[str] = []
        for query in queries:
            cleaned = " ".join(str(query).split()).strip()
            if not cleaned:
                continue
            fingerprint = self._query_fingerprint(cleaned)
            if fingerprint in seen or cleaned in deduped:
                continue
            deduped.append(cleaned)
        return deduped

    def _operand_terms(self, operand_name: str) -> list[str]:
        parts = [part for part in re.split(r"[^a-z0-9]+", operand_name.lower()) if part]
        terms = list(parts)
        joined = " ".join(parts)
        if joined and joined not in terms:
            terms.append(joined)
        return terms

    def _formula_terms(self, formula: str) -> list[str]:
        return [token for token in re.findall(r"[a-z][a-z0-9_]{1,}", formula.lower()) if token not in {"and", "or"}]

    def _phrase_to_pattern(self, phrase: str) -> str:
        tokens = [re.escape(token) for token in re.findall(r"[a-z0-9]+", phrase.lower())]
        if not tokens:
            return ""
        return r"\b" + r"[-\s]+".join(tokens) + r"\b"

    def _build_scan_targets(
        self,
        request: DiligenceRequest,
        intake: IntakeAnalysis | None,
        coverage: CoverageAssessment | None,
        active_queries: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        base_terms: list[str] = []
        if coverage is not None and coverage.missing_operands:
            base_terms.extend(coverage.missing_operands)
        elif coverage is not None and coverage.required_operands:
            base_terms.extend(coverage.required_operands)
        elif intake is not None:
            base_terms.extend(intake.required_metrics)

        scan_terms: list[str] = []
        for term in base_terms:
            cleaned = str(term).replace("_", " ").strip()
            if not cleaned:
                continue
            scan_terms.extend(expand_search_terms(cleaned))
            scan_terms.extend(expand_missing_operand_queries([cleaned]))

        scan_terms.extend(expand_search_terms(request.question))
        scan_terms = self._dedupe_active_queries(scan_terms)

        regex_patterns: list[str] = []
        for term in scan_terms:
            pattern = self._phrase_to_pattern(term)
            if pattern:
                regex_patterns.append(pattern)
        if any("debt" in term or "borrow" in term for term in scan_terms):
            regex_patterns.extend(
                [
                    r"\b(total[-\s]+debt(?:\s+outstanding)?|debt\s+outstanding)\b",
                    r"\b(borrowings?|senior\s+notes?|notes\s+due)\b",
                ]
            )
        if any("lease" in term for term in scan_terms):
            regex_patterns.extend(
                [
                    r"\b(operating[-\s]+lease\s+liabilities|lease\s+obligations|lease\s+liabilities)\b",
                    r"\b(right[-\s]+of[-\s]+use\s+assets?)\b",
                ]
            )
        if any("cash" in term for term in scan_terms):
            regex_patterns.extend(
                [
                    r"\b(cash\s+and\s+cash\s+equivalents?)\b",
                    r"\b(short[-\s]+term\s+investments|marketable\s+securities)\b",
                ]
            )
        if any("equity" in term for term in scan_terms):
            regex_patterns.append(r"\b(stockholders'?\s+equity|shareholders'?\s+equity|total\s+equity)\b")
        regex_patterns = self._dedupe_active_queries(regex_patterns)

        token_queries = self._dedupe_active_queries(active_queries)
        return scan_terms, regex_patterns, token_queries

    def _prioritize_scan_hits(self, hits: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
        prioritized = sorted(
            hits,
            key=lambda item: (
                0
                if item.get("match_mode") == "phrase"
                else 1
                if item.get("match_mode") == "regex"
                else 2,
                -float(item.get("score", 0.0)),
                int(item.get("page_number", 0)),
            ),
        )
        unique: list[dict[str, Any]] = []
        seen_pages: set[int] = set()
        for hit in prioritized:
            page_number = int(hit.get("page_number", 0))
            if page_number <= 0 or page_number in seen_pages:
                continue
            seen_pages.add(page_number)
            unique.append(hit)
            if len(unique) >= limit:
                break
        return unique

    def _sort_chunks_with_primary_bias(self, chunks: list[EvidenceChunk], primary_file: str | None) -> list[EvidenceChunk]:
        return sorted(
            chunks,
            key=lambda item: (
                0 if primary_file and item.source_path == primary_file else 1,
                -item.score,
                item.source_label,
                item.page_number,
            ),
        )

    def _append_unique_paths(self, paths: list[str], new_path: str) -> list[str]:
        if new_path and new_path not in paths:
            paths.append(new_path)
        return paths

    def _pages_with_window(self, page_hits: list[dict[str, Any]], page_count: int, before: int, after: int) -> list[int]:
        pages: set[int] = set()
        for hit in page_hits:
            hit_page = int(hit["page_number"])
            for page_number in range(max(1, hit_page - before), min(page_count, hit_page + after) + 1):
                pages.add(page_number)
        return sorted(pages)

    def _carryforward_queries(
        self,
        candidate_files: list[str],
        queries: list[str],
        searched_pairs: list[str],
    ) -> list[str]:
        carryforward: list[str] = []
        searched = set(searched_pairs)
        for query in queries:
            if any(self._file_query_pair_key(file_path, query) not in searched for file_path in candidate_files):
                carryforward.append(query)
        return self._dedupe_active_queries(carryforward)

    def _resolve_fy_dir_path(self, plan: RetrievalPlan | None) -> Path | None:
        if plan is None:
            return None
        if plan.fiscal_year_dir:
            return Path(plan.fiscal_year_dir)
        if plan.primary_candidate_file:
            return Path(plan.primary_candidate_file).parent
        if plan.candidate_files:
            return Path(plan.candidate_files[0]).parent
        return None

    def _build_vector_queries(
        self,
        request: DiligenceRequest,
        active_queries: list[str],
        scan_terms: list[str],
    ) -> list[str]:
        queries = self._dedupe_active_queries(
            [request.question] + list(active_queries) + list(scan_terms[:6])
        )
        return queries[:VECTOR_QUERY_LIMIT]

    def _prioritize_vector_hits(
        self,
        hits: list[dict[str, Any]],
        primary_file: str | None,
        limit: int = VECTOR_HIT_LIMIT,
    ) -> list[dict[str, Any]]:
        prioritized = sorted(
            hits,
            key=lambda item: (
                0 if primary_file and item.get("file_path") == primary_file else 1,
                int(item.get("best_rank", 10_000)),
                -float(item.get("score", 0.0)),
                str(item.get("file_name", "")),
                int(item.get("center_page", 0)),
            ),
        )
        unique: list[dict[str, Any]] = []
        seen_pages: set[tuple[str, int]] = set()
        for hit in prioritized:
            key = (str(hit.get("file_path", "")), int(hit.get("center_page", 0)))
            if not key[0] or key[1] <= 0 or key in seen_pages:
                continue
            seen_pages.add(key)
            unique.append(hit)
            if len(unique) >= limit:
                break
        return unique

    def _search_vector_index(
        self,
        fy_dir: Path | None,
        vector_queries: list[str],
        primary_file: str | None,
    ) -> tuple[list[dict[str, Any]], bool, int | None]:
        if fy_dir is None or self.index_manager is None or not vector_queries:
            return [], False, None

        loaded_index = self.index_manager.load_fy_index(fy_dir)
        if loaded_index is None:
            return [], False, None

        aggregated_hits: dict[tuple[str, int], dict[str, Any]] = {}
        for query in vector_queries:
            query_hits = loaded_index.vector_store.similarity_search_with_score(query, k=VECTOR_SEARCH_TOP_K)
            for rank, (document, score) in enumerate(query_hits, start=1):
                if not isinstance(document, Document):
                    continue
                file_path = str(document.metadata.get("file_path", ""))
                center_page = int(document.metadata.get("center_page", 0))
                if not file_path or center_page <= 0:
                    continue
                key = (file_path, center_page)
                existing = aggregated_hits.get(key)
                match_reasons = [f"vector_query:{query}"]
                if existing is None:
                    aggregated_hits[key] = {
                        "file_path": file_path,
                        "file_name": str(document.metadata.get("file_name", Path(file_path).name)),
                        "center_page": center_page,
                        "window_start": int(document.metadata.get("window_start", center_page)),
                        "window_end": int(document.metadata.get("window_end", center_page)),
                        "score": float(score),
                        "best_rank": rank,
                        "match_reasons": match_reasons,
                    }
                    continue

                if rank < int(existing["best_rank"]):
                    existing["best_rank"] = rank
                if float(score) > float(existing["score"]):
                    existing["score"] = float(score)
                existing_reasons = list(existing.get("match_reasons", []))
                for reason in match_reasons:
                    if reason not in existing_reasons:
                        existing_reasons.append(reason)
                existing["match_reasons"] = existing_reasons

        prioritized_hits = self._prioritize_vector_hits(list(aggregated_hits.values()), primary_file)
        primary_rank = None
        if primary_file:
            for rank, hit in enumerate(prioritized_hits, start=1):
                if hit["file_path"] == primary_file:
                    primary_rank = rank
                    break
        return prioritized_hits, True, primary_rank

    def _hydrate_vector_hits(
        self,
        hits: list[dict[str, Any]],
        opened_files: list[str],
        deep_read_files: list[str],
        pages_opened_count: int,
    ) -> tuple[list[EvidenceChunk], list[str], list[str], int]:
        evidence: list[EvidenceChunk] = []
        seen_pages: set[tuple[str, int]] = set()
        for hit in hits:
            source_path = str(hit["file_path"])
            source_label = str(hit.get("file_name", Path(source_path).name))
            window_start = int(hit.get("window_start", hit["center_page"]))
            window_end = int(hit.get("window_end", hit["center_page"]))
            page_numbers = list(range(window_start, window_end + 1))

            self._append_unique_paths(opened_files, source_path)
            self._append_unique_paths(deep_read_files, source_path)

            exact_pages = read_pdf_pages.invoke(
                {
                    "file_path": source_path,
                    "pages": page_numbers,
                    "max_chars": None,
                }
            )
            for page in exact_pages:
                page_number = int(page["page_number"])
                if (source_path, page_number) not in seen_pages:
                    seen_pages.add((source_path, page_number))
                    pages_opened_count += 1
                page_score = max(1.0, 50.0 - float(hit.get("best_rank", 1)))
                if page_number != int(hit["center_page"]):
                    page_score *= 0.85
                evidence.append(
                    EvidenceChunk(
                        citation_id=f"local_pdf:{Path(source_label).stem}:{page_number}",
                        source_type="local_pdf",
                        source_label=source_label,
                        source_path=source_path,
                        page_number=page_number,
                        snippet=str(page["text"]),
                        score=round(page_score, 4),
                        match_reasons=list(hit.get("match_reasons", []))
                        + [f"vector_window:{window_start}-{window_end}"],
                    )
                )
        return evidence, opened_files, deep_read_files, pages_opened_count

    def _select_prompt_evidence(
        self,
        evidence_pool: list[EvidenceChunk],
        assessment: CoverageAssessment | None,
        plan: RetrievalPlan | None = None,
        limit: int = PROMPT_EVIDENCE_LIMIT,
    ) -> list[EvidenceChunk]:
        if not evidence_pool:
            return []

        operands: list[str] = []
        formula_terms: list[str] = []
        if assessment is not None:
            operands.extend(assessment.found_operands)
            operands.extend(assessment.missing_operands)
            operands.extend(assessment.required_operands)
            formula_terms.extend(self._formula_terms(assessment.candidate_formula))

        primary_file = plan.primary_candidate_file if plan is not None else None
        ranked_pool = self._sort_chunks_with_primary_bias(evidence_pool, primary_file)

        lower_snippets = {
            chunk.citation_id: " ".join(chunk.snippet.lower().split())
            for chunk in ranked_pool
        }
        selected: list[EvidenceChunk] = []
        seen_ids: set[str] = set()
        selected_by_source_page: set[tuple[str, int]] = set()

        def _add_chunk(chunk: EvidenceChunk) -> bool:
            key = (chunk.source_path, chunk.page_number)
            if chunk.citation_id in seen_ids or key in selected_by_source_page:
                return False
            selected.append(chunk)
            seen_ids.add(chunk.citation_id)
            selected_by_source_page.add(key)
            return len(selected) >= limit

        for operand in operands:
            terms = self._operand_terms(operand)
            matches = [
                chunk
                for chunk in ranked_pool
                if chunk.citation_id not in seen_ids
                and any(term in lower_snippets[chunk.citation_id] for term in terms)
            ]
            if not matches:
                continue
            matches = self._sort_chunks_with_primary_bias(matches, primary_file)
            if _add_chunk(matches[0]):
                return selected

            contextual_matches = [
                chunk
                for chunk in ranked_pool
                if chunk.citation_id not in seen_ids
                and chunk.source_path == matches[0].source_path
                and abs(chunk.page_number - matches[0].page_number) <= 1
            ]
            if contextual_matches:
                contextual_matches = self._sort_chunks_with_primary_bias(contextual_matches, primary_file)
                if _add_chunk(contextual_matches[0]):
                    return selected

        if formula_terms:
            formula_matches = [
                chunk
                for chunk in ranked_pool
                if chunk.citation_id not in seen_ids
                and any(term.replace("_", " ") in lower_snippets[chunk.citation_id] for term in formula_terms)
            ]
            for chunk in self._sort_chunks_with_primary_bias(formula_matches, primary_file):
                if _add_chunk(chunk):
                    return selected

        for chunk in ranked_pool:
            if _add_chunk(chunk):
                break
        return selected

    def _remaining_file_query_pairs_exist(self, candidate_files: list[str], queries: list[str], searched_pairs: list[str]) -> bool:
        searched = set(searched_pairs)
        for query in queries:
            for file_path in candidate_files:
                if self._file_query_pair_key(file_path, query) not in searched:
                    return True
        return False

    def _fallback_metrics(self, question: str) -> list[str]:
        return merged_required_metrics(normalize_task_type(question), [])

    def _infer_fiscal_year(self, question: str) -> int | None:
        match = re.search(r"FY\s*(20\d{2})", question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(20\d{2})\b", question)
        return int(match.group(1)) if match else None

    def _build_missing_fields_response(self, missing_fields: list[str], errors: list[str]) -> DiligenceResponse:
        missing_text = ", ".join(missing_fields)
        return DiligenceResponse(
            final_answer=None,
            answer_kind="unknown",
            explanation=(
                f"I cannot safely search the dataroom yet because these required fields are missing: {missing_text}. "
                "A human or upstream form should provide them before the workflow continues."
            ),
            citations=[],
            confidence=0.0,
            needs_human_review=True,
            errors=_unique_errors(errors + [f"Missing required fields: {missing_text}"]),
        )

    def intake(self, state: DiligenceState) -> dict[str, Any]:
        request = self._request_with_inferred_identity(state["request"])
        errors = list(state["errors"])
        fallback_year = request.fiscal_year or self._infer_fiscal_year(request.question)

        prompt = (
            "Normalize this diligence request for a LangGraph workflow.\n"
            "Return the resolved company, fiscal year, task type, required metrics, and missing fields.\n"
            "Only use missing_fields for missing request inputs such as question, company, or fiscal year.\n"
            "Do not mark finance facts like revenue, debt, lease liabilities, or cash as missing.\n"
            "If the company or fiscal year is already supplied, keep it unless the question clearly contradicts it.\n\n"
            f"Question: {request.question}\n"
            f"Provided company: {request.company or 'None'}\n"
            f"Provided fiscal year: {request.fiscal_year or 'None'}\n"
        )

        try:
            analysis = self.intake_llm.invoke(prompt)
        except Exception as exc:
            errors.append(f"Intake model fallback used: {exc}")
            analysis = IntakeAnalysis(
                company=request.company,
                fiscal_year=fallback_year,
                task_type="manual_review",
                required_metrics=self._fallback_metrics(request.question),
                missing_fields=[],
                notes=["Heuristic intake fallback was used because structured parsing failed."],
            )

        resolved_company = analysis.company or request.company
        resolved_year = analysis.fiscal_year or fallback_year
        task_type = normalize_task_type(request.question, analysis.task_type)
        request_missing_fields, retrievable_fields = request_identity_missing_fields(analysis.missing_fields)
        missing_fields = list(request_missing_fields)
        if not request.question.strip():
            missing_fields.append("question")
        if not resolved_company:
            missing_fields.append("company")
        if not resolved_year:
            missing_fields.append("fiscal_year")

        notes = list(analysis.notes)
        if retrievable_fields:
            notes.append(
                "Treated retrievable finance values as retrieval targets rather than blocking inputs: "
                + ", ".join(retrievable_fields)
            )

        analysis = analysis.model_copy(
            update={
                "company": resolved_company,
                "fiscal_year": resolved_year,
                "task_type": task_type,
                "required_metrics": merged_required_metrics(task_type, analysis.required_metrics + retrievable_fields),
                "missing_fields": sorted(set(missing_fields)),
                "notes": notes,
            }
        )

        if analysis.missing_fields:
            response = self._build_missing_fields_response(analysis.missing_fields, errors)
            return {
                "request": request,
                "intake": analysis,
                "final_response": response,
                "status": "missing_fields",
                "errors": response.errors,
            }

        return {
            "request": request,
            "intake": analysis,
            "status": "intake_complete",
            "errors": _unique_errors(errors),
        }

    def _resolve_company_dir(self, company_name: str | None) -> Path | None:
        if not company_name or not self.dataroom_root.exists():
            return None

        normalized_company = _normalize_name(company_name)
        candidates = [path for path in self.dataroom_root.iterdir() if path.is_dir()]

        for candidate in candidates:
            normalized_candidate = _normalize_name(candidate.name)
            if normalized_candidate == normalized_company:
                return candidate

        for candidate in candidates:
            normalized_candidate = _normalize_name(candidate.name)
            if normalized_candidate in normalized_company or normalized_company in normalized_candidate:
                return candidate

        requested_tokens = set(re.findall(r"[a-z0-9]+", company_name.lower()))
        best_match: Path | None = None
        best_overlap = 0
        for candidate in candidates:
            candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.name.lower()))
            overlap = len(requested_tokens & candidate_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = candidate
        return best_match

    def _resolve_fiscal_year_dir(self, company_dir: Path | None, fiscal_year: int | None) -> Path | None:
        if not company_dir or fiscal_year is None or not company_dir.exists():
            return None
        exact = company_dir / f"FY {fiscal_year}"
        if exact.exists():
            return exact
        for child in company_dir.iterdir():
            if child.is_dir() and str(fiscal_year) in child.name:
                return child
        return None

    def _rank_candidate_files(self, company_dir: Path | None, fy_dir: Path | None, fiscal_year: int | None) -> list[str]:
        if not fy_dir or not fy_dir.exists():
            return []
        company_tokens = set(re.findall(r"[a-z0-9]+", company_dir.name.lower())) if company_dir else set()
        ranked: list[tuple[float, str]] = []

        for pdf_path in fy_dir.iterdir():
            if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
                continue
            lower_name = pdf_path.name.lower()
            score = 0.0
            if "10-k" in lower_name or "10k" in lower_name:
                score += 5.0
            if "annual" in lower_name:
                score += 4.0
            if "report" in lower_name:
                score += 1.5
            if fiscal_year and str(fiscal_year) in lower_name:
                score += 1.0
            score += 0.25 * sum(1 for token in company_tokens if token in lower_name)
            ranked.append((score, str(pdf_path)))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in ranked]

    def plan_sources(self, state: DiligenceState) -> dict[str, Any]:
        intake = state.get("intake")
        request = state["request"]
        errors = list(state["errors"])
        task_type = intake.task_type if intake else normalize_task_type(request.question)

        company_dir = self._resolve_company_dir(intake.company if intake else request.company)
        fy_dir = self._resolve_fiscal_year_dir(company_dir, intake.fiscal_year if intake else request.fiscal_year)
        local_queries = build_local_queries(
            task_type=task_type,
            question=request.question,
            metrics=intake.required_metrics if intake else [],
        )
        candidate_files = self._rank_candidate_files(
            company_dir,
            fy_dir,
            intake.fiscal_year if intake else request.fiscal_year,
        )

        if company_dir is None:
            errors.append(f"Could not resolve company folder for '{(intake.company if intake else request.company) or 'unknown'}'.")
        if fy_dir is None:
            errors.append(f"Could not resolve fiscal year folder for FY {(intake.fiscal_year if intake else request.fiscal_year) or 'unknown'}.")
        if not candidate_files:
            errors.append("No likely annual filing candidates were found for this request.")

        retrieval_plan = RetrievalPlan(
            company_dir=str(company_dir) if company_dir else None,
            fiscal_year_dir=str(fy_dir) if fy_dir else None,
            candidate_files=candidate_files,
            primary_candidate_file=candidate_files[0] if candidate_files else None,
            secondary_candidate_files=candidate_files[1:] if len(candidate_files) > 1 else [],
            search_queries=local_queries,
            active_local_queries=local_queries,
        )

        return {
            "retrieval_plan": retrieval_plan,
            "status": "planning_complete",
            "errors": _unique_errors(errors),
        }

    def _hydrate_evidence(
        self,
        source_path: str,
        source_label: str,
        page_hits: list[dict[str, Any]],
        before: int = PRIMARY_PAGE_WINDOW[0],
        after: int = PRIMARY_PAGE_WINDOW[1],
    ) -> list[EvidenceChunk]:
        with pymupdf.open(source_path) as pdf_document:
            page_numbers = self._pages_with_window(page_hits, pdf_document.page_count, before, after)

        exact_pages = read_pdf_pages.invoke(
            {
                "file_path": source_path,
                "pages": page_numbers,
                "max_chars": None,
            }
        )
        score_by_page = {int(hit["page_number"]): float(hit["score"]) for hit in page_hits}
        reasons_by_page = {
            int(hit["page_number"]): list(hit.get("match_reasons", []))
            for hit in page_hits
        }
        evidence: list[EvidenceChunk] = []
        for page in exact_pages:
            page_number = int(page["page_number"])
            score = score_by_page.get(page_number)
            if score is None:
                nearby_scores = [
                    hit_score
                    for hit_page, hit_score in score_by_page.items()
                    if abs(hit_page - page_number) <= max(before, after)
                ]
                score = max(nearby_scores, default=0.05) * 0.8
            evidence.append(
                EvidenceChunk(
                    citation_id=f"local_pdf:{Path(source_label).stem}:{page_number}",
                    source_type="local_pdf",
                    source_label=source_label,
                    source_path=source_path,
                    page_number=page_number,
                    snippet=str(page["text"]),
                    score=round(score, 4),
                    match_reasons=reasons_by_page.get(page_number, []),
                )
            )
        return evidence

    def _hydrate_full_text_evidence(self, source_path: str, source_label: str) -> list[EvidenceChunk]:
        pages = get_full_pdf_text.invoke(
            {
                "file_path": source_path,
                "max_chars_per_page": None,
            }
        )
        evidence: list[EvidenceChunk] = []
        for page in pages:
            page_number = int(page["page_number"])
            evidence.append(
                EvidenceChunk(
                    citation_id=f"local_pdf:{Path(source_label).stem}:{page_number}",
                    source_type="local_pdf",
                    source_label=source_label,
                    source_path=source_path,
                    page_number=page_number,
                    snippet=str(page["text"]),
                    score=0.02,
                    match_reasons=["full_text_fallback"],
                )
            )
        return evidence

    def _merge_evidence(self, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        merged: dict[str, EvidenceChunk] = {}
        for chunk in chunks:
            existing = merged.get(chunk.citation_id)
            if existing is None or (chunk.score, len(chunk.snippet)) > (existing.score, len(existing.snippet)):
                merged[chunk.citation_id] = chunk
        return sorted(merged.values(), key=lambda item: (-item.score, item.source_label, item.page_number))

    def _scan_files_for_targets(
        self,
        files: list[str],
        scan_terms: list[str],
        regex_patterns: list[str],
        token_queries: list[str],
        evidence_pool: list[EvidenceChunk],
        searched_pairs: list[str],
        opened_files: list[str],
        deep_read_files: list[str],
        pages_scanned_count: int,
        pages_opened_count: int,
        exact_scan_match_count: int,
    ) -> tuple[list[EvidenceChunk], list[str], list[str], list[str], int, int, int]:
        for candidate_file in files:
            self._append_unique_paths(opened_files, candidate_file)
            with pymupdf.open(candidate_file) as document:
                file_page_count = document.page_count
                pages_scanned_count += file_page_count
            for search_query in token_queries:
                pair_key = self._file_query_pair_key(candidate_file, search_query)
                if pair_key in searched_pairs:
                    continue
                searched_pairs.append(pair_key)
                page_hits = scan_pdf_pages.invoke(
                    {
                        "file_path": candidate_file,
                        "search_terms": scan_terms,
                        "regex_patterns": regex_patterns,
                        "token_bundle_queries": [search_query],
                        "snippet_chars": 420,
                    }
                )
                if not page_hits:
                    heuristic_hits = search_document_pages.invoke(
                        {
                            "file_path": candidate_file,
                            "query": search_query,
                            "top_k": SECONDARY_HEURISTIC_TOP_K,
                            "snippet_chars": 420,
                        }
                    )
                    page_hits = [
                        {
                            "page_number": int(hit["page_number"]),
                            "score": float(hit["score"]),
                            "match_mode": "token_bundle",
                            "match_reasons": [f"heuristic:{search_query}"],
                            "snippet": hit["snippet"],
                        }
                        for hit in heuristic_hits
                    ]
                if not page_hits:
                    continue
                prioritized_hits = self._prioritize_scan_hits(page_hits)
                self._append_unique_paths(deep_read_files, candidate_file)
                exact_scan_match_count += len(prioritized_hits)
                pages_opened_count += len(
                    self._pages_with_window(prioritized_hits, file_page_count, PRIMARY_PAGE_WINDOW[0], PRIMARY_PAGE_WINDOW[1])
                )
                evidence_pool.extend(
                    self._hydrate_evidence(
                        source_path=candidate_file,
                        source_label=Path(candidate_file).name,
                        page_hits=prioritized_hits,
                    )
                )
                evidence_pool = self._merge_evidence(evidence_pool)
                break
        return (
            evidence_pool,
            searched_pairs,
            opened_files,
            deep_read_files,
            pages_scanned_count,
            pages_opened_count,
            exact_scan_match_count,
        )

    def _run_full_text_fallback(
        self,
        files: list[str],
        evidence_pool: list[EvidenceChunk],
        opened_files: list[str],
        deep_read_files: list[str],
    ) -> tuple[list[EvidenceChunk], list[str], list[str], bool]:
        used_fallback = False
        for candidate_file in files[:FULL_TEXT_FALLBACK_FILES]:
            self._append_unique_paths(opened_files, candidate_file)
            self._append_unique_paths(deep_read_files, candidate_file)
            evidence_pool.extend(
                self._hydrate_full_text_evidence(
                    source_path=candidate_file,
                    source_label=Path(candidate_file).name,
                )
            )
            evidence_pool = self._merge_evidence(evidence_pool)
            used_fallback = True
        return evidence_pool, opened_files, deep_read_files, used_fallback

    def retrieve_local_evidence(self, state: DiligenceState) -> dict[str, Any]:
        plan = state.get("retrieval_plan")
        intake = state.get("intake")
        coverage = state.get("coverage_assessment")
        request = state["request"]
        errors = list(state["errors"])
        evidence_pool = list(state.get("evidence_pool", []))
        seen_query_fingerprints = list(state.get("seen_query_fingerprints", []))
        searched_pairs = list(state.get("searched_file_query_pairs", []))
        opened_files = list(state.get("opened_files", []))
        deep_read_files = list(state.get("deep_read_files", []))
        full_text_fallback_used = bool(state.get("full_text_fallback_used", False))
        pages_scanned_count = int(state.get("pages_scanned_count", 0))
        pages_opened_count = int(state.get("pages_opened_count", 0))
        exact_scan_match_count = int(state.get("exact_scan_match_count", 0))
        vector_hits_count = int(state.get("vector_hits_count", 0))
        vector_index_used = bool(state.get("vector_index_used", False))
        vector_primary_hit_rank = state.get("vector_primary_hit_rank")
        vector_retrieval_queries = list(state.get("vector_retrieval_queries", []))

        if not plan or not plan.candidate_files:
            errors.append("Local evidence retrieval skipped because no candidate filings were available.")
            return {
                "evidence": state.get("evidence", []),
                "evidence_pool": evidence_pool,
                "citations": [chunk.to_citation() for chunk in evidence_pool],
                "last_retrieval_added_count": 0,
                "retrieval_stop_reason": "no_candidate_files",
                "pages_scanned_count": pages_scanned_count,
                "pages_opened_count": pages_opened_count,
                "exact_scan_match_count": exact_scan_match_count,
                "vector_hits_count": vector_hits_count,
                "vector_index_used": vector_index_used,
                "vector_primary_hit_rank": vector_primary_hit_rank,
                "vector_retrieval_queries": vector_retrieval_queries,
                "status": "local_evidence_missing",
                "errors": _unique_errors(errors),
            }

        query_candidates = plan.active_local_queries or plan.search_queries
        active_queries = self._dedupe_active_queries(query_candidates)
        scan_terms, regex_patterns, token_queries = self._build_scan_targets(
            request=request,
            intake=intake,
            coverage=coverage,
            active_queries=active_queries,
        )
        baseline_count = len(evidence_pool)
        primary_file = plan.primary_candidate_file
        iteration = int(state.get("retrieval_iteration", 0))
        fy_dir = self._resolve_fy_dir_path(plan)
        vector_queries = self._build_vector_queries(request, active_queries, scan_terms)
        vector_retrieval_queries = self._dedupe_active_queries(vector_retrieval_queries + vector_queries)

        for search_query in token_queries:
            fingerprint = self._query_fingerprint(search_query)
            if fingerprint not in seen_query_fingerprints:
                seen_query_fingerprints.append(fingerprint)
        for vector_query in vector_queries:
            fingerprint = self._query_fingerprint(vector_query)
            if fingerprint not in seen_query_fingerprints:
                seen_query_fingerprints.append(fingerprint)

        semantic_hits: list[dict[str, Any]] = []
        if fy_dir is not None and vector_queries:
            semantic_hits, vector_index_used_now, primary_hit_rank = self._search_vector_index(
                fy_dir=fy_dir,
                vector_queries=vector_queries,
                primary_file=primary_file,
            )
            vector_index_used = vector_index_used or vector_index_used_now
            if semantic_hits:
                (
                    vector_evidence,
                    opened_files,
                    deep_read_files,
                    pages_opened_count,
                ) = self._hydrate_vector_hits(
                    semantic_hits,
                    opened_files,
                    deep_read_files,
                    pages_opened_count,
                )
                evidence_pool.extend(vector_evidence)
                evidence_pool = self._merge_evidence(evidence_pool)
                vector_hits_count += len(semantic_hits)
                vector_primary_hit_rank = primary_hit_rank

        vector_added_count = len(evidence_pool) - baseline_count
        should_run_exact_scan = bool(primary_file) and (
            self.index_manager is None
            or fy_dir is None
            or vector_added_count == 0
            or (
                coverage is not None
                and not coverage.enough_evidence_to_answer
                and iteration > 0
            )
        )

        if should_run_exact_scan and primary_file:
            (
                evidence_pool,
                searched_pairs,
                opened_files,
                deep_read_files,
                pages_scanned_count,
                pages_opened_count,
                exact_scan_match_count,
            ) = self._scan_files_for_targets(
                [primary_file],
                scan_terms,
                regex_patterns,
                token_queries or active_queries,
                evidence_pool,
                searched_pairs,
                opened_files,
                deep_read_files,
                pages_scanned_count,
                pages_opened_count,
                exact_scan_match_count,
            )
            if not semantic_hits and self.index_manager is not None and fy_dir is not None:
                errors.append(
                    f"Vector index was unavailable or returned no hits for FY folder {fy_dir}. Exact page scanning was used instead."
                )

        primary_added_count = len(evidence_pool) - baseline_count
        should_deep_read_primary = bool(primary_file) and (
            primary_added_count == 0
            or (
                coverage is not None
                and not coverage.enough_evidence_to_answer
                and primary_file not in deep_read_files
            )
        )
        if should_deep_read_primary:
            evidence_pool, opened_files, deep_read_files, used_fallback = self._run_full_text_fallback(
                [primary_file],
                evidence_pool,
                opened_files,
                deep_read_files,
            )
            full_text_fallback_used = full_text_fallback_used or used_fallback
            if used_fallback:
                with pymupdf.open(primary_file) as document:
                    pages_scanned_count += document.page_count
                    pages_opened_count += document.page_count

        if iteration > 0 and plan.secondary_candidate_files:
            (
                evidence_pool,
                searched_pairs,
                opened_files,
                deep_read_files,
                pages_scanned_count,
                pages_opened_count,
                exact_scan_match_count,
            ) = self._scan_files_for_targets(
                plan.secondary_candidate_files,
                scan_terms,
                regex_patterns,
                token_queries or active_queries,
                evidence_pool,
                searched_pairs,
                opened_files,
                deep_read_files,
                pages_scanned_count,
                pages_opened_count,
                exact_scan_match_count,
            )

        if not evidence_pool:
            errors.append("No relevant local evidence pages were found in the candidate filing set.")
        if not token_queries and not scan_terms:
            errors.append("No new operand-oriented retrieval targets were available for this pass.")

        added_count = len(evidence_pool) - baseline_count
        empty_pass_count = int(state.get("empty_retrieval_pass_count", 0)) + (1 if added_count == 0 else 0)
        if added_count > 0:
            empty_pass_count = 0
        return {
            "evidence": state.get("evidence", []),
            "evidence_pool": evidence_pool,
            "citations": [chunk.to_citation() for chunk in evidence_pool],
            "seen_query_fingerprints": seen_query_fingerprints,
            "searched_file_query_pairs": searched_pairs,
            "opened_files": opened_files,
            "deep_read_files": deep_read_files,
            "full_text_fallback_used": full_text_fallback_used,
            "last_retrieval_added_count": added_count,
            "empty_retrieval_pass_count": empty_pass_count,
            "pages_scanned_count": pages_scanned_count,
            "pages_opened_count": pages_opened_count,
            "exact_scan_match_count": exact_scan_match_count,
            "vector_hits_count": vector_hits_count,
            "vector_index_used": vector_index_used,
            "vector_primary_hit_rank": vector_primary_hit_rank,
            "vector_retrieval_queries": vector_retrieval_queries,
            "status": "local_evidence_loaded" if evidence_pool else "local_evidence_missing",
            "errors": _unique_errors(errors),
        }

    def _fallback_coverage_assessment(self, state: DiligenceState) -> CoverageAssessment:
        intake = state.get("intake")
        metrics = list(intake.required_metrics if intake else [])
        evidence_pool = list(state.get("evidence_pool", []))
        combined_text = " ".join(chunk.snippet.lower() for chunk in evidence_pool)
        found: list[str] = []
        missing: list[str] = []
        for metric in metrics:
            terms = self._operand_terms(metric)
            if any(term in combined_text for term in terms):
                found.append(metric)
            else:
                missing.append(metric)
        return CoverageAssessment(
            candidate_formula="",
            required_operands=metrics,
            found_operands=found,
            missing_operands=missing,
            follow_up_local_queries=missing[:4],
            enough_evidence_to_answer=bool(evidence_pool) and not missing,
            reasoning_notes=["Heuristic coverage fallback was used because structured coverage assessment failed."],
        )

    def assess_evidence_coverage(self, state: DiligenceState) -> dict[str, Any]:
        plan = state.get("retrieval_plan")
        errors = list(state["errors"])
        evidence_pool = list(state.get("evidence_pool", []))
        coverage_notes = list(state.get("coverage_notes", []))
        next_iteration = int(state.get("retrieval_iteration", 0)) + 1
        assessment_input = self._select_prompt_evidence(
            evidence_pool,
            state.get("coverage_assessment"),
            plan=plan,
            limit=ASSESSMENT_EVIDENCE_LIMIT,
        )

        prompt = (
            "You are assessing whether the workflow has enough grounded local evidence to answer a diligence question.\n"
            "Use only the evidence below.\n"
            "Return a formula-agnostic coverage assessment.\n"
            "Requirements:\n"
            "- candidate_formula should reflect the likely formula shape using snake_case operand names when possible.\n"
            "- required_operands should list the operands needed to answer the question.\n"
            "- found_operands should include operands already grounded in evidence.\n"
            "- missing_operands should include operands that are still needed.\n"
            "- follow_up_local_queries should be short targeted searches for the missing operands.\n"
            "- enough_evidence_to_answer should be true only when the evidence is sufficient for a grounded final answer.\n"
            "- reasoning_notes should briefly explain why the evidence is or is not sufficient.\n\n"
            f"Question:\n{state['request'].question}\n\n"
            f"Current retrieval iteration: {next_iteration}\n"
            f"Coverage notes:\n" + ("\n".join(f"- {note}" for note in coverage_notes) if coverage_notes else "- None") + "\n\n"
            f"Evidence:\n{self._render_evidence_context(assessment_input, limit=ASSESSMENT_EVIDENCE_LIMIT)}\n"
        )

        try:
            assessment = self.coverage_llm.invoke(prompt)
        except Exception as exc:
            errors.append(f"Coverage assessment fallback used: {exc}")
            assessment = self._fallback_coverage_assessment(state)

        assessment = assessment.model_copy(
            update={
                "required_operands": list(dict.fromkeys(assessment.required_operands)),
                "found_operands": list(dict.fromkeys(assessment.found_operands)),
                "missing_operands": list(dict.fromkeys(assessment.missing_operands)),
                "follow_up_local_queries": list(dict.fromkeys(assessment.follow_up_local_queries)),
                "reasoning_notes": list(dict.fromkeys(assessment.reasoning_notes)),
            }
        )

        carryforward_queries = self._carryforward_queries(
            plan.candidate_files if plan is not None else [],
            plan.active_local_queries if plan is not None else [],
            state.get("searched_file_query_pairs", []),
        )
        expanded_follow_up_queries = self._dedupe_active_queries(
            list(assessment.follow_up_local_queries) + expand_missing_operand_queries(assessment.missing_operands),
            state.get("seen_query_fingerprints", []),
        )
        next_local_queries = self._dedupe_active_queries(carryforward_queries + expanded_follow_up_queries)
        updated_plan = plan
        if plan is not None:
            updated_plan = plan.model_copy(update={"active_local_queries": next_local_queries})

        all_pairs_exhausted = not updated_plan or not self._remaining_file_query_pairs_exist(
            updated_plan.candidate_files,
            next_local_queries,
            state.get("searched_file_query_pairs", []),
        )
        empty_pass_count = int(state.get("empty_retrieval_pass_count", 0))
        needs_more_evidence = (
            not assessment.enough_evidence_to_answer
            and next_iteration < MAX_RETRIEVAL_PASSES
            and empty_pass_count < MAX_EMPTY_PASSES
            and bool(next_local_queries)
            and not all_pairs_exhausted
        )

        prompt_evidence = self._select_prompt_evidence(
            evidence_pool,
            assessment,
            plan=updated_plan,
            limit=PROMPT_EVIDENCE_LIMIT,
        )
        notes = list(assessment.reasoning_notes)
        stop_reason = None
        if assessment.enough_evidence_to_answer:
            stop_reason = "coverage_complete"
        if not assessment.enough_evidence_to_answer and not next_local_queries:
            notes.append("Stopped retrieval because no new follow-up local queries remained after dedupe.")
            stop_reason = stop_reason or "no_new_queries"
        if not assessment.enough_evidence_to_answer and all_pairs_exhausted:
            notes.append("Stopped retrieval because all file/query combinations were exhausted.")
            stop_reason = stop_reason or "all_pairs_exhausted"
        if not assessment.enough_evidence_to_answer and empty_pass_count >= MAX_EMPTY_PASSES:
            notes.append(
                f"Stopped retrieval because the workflow reached the empty-pass budget of {MAX_EMPTY_PASSES}."
            )
            stop_reason = stop_reason or "empty_pass_budget_reached"
        if not assessment.enough_evidence_to_answer and next_iteration >= MAX_RETRIEVAL_PASSES:
            notes.append(f"Stopped retrieval because the loop budget of {MAX_RETRIEVAL_PASSES} passes was reached.")
            stop_reason = stop_reason or "pass_budget_reached"

        return {
            "retrieval_plan": updated_plan,
            "coverage_assessment": assessment,
            "coverage_notes": _unique_errors(coverage_notes + notes),
            "retrieval_iteration": next_iteration,
            "evidence": prompt_evidence,
            "evidence_pool": evidence_pool,
            "citations": [chunk.to_citation() for chunk in evidence_pool],
            "retrieval_stop_reason": None if needs_more_evidence else stop_reason,
            "status": "needs_more_evidence" if needs_more_evidence else "coverage_ready",
            "errors": _unique_errors(errors),
        }

    def _render_evidence_context(self, evidence: list[EvidenceChunk], limit: int = PROMPT_EVIDENCE_LIMIT) -> str:
        lines: list[str] = []
        for chunk in evidence[:limit]:
            snippet = " ".join(chunk.snippet.split())
            if len(snippet) > 900:
                snippet = snippet[:900].rstrip() + "... [truncated]"
            lines.append(
                f"[{chunk.citation_id}] source={chunk.source_label} page={chunk.page_number} "
                f"snippet={snippet}"
            )
        return "\n".join(lines)

    def reason_with_evidence(self, state: DiligenceState) -> dict[str, Any]:
        evidence = state.get("evidence") or self._select_prompt_evidence(
            state.get("evidence_pool", []),
            state.get("coverage_assessment"),
            plan=state.get("retrieval_plan"),
            limit=PROMPT_EVIDENCE_LIMIT,
        )
        errors = list(state["errors"])
        verification_notes = state.get("verification_notes", [])
        coverage = state.get("coverage_assessment")

        if not evidence:
            draft = ReasonedAnswer(
                answer_kind="unknown",
                proposed_answer="",
                formula="",
                operands=[],
                explanation=(
                    "The workflow could not retrieve enough grounded evidence to answer safely. "
                    "Escalating to human review is better than producing an unsupported number."
                ),
                citation_ids=[],
                assumptions=[],
                completion_status="incomplete",
                missing_operands=coverage.missing_operands if coverage else [],
                confidence=0.0,
            )
            return {
                "reasoned_answer": draft,
                "status": "reasoned_without_evidence",
                "errors": _unique_errors(errors),
            }

        coverage_block = "- None"
        if coverage is not None:
            coverage_block = "\n".join(
                [
                    f"- candidate_formula: {coverage.candidate_formula or 'None'}",
                    f"- required_operands: {coverage.required_operands or []}",
                    f"- found_operands: {coverage.found_operands or []}",
                    f"- missing_operands: {coverage.missing_operands or []}",
                    f"- enough_evidence_to_answer: {coverage.enough_evidence_to_answer}",
                ]
            )

        note_block = "\n".join(f"- {note}" for note in verification_notes) if verification_notes else "- None"
        prompt = (
            "You are a finance diligence assistant.\n"
            "Use only the retrieved evidence below.\n"
            "Produce a structured reasoning object that shows how you got the answer.\n"
            "Requirements:\n"
            "- If enough_evidence_to_answer is true, set completion_status=complete and answer_kind to number or percent.\n"
            "- If enough_evidence_to_answer is false, set completion_status=incomplete, answer_kind=unknown, leave proposed_answer empty, and list missing_operands.\n"
            "- proposed_answer must be digits only for number, or NN.NN% for percent.\n"
            "- operands must include the values you used, each with a snake_case name, numeric value, kind, and citation_id.\n"
            "- formula must use only operand names, numeric constants, parentheses, and + - * / operators.\n"
            "- citation_ids must reference only evidence IDs below and must include every operand citation_id.\n"
            "- explanation should summarize the evidence and arithmetic in 3-5 sentences.\n"
            "- assumptions should be empty unless the evidence explicitly forces an assumption.\n\n"
            f"Question:\n{state['request'].question}\n\n"
            f"Coverage assessment:\n{coverage_block}\n\n"
            f"Validation feedback to address:\n{note_block}\n\n"
            f"Evidence:\n{self._render_evidence_context(evidence)}\n"
        )

        try:
            draft = self.reasoning_llm.invoke(prompt)
        except Exception as exc:
            errors.append(f"Reasoning model fallback used: {exc}")
            draft = ReasonedAnswer(
                answer_kind="unknown",
                proposed_answer="",
                formula="",
                operands=[],
                explanation=(
                    "The reasoning step failed before it could translate the evidence into an answer. "
                    "This request should be reviewed manually."
                ),
                citation_ids=[],
                assumptions=[],
                completion_status="incomplete",
                missing_operands=coverage.missing_operands if coverage else [],
                confidence=0.0,
            )

        return {
            "reasoned_answer": draft,
            "evidence": evidence,
            "status": "reasoning_complete",
            "errors": _unique_errors(errors),
        }

    def validate_numeric_answer(self, state: DiligenceState) -> dict[str, Any]:
        validation = validate_reasoned_answer(
            state.get("reasoned_answer"),
            state.get("evidence_pool", state.get("evidence", [])),
        )
        return {
            "validation_result": validation,
            "status": "validation_complete" if not validation.issues else "validation_failed",
            "errors": _unique_errors(list(state["errors"]) + validation.issues),
        }

    def verify_answer(self, state: DiligenceState) -> dict[str, Any]:
        reasoned_answer = state.get("reasoned_answer")
        validation = state.get("validation_result")
        coverage_assessment = state.get("coverage_assessment")
        evidence = state.get("evidence_pool", state.get("evidence", []))
        existing_errors = list(state["errors"])
        evidence_by_id = {chunk.citation_id: chunk for chunk in evidence}
        issues = list(validation.issues if validation else [])

        if reasoned_answer is None:
            issues.append("Reasoning step did not produce an answer object.")
        else:
            if reasoned_answer.completion_status == "complete":
                if reasoned_answer.answer_kind not in {"number", "percent"}:
                    issues.append("Answer kind must be 'number' or 'percent'.")
                if parse_answer_value(reasoned_answer.answer_kind, reasoned_answer.proposed_answer) is None:
                    issues.append("Proposed answer is malformed for the declared answer kind.")
                if not reasoned_answer.citation_ids:
                    issues.append("Answer must cite at least one retrieved evidence chunk.")
            else:
                if reasoned_answer.answer_kind != "unknown":
                    issues.append("Incomplete answers must use answer_kind 'unknown'.")
                if not reasoned_answer.missing_operands:
                    issues.append("Incomplete answers must list missing_operands.")
            if (
                coverage_assessment is not None
                and not coverage_assessment.enough_evidence_to_answer
                and reasoned_answer.completion_status == "complete"
            ):
                issues.append("Answer was marked complete even though coverage assessment found evidence gaps.")

        if issues and state["attempt_count"] < 1:
            return {
                "attempt_count": state["attempt_count"] + 1,
                "status": "retry_reasoning",
                "verification_notes": _unique_errors(issues),
                "errors": _unique_errors(existing_errors + issues),
            }

        if reasoned_answer is None:
            response = DiligenceResponse(
                final_answer=None,
                answer_kind="unknown",
                explanation="The workflow could not reason over the evidence and has escalated the request for human review.",
                citations=[],
                confidence=0.0,
                needs_human_review=True,
                errors=_unique_errors(existing_errors + issues),
            )
            return {
                "final_response": response,
                "status": "respond_ready",
                "errors": response.errors,
            }

        valid_citation_ids = [citation_id for citation_id in reasoned_answer.citation_ids if citation_id in evidence_by_id]
        selected_citations = [evidence_by_id[citation_id].to_citation() for citation_id in valid_citation_ids]
        if not selected_citations and evidence:
            selected_citations = [evidence[0].to_citation()]

        incomplete_answer = reasoned_answer.completion_status != "complete"
        needs_human_review = bool(issues) or incomplete_answer or reasoned_answer.answer_kind == "unknown"
        confidence = min(reasoned_answer.confidence, 0.35) if needs_human_review else reasoned_answer.confidence
        explanation = reasoned_answer.explanation
        if validation and validation.recomputed_answer and not validation.matches_proposed_answer:
            explanation += f"\n\nValidation note: recomputed answer was {validation.recomputed_answer}."
        if incomplete_answer and reasoned_answer.missing_operands:
            explanation += "\n\nMissing operands: " + ", ".join(reasoned_answer.missing_operands)
        if issues:
            explanation += "\n\nVerification note: " + " ".join(_unique_errors(issues))

        response = DiligenceResponse(
            final_answer=reasoned_answer.proposed_answer or None,
            answer_kind=reasoned_answer.answer_kind,
            explanation=explanation,
            citations=selected_citations,
            confidence=confidence,
            needs_human_review=needs_human_review,
            errors=_unique_errors(existing_errors + issues),
        )
        return {
            "final_response": response,
            "status": "respond_ready",
            "errors": response.errors,
        }

    def respond(self, state: DiligenceState) -> dict[str, Any]:
        if state.get("final_response") is not None:
            return {"status": "done"}

        response = DiligenceResponse(
            final_answer=None,
            answer_kind="unknown",
            explanation=(
                "The workflow reached the response node without a verified answer. "
                "This request should be reviewed manually."
            ),
            citations=[],
            confidence=0.0,
            needs_human_review=True,
            errors=_unique_errors(state["errors"] + ["No verified response was produced."]),
        )
        return {
            "final_response": response,
            "status": "done",
            "errors": response.errors,
        }
