from __future__ import annotations

import ast
import math
import re
from typing import Any

from .models import EvidenceChunk, ReasonedAnswer, ValidationResult
from .scoring import TIER_THRESHOLDS, is_exact_match, parse_answer_value, relative_percent_error


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FormulaValidationError(ValueError):
    pass


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_answer(answer_kind: str, value: float) -> str:
    if answer_kind == "percent":
        return f"{value:.2f}%"
    return _format_number(value)


def _is_supported_node(node: ast.AST) -> bool:
    return isinstance(
        node,
        (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Name,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
            ast.UAdd,
            ast.Load,
        ),
    )


def _evaluate_formula_node(node: ast.AST, operands: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_formula_node(node.body, operands)
    if isinstance(node, ast.Name):
        if node.id not in operands:
            raise FormulaValidationError(f"Formula references unknown operand '{node.id}'.")
        return float(operands[node.id])
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_formula_node(node.operand, operands)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _evaluate_formula_node(node.left, operands)
        right = _evaluate_formula_node(node.right, operands)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if math.isclose(right, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise FormulaValidationError("Formula divides by zero.")
        return left / right
    raise FormulaValidationError("Formula uses an unsupported expression shape.")


def safe_evaluate_formula(formula: str, operands: dict[str, float]) -> float:
    cleaned = formula.strip()
    if not cleaned:
        raise FormulaValidationError("Formula is required.")
    if len(cleaned) > 300:
        raise FormulaValidationError("Formula is too long to validate safely.")
    try:
        parsed = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise FormulaValidationError("Formula is malformed.") from exc
    for node in ast.walk(parsed):
        if not _is_supported_node(node):
            raise FormulaValidationError("Formula uses unsupported syntax.")
    return _evaluate_formula_node(parsed, operands)


def _answers_match(answer_kind: str, proposed_value: float, recomputed_value: float) -> tuple[bool, float]:
    if is_exact_match(answer_kind, proposed_value, recomputed_value):
        return True, 0.0
    error = relative_percent_error(proposed_value, recomputed_value)
    return error <= TIER_THRESHOLDS["strict"], error


def validate_reasoned_answer(
    reasoned_answer: ReasonedAnswer | None,
    evidence: list[EvidenceChunk],
) -> ValidationResult:
    issues: list[str] = []
    evidence_ids = {chunk.citation_id for chunk in evidence}

    if reasoned_answer is None:
        return ValidationResult(
            recomputed_answer=None,
            matches_proposed_answer=False,
            format_ok=False,
            citation_ids_valid=False,
            coverage_complete=False,
            issues=["Reasoning step did not produce an answer object."],
        )

    coverage_complete = reasoned_answer.completion_status == "complete"

    if not coverage_complete:
        citation_ids_valid = True
        cited_ids = list(reasoned_answer.citation_ids)
        if not cited_ids:
            issues.append("Incomplete answers must cite at least one retrieved evidence chunk.")
            citation_ids_valid = False
        else:
            for citation_id in cited_ids:
                if citation_id not in evidence_ids:
                    issues.append(f"Citation '{citation_id}' was not found in retrieved evidence.")
                    citation_ids_valid = False

        if reasoned_answer.answer_kind != "unknown":
            issues.append("Incomplete answers must use answer_kind 'unknown'.")
        if reasoned_answer.proposed_answer.strip():
            issues.append("Incomplete answers must not provide a proposed numeric answer.")
        if not reasoned_answer.missing_operands:
            issues.append("Incomplete answers must list missing_operands.")

        return ValidationResult(
            recomputed_answer=None,
            matches_proposed_answer=False,
            format_ok=not bool(reasoned_answer.proposed_answer.strip()),
            citation_ids_valid=citation_ids_valid,
            coverage_complete=False,
            issues=issues,
        )

    if reasoned_answer.answer_kind not in {"number", "percent"}:
        issues.append("Answer kind must be 'number' or 'percent'.")

    proposed_value = parse_answer_value(reasoned_answer.answer_kind, reasoned_answer.proposed_answer)
    format_ok = proposed_value is not None
    if not format_ok:
        issues.append("Proposed answer is malformed for the declared answer kind.")

    operand_names: set[str] = set()
    operands_by_name: dict[str, float] = {}
    citation_ids_valid = True
    for operand in reasoned_answer.operands:
        if not IDENTIFIER_RE.match(operand.name):
            issues.append(f"Operand name '{operand.name}' must be a simple identifier.")
        if operand.name in operand_names:
            issues.append(f"Operand name '{operand.name}' is duplicated.")
        operand_names.add(operand.name)
        operands_by_name[operand.name] = float(operand.value)
        if operand.citation_id not in evidence_ids:
            issues.append(f"Operand citation '{operand.citation_id}' was not found in retrieved evidence.")
            citation_ids_valid = False

    cited_ids = list(reasoned_answer.citation_ids)
    if not cited_ids:
        issues.append("Reasoned answer must cite at least one retrieved evidence chunk.")
        citation_ids_valid = False
    else:
        for citation_id in cited_ids:
            if citation_id not in evidence_ids:
                issues.append(f"Citation '{citation_id}' was not found in retrieved evidence.")
                citation_ids_valid = False

    for operand in reasoned_answer.operands:
        if operand.citation_id not in cited_ids:
            issues.append(f"Operand citation '{operand.citation_id}' must also appear in citation_ids.")
            citation_ids_valid = False

    recomputed_value: float | None = None
    recomputed_answer: str | None = None
    matches_proposed_answer = False
    match_error: float | None = None

    if not reasoned_answer.formula.strip():
        issues.append("Formula is required.")
    else:
        try:
            recomputed_value = safe_evaluate_formula(reasoned_answer.formula, operands_by_name)
            recomputed_answer = _format_answer(reasoned_answer.answer_kind, recomputed_value)
        except FormulaValidationError as exc:
            issues.append(str(exc))

    if format_ok and recomputed_value is not None and reasoned_answer.answer_kind in {"number", "percent"}:
        matches_proposed_answer, match_error = _answers_match(
            reasoned_answer.answer_kind,
            proposed_value,
            recomputed_value,
        )
        if not matches_proposed_answer:
            issues.append(
                "Proposed answer does not match the recomputed formula result"
                + (f" (relative percent error {match_error:.4f})." if match_error is not None else ".")
            )

    return ValidationResult(
        recomputed_answer=recomputed_answer,
        matches_proposed_answer=matches_proposed_answer,
        format_ok=format_ok,
        citation_ids_valid=citation_ids_valid,
        coverage_complete=coverage_complete,
        issues=issues,
        relative_percent_error=match_error,
    )
