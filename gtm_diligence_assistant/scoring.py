from __future__ import annotations

import re
from typing import Any


PERCENT_LINE_RE = re.compile(
    r"^Final Answer \(percent\):\s*([-+]?\d+(?:\.\d+)?)\s*%$",
    re.MULTILINE,
)
NUMBER_LINE_RE = re.compile(r"^Final Answer \(number\):\s*(-?\d+(?:\.\d+)?)$", re.MULTILINE)
CURRENCY_LINE_RE = re.compile(r"^Final Answer \(currency\):\s*\$([0-9][\d,]*\.\d{2})$", re.MULTILINE)

TIER_THRESHOLDS = {
    "strict": 0.05,
    "near": 0.50,
    "loose": 1.50,
}


def parse_canonical_value(
    answer_text: str | None,
    expected_kind: str,
) -> tuple[float | None, bool, dict[str, Any]]:
    text = answer_text or ""
    meta: dict[str, Any] = {"matched": None}

    if expected_kind == "percent":
        match = PERCENT_LINE_RE.search(text)
        if match:
            meta["matched"] = "percent_line"
            return float(match.group(1)), True, meta

        inline_percent = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*%", text)
        if inline_percent:
            return float(inline_percent[-1]), False, meta

        bare_number = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if bare_number:
            value = float(bare_number[-1])
            return (value * 100.0 if 0 <= abs(value) <= 1.0 else value), False, meta
        return None, False, meta

    match = NUMBER_LINE_RE.search(text)
    if match:
        meta["matched"] = "number_line"
        return float(match.group(1)), True, meta

    match = CURRENCY_LINE_RE.search(text)
    if match:
        meta["matched"] = "currency_line"
        return float(match.group(1).replace(",", "")), False, meta

    bare_number = re.findall(r"\$?\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", text)
    if bare_number:
        return float(bare_number[-1].replace(",", "")), False, meta
    return None, False, meta


def parse_answer_value(answer_kind: str, value: str | None) -> float | None:
    answer_text = value or ""
    if answer_kind == "percent":
        answer_text = f"Final Answer (percent): {answer_text}"
    elif answer_kind == "number":
        answer_text = f"Final Answer (number): {answer_text}"
    parsed_value, _, _ = parse_canonical_value(answer_text, answer_kind)
    return parsed_value


def is_exact_match(expected_kind: str, got_value: float | None, expected_value: float | None) -> bool:
    if got_value is None or expected_value is None:
        return False
    if expected_kind == "percent":
        return round(got_value, 2) == round(expected_value, 2)
    decimals = 2 if (abs(expected_value) < 1 or abs(got_value) < 1) else 0
    return round(got_value, decimals) == round(expected_value, decimals)


def relative_percent_error(got: float | None, expected: float | None) -> float:
    if got is None or expected is None:
        return float("inf")
    denominator = max(abs(expected), 1.0)
    return 100.0 * abs(got - expected) / denominator


def evaluate_numeric_answer(
    answer_kind: str,
    final_answer: str | None,
    expected_kind: str | None,
    expected_value: float | None,
) -> dict[str, Any]:
    if not final_answer or not expected_kind or expected_value is None:
        return {
            "parsed_value": None,
            "exact": False,
            "relative_percent_error": None,
            "tier": "none",
            "within_tolerance": False,
        }

    parsed_value = parse_answer_value(answer_kind, final_answer)
    if parsed_value is None:
        return {
            "parsed_value": None,
            "exact": False,
            "relative_percent_error": None,
            "tier": "none",
            "within_tolerance": False,
        }

    if is_exact_match(expected_kind, parsed_value, expected_value):
        return {
            "parsed_value": parsed_value,
            "exact": True,
            "relative_percent_error": 0.0,
            "tier": "exact",
            "within_tolerance": True,
        }

    percent_error = relative_percent_error(parsed_value, expected_value)
    if percent_error <= TIER_THRESHOLDS["strict"]:
        tier = "strict"
    elif percent_error <= TIER_THRESHOLDS["near"]:
        tier = "near"
    elif percent_error <= TIER_THRESHOLDS["loose"]:
        tier = "loose"
    else:
        tier = "none"

    return {
        "parsed_value": parsed_value,
        "exact": False,
        "relative_percent_error": percent_error,
        "tier": tier,
        "within_tolerance": tier != "none",
    }
