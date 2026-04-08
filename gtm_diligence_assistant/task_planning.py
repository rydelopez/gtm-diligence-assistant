from __future__ import annotations

import re
from typing import Iterable


IDENTITY_FIELDS = {"question", "company", "fiscal_year"}

FINANCE_ALIASES: dict[str, list[str]] = {
    "revenue": ["net sales", "net revenues", "total revenue", "total revenues"],
    "cash and cash equivalents": ["cash equivalents", "cash", "cash and equivalents"],
    "short-term investments": ["short term investments", "marketable securities"],
    "operating lease liabilities": ["lease liabilities", "lease obligations", "operating lease obligations"],
    "total debt": ["current portion of long-term debt", "long-term debt", "borrowings"],
    "long-term debt": ["long term debt", "noncurrent borrowings", "borrowings"],
    "current debt": ["current portion of long-term debt", "short-term borrowings"],
    "total stockholders' equity": ["stockholders equity", "shareholders equity", "total equity"],
    "repurchases of common stock": ["share repurchases", "stock repurchases", "buybacks"],
    "cash dividends to stockholders": ["cash dividends", "dividends paid", "dividends to stockholders"],
}

TASK_TYPE_METRICS: dict[str, list[str]] = {
    "net_debt": [
        "total debt",
        "cash and cash equivalents",
        "short-term investments",
        "operating lease liabilities",
    ],
    "adjusted_net_debt_dividends": [
        "total debt",
        "cash and cash equivalents",
        "short-term investments",
        "operating lease liabilities",
        "cash dividends to stockholders",
    ],
    "adjusted_net_debt_share_repurchases": [
        "total debt",
        "cash and cash equivalents",
        "short-term investments",
        "operating lease liabilities",
        "repurchases of common stock",
    ],
    "revenue_projection": [
        "revenue",
        "producer price index",
        "growth rate",
    ],
    "roe_projection": [
        "net income",
        "total stockholders' equity",
        "labor productivity",
        "growth rate",
    ],
}

TASK_TYPE_LOCAL_QUERIES: dict[str, list[str]] = {
    "net_debt": [
        "consolidated balance sheets cash and cash equivalents short-term investments current portion of long-term debt long-term debt",
        "total debt cash and cash equivalents operating lease liabilities",
        "current operating lease liabilities total operating lease liabilities",
    ],
    "adjusted_net_debt_dividends": [
        "consolidated balance sheets cash and cash equivalents short-term investments current portion of long-term debt long-term debt",
        "total debt cash and cash equivalents operating lease liabilities",
        "current operating lease liabilities total operating lease liabilities",
        "cash dividends to stockholders financing activities statements of cash flows",
    ],
    "adjusted_net_debt_share_repurchases": [
        "consolidated balance sheets cash and cash equivalents short-term investments current portion of long-term debt long-term debt",
        "total debt cash and cash equivalents operating lease liabilities",
        "current operating lease liabilities total operating lease liabilities",
        "repurchases of common stock financing activities statements of cash flows",
    ],
    "revenue_projection": [
        "consolidated results revenues total revenue net sales",
        "consolidated statements of earnings total revenue net revenues",
    ],
    "roe_projection": [
        "consolidated statements of operations net income",
        "consolidated balance sheets total stockholders equity",
        "consolidated statements of stockholders equity total stockholders equity",
    ],
}

def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        if cleaned and cleaned.lower() not in {item.lower() for item in seen}:
            seen.append(cleaned)
    return seen


def normalize_task_type(question: str, llm_task_type: str | None = None) -> str:
    lower_question = question.lower()
    lower_task = (llm_task_type or "").strip().lower().replace(" ", "_")

    if "return on equity" in lower_question or " roe " in f" {lower_question} ":
        return "roe_projection"
    if "revenue" in lower_question and ("projected" in lower_question or "grow" in lower_question):
        return "revenue_projection"
    if "net debt" in lower_question and "dividend" in lower_question:
        return "adjusted_net_debt_dividends"
    if "net debt" in lower_question and (
        "share repurchase" in lower_question
        or "stock repurchase" in lower_question
        or "repurchases of common stock" in lower_question
        or "buyback" in lower_question
    ):
        return "adjusted_net_debt_share_repurchases"
    if "net debt" in lower_question:
        return "net_debt"
    if lower_task in TASK_TYPE_METRICS:
        return lower_task
    return lower_task or "unknown"


def merged_required_metrics(task_type: str, llm_metrics: list[str]) -> list[str]:
    return _dedupe_strings(list(llm_metrics) + TASK_TYPE_METRICS.get(task_type, []))


def expand_search_terms(text: str) -> list[str]:
    lowered = text.lower()
    alias_queries: list[str] = [text.strip()] if text and text.strip() else []
    for canonical, aliases in FINANCE_ALIASES.items():
        related_terms = [canonical, *aliases]
        if any(term in lowered for term in related_terms):
            alias_queries.append(" ".join(related_terms))
            alias_queries.extend(related_terms)
    return _dedupe_strings(alias_queries)


def build_local_queries(task_type: str, question: str, metrics: list[str]) -> list[str]:
    queries = [question]
    queries.extend(TASK_TYPE_LOCAL_QUERIES.get(task_type, []))
    queries.extend(metrics[:6])
    for metric in metrics[:6]:
        queries.extend(expand_search_terms(metric))
    queries.extend(expand_search_terms(question))
    return _dedupe_strings(queries)


def expand_missing_operand_queries(operands: Iterable[str]) -> list[str]:
    queries: list[str] = []
    for operand in operands:
        lowered = str(operand).strip().lower()
        if not lowered:
            continue
        phrase = lowered.replace("_", " ")
        queries.append(phrase)
        if "debt" in lowered or "borrow" in lowered:
            queries.extend(["long-term debt", "noncurrent borrowings", "borrowings"])
        if "lease" in lowered:
            queries.extend(["lease liabilities", "lease obligations", "operating lease liabilities"])
        if "cash" in lowered:
            queries.extend(["cash and cash equivalents", "cash equivalents"])
        if "investment" in lowered or "marketable" in lowered:
            queries.extend(["short-term investments", "marketable securities"])
        if "revenue" in lowered or "sales" in lowered:
            queries.extend(["revenue", "net sales", "net revenues"])
        if "equity" in lowered:
            queries.extend(["stockholders equity", "shareholders equity", "total equity"])
        if "dividend" in lowered:
            queries.extend(["cash dividends", "dividends paid"])
        if "repurchase" in lowered or "buyback" in lowered:
            queries.extend(["repurchases of common stock", "share repurchases", "buybacks"])
        queries.extend(expand_search_terms(phrase))
    return _dedupe_strings(queries)


def summarize_missing_field(field: str) -> str:
    normalized = re.sub(r"[^a-z_]+", "_", field.strip().lower()).strip("_")
    if normalized in {"fy", "year"}:
        return "fiscal_year"
    if "fiscal" in normalized and "year" in normalized:
        return "fiscal_year"
    if "company" in normalized or "issuer" in normalized or "entity" in normalized:
        return "company"
    if "question" in normalized or "prompt" in normalized:
        return "question"
    return normalized


def request_identity_missing_fields(fields: Iterable[str]) -> tuple[list[str], list[str]]:
    request_fields: list[str] = []
    retrievable_fields: list[str] = []
    seen_request_fields: set[str] = set()
    seen_retrievable_fields: set[str] = set()

    for field in fields:
        cleaned = " ".join(str(field).split()).strip()
        if not cleaned:
            continue
        summarized = summarize_missing_field(cleaned)
        if summarized in IDENTITY_FIELDS:
            if summarized not in seen_request_fields:
                request_fields.append(summarized)
                seen_request_fields.add(summarized)
        elif cleaned not in seen_retrievable_fields:
            retrievable_fields.append(cleaned)
            seen_retrievable_fields.add(cleaned)

    return request_fields, retrievable_fields
