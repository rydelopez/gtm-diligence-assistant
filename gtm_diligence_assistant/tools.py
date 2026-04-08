from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "what",
    "with",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]{1,}", text.lower())
    seen: list[str] = []
    for token in tokens:
        if token not in STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def _expanded_query_terms(query: str) -> set[str]:
    lower_query = query.lower()
    terms = set(_tokenize(query))
    if "net debt" in lower_query:
        terms.update({"debt", "cash", "equivalents", "lease", "liabilities"})
    if "return on equity" in lower_query or "roe" in lower_query:
        terms.update({"equity", "income", "stockholders", "shareholders"})
    if "revenue" in lower_query:
        terms.update({"revenue", "sales"})
    if "dividend" in lower_query or "payout ratio" in lower_query:
        terms.update({"dividends", "share", "diluted", "earnings", "eps"})
    if "growth" in lower_query:
        terms.update({"growth", "rate", "percent", "annual"})
    return terms


def _best_snippet(page_text: str, query_terms: set[str], snippet_chars: int) -> str:
    normalized_text = " ".join(page_text.split())
    if not normalized_text:
        return ""
    lower_text = normalized_text.lower()
    best_index = 0
    for term in query_terms:
        position = lower_text.find(term.lower())
        if position >= 0:
            best_index = position
            break
    start = max(best_index - snippet_chars // 3, 0)
    end = min(start + snippet_chars, len(normalized_text))
    return normalized_text[start:end].strip()


def search_document_pages_impl(file_path: str, query: str, top_k: int = 3, snippet_chars: int = 300) -> list[dict[str, Any]]:
    """Rank PDF pages by keyword overlap and return the top hits with snippets."""
    query_terms = _expanded_query_terms(query)
    if not query_terms:
        return []

    scored_pages: list[dict[str, Any]] = []
    with pymupdf.open(file_path) as document:
        for page_index in range(document.page_count):
            page_text = document.load_page(page_index).get_text("text")
            if not page_text.strip():
                continue
            page_tokens = set(_tokenize(page_text))
            overlap = query_terms & page_tokens
            if not overlap:
                continue
            score = round(len(overlap) / max(len(query_terms), 1), 4)
            scored_pages.append(
                {
                    "page_number": page_index + 1,
                    "score": score,
                    "snippet": _best_snippet(page_text, overlap, snippet_chars),
                }
            )

    scored_pages.sort(key=lambda item: (-item["score"], item["page_number"]))
    return scored_pages[:top_k]


def scan_pdf_pages_impl(
    file_path: str,
    search_terms: list[str],
    regex_patterns: list[str] | None = None,
    token_bundle_queries: list[str] | None = None,
    snippet_chars: int = 300,
) -> list[dict[str, Any]]:
    """Scan every page and return all phrase/regex/token matches with reasons."""
    cleaned_terms = [term.strip() for term in search_terms if str(term).strip()]
    compiled_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (regex_patterns or [])
        if str(pattern).strip()
    ]
    token_queries = [query.strip() for query in (token_bundle_queries or []) if str(query).strip()]

    results: list[dict[str, Any]] = []
    with pymupdf.open(file_path) as document:
        for page_index in range(document.page_count):
            page_text = document.load_page(page_index).get_text("text")
            if not page_text.strip():
                continue
            normalized_text = " ".join(page_text.split())
            lower_text = normalized_text.lower()
            match_reasons: list[str] = []
            match_mode = "phrase"
            score = 0.0

            phrase_matches = [term for term in cleaned_terms if term.lower() in lower_text]
            if phrase_matches:
                match_reasons.extend(f"phrase:{term}" for term in phrase_matches)
                score += 100.0 + (5.0 * len(phrase_matches))

            pattern_matches: list[str] = []
            for pattern in compiled_patterns:
                if pattern.search(normalized_text):
                    pattern_matches.append(pattern.pattern)
            if pattern_matches:
                match_reasons.extend(f"regex:{pattern}" for pattern in pattern_matches)
                score += 40.0 + (4.0 * len(pattern_matches))
                if not phrase_matches:
                    match_mode = "regex"

            token_score = 0.0
            token_match_query = ""
            page_tokens = set(_tokenize(page_text))
            if not phrase_matches and not pattern_matches and token_queries:
                for token_query in token_queries:
                    query_terms = _expanded_query_terms(token_query)
                    overlap = query_terms & page_tokens
                    if not overlap:
                        continue
                    normalized_overlap = len(overlap) / max(len(query_terms), 1)
                    if len(overlap) >= 2 or normalized_overlap >= 0.35:
                        if normalized_overlap > token_score:
                            token_score = normalized_overlap
                            token_match_query = token_query
                if token_score > 0.0:
                    match_mode = "token_bundle"
                    match_reasons.append(f"token_bundle:{token_match_query}")
                    score += round(token_score * 10.0, 4)

            if not match_reasons:
                continue

            highlight_terms = [reason.split(":", 1)[1] for reason in match_reasons]
            results.append(
                {
                    "page_number": page_index + 1,
                    "score": round(score, 4),
                    "match_mode": match_mode,
                    "match_reasons": match_reasons,
                    "snippet": _best_snippet(page_text, set(_tokenize(" ".join(highlight_terms))), snippet_chars),
                }
            )

    results.sort(
        key=lambda item: (
            0 if item["match_mode"] == "phrase" else 1 if item["match_mode"] == "regex" else 2,
            -item["score"],
            item["page_number"],
        )
    )
    return results


def read_pdf_pages_impl(file_path: str, pages: list[int], max_chars: int | None = None) -> list[dict[str, Any]]:
    """Read specific one-based PDF pages and return exact text unless a cap is provided."""
    extracted_pages: list[dict[str, Any]] = []
    unique_pages = sorted({page for page in pages if page > 0})
    with pymupdf.open(file_path) as document:
        for page_number in unique_pages:
            if page_number > document.page_count:
                continue
            page_text = document.load_page(page_number - 1).get_text("text").strip()
            if max_chars is not None and len(page_text) > max_chars:
                page_text = page_text[:max_chars].rstrip() + "... [truncated]"
            extracted_pages.append({"page_number": page_number, "text": page_text})
    return extracted_pages


def get_full_pdf_text_impl(
    file_path: str,
    max_chars_per_page: int | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Read an entire local PDF and return clipped page-backed text."""
    extracted_pages: list[dict[str, Any]] = []
    with pymupdf.open(file_path) as document:
        page_limit = document.page_count if max_pages is None else min(document.page_count, max_pages)
        for page_index in range(page_limit):
            page_text = document.load_page(page_index).get_text("text").strip()
            if max_chars_per_page is not None and len(page_text) > max_chars_per_page:
                page_text = page_text[:max_chars_per_page].rstrip() + "... [truncated]"
            extracted_pages.append({"page_number": page_index + 1, "text": page_text})
    return extracted_pages


class ScanPDFPagesInput(BaseModel):
    file_path: str = Field(description="Absolute path to a local PDF file.")
    search_terms: list[str] = Field(description="Exact phrases or aliases to match across PDF pages.")
    regex_patterns: list[str] = Field(default_factory=list, description="Regex-like patterns for table and note cues.")
    token_bundle_queries: list[str] = Field(
        default_factory=list,
        description="Fallback token bundle queries used only when phrase and regex matching are sparse.",
    )
    snippet_chars: int = Field(default=300, ge=100, le=1200)


class SearchDocumentPagesInput(BaseModel):
    file_path: str = Field(description="Absolute path to a local PDF file.")
    query: str = Field(description="Finance question or keyword query used to score relevant pages.")
    top_k: int = Field(default=3, ge=1, le=10)
    snippet_chars: int = Field(default=300, ge=100, le=1000)


class ReadPDFPagesInput(BaseModel):
    file_path: str = Field(description="Absolute path to a local PDF file.")
    pages: list[int] = Field(description="One-based page numbers to extract.")
    max_chars: int | None = Field(default=None, ge=500, le=20000)


class GetFullPDFTextInput(BaseModel):
    file_path: str = Field(description="Absolute path to a local PDF file.")
    max_chars_per_page: int | None = Field(default=None, ge=500, le=20000)
    max_pages: int | None = Field(default=None, ge=1, le=500)


scan_pdf_pages = tool(
    "scan_pdf_pages",
    args_schema=ScanPDFPagesInput,
    return_direct=False,
)(scan_pdf_pages_impl)

search_document_pages = tool(
    "search_document_pages",
    args_schema=SearchDocumentPagesInput,
    return_direct=False,
)(search_document_pages_impl)

read_pdf_pages = tool(
    "read_pdf_pages",
    args_schema=ReadPDFPagesInput,
    return_direct=False,
)(read_pdf_pages_impl)

get_full_pdf_text = tool(
    "get_full_pdf_text",
    args_schema=GetFullPDFTextInput,
    return_direct=False,
)(get_full_pdf_text_impl)
