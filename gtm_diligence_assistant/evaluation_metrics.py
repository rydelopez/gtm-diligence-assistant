from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import Citation, DiligenceResponse


def expected_files_from_record(record: dict[str, Any] | None) -> list[str]:
    if not isinstance(record, dict):
        return []
    outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    values = []
    values.extend(outputs.get("correct_any_of_files", []) or [])
    values.extend(metadata.get("correct_any_of_files", []) or [])
    deduped: list[str] = []
    for value in values:
        filename = Path(str(value)).name
        if filename and filename not in deduped:
            deduped.append(filename)
    return deduped


def _response_citations(response: DiligenceResponse | dict[str, Any]) -> list[Citation | dict[str, Any]]:
    if isinstance(response, DiligenceResponse):
        return list(response.citations)
    if isinstance(response, dict):
        citations = response.get("citations")
        if isinstance(citations, list):
            return citations
    return []


def compute_file_metrics(
    response: DiligenceResponse | dict[str, Any],
    telemetry: dict[str, Any] | None,
    expected_files: list[str],
) -> dict[str, Any]:
    telemetry = telemetry or {}
    opened_files = [Path(str(path)).name for path in telemetry.get("opened_files", [])]
    opened_expected_file = any(path in expected_files for path in opened_files) if expected_files else False

    citation_labels: list[str] = []
    for citation in _response_citations(response):
        if isinstance(citation, Citation):
            citation_labels.append(citation.source_label)
        elif isinstance(citation, dict):
            label = citation.get("source_label")
            if label:
                citation_labels.append(str(label))

    main_cited_file = Counter(citation_labels).most_common(1)[0][0] if citation_labels else None
    primary_candidate_file = telemetry.get("primary_candidate_file")
    primary_candidate_label = Path(str(primary_candidate_file)).name if primary_candidate_file else None
    primary_file_used = bool(telemetry.get("primary_file_used", False))

    primary_file_alignment = False
    if expected_files:
        if main_cited_file and main_cited_file in expected_files:
            primary_file_alignment = True
        elif primary_candidate_label and primary_candidate_label in expected_files and primary_file_used:
            primary_file_alignment = True

    expected_file_behavior = 0.0
    if primary_file_alignment:
        expected_file_behavior = 1.0
    elif opened_expected_file:
        expected_file_behavior = 0.5

    return {
        "expected_files": expected_files,
        "opened_files": opened_files,
        "opened_expected_file": opened_expected_file,
        "main_cited_file": main_cited_file,
        "primary_candidate_file": primary_candidate_label,
        "primary_file_alignment": primary_file_alignment,
        "expected_file_behavior": expected_file_behavior,
    }
