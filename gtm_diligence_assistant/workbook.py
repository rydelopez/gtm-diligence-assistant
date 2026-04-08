from __future__ import annotations

"""Archival helpers for regenerating the curated JSONL fixture from the workbook.

This module is not used by the take-home runtime. The Docker-first runtime reads
`inputs/diligence_eval_examples.jsonl` directly so grader metadata stays outside
the agent request path.
"""

import json
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd


CURATED_EVAL_QIDS = (4, 5, 6, 8, 9)
URL_RE = re.compile(r"https?://[^\s,)]+")


def parse_links(cell: object) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    value = str(cell).strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            loaded = json.loads(value)
            return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    matches = URL_RE.findall(value)
    if matches:
        return list(dict.fromkeys(matches))
    return [part.strip() for part in re.split(r"[,\n;]+", value) if part.strip()]


def _parse_filename_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[;,]", text) if part.strip()]


def load_workbook(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)


def build_eval_examples(path: str | Path, qids: tuple[int, ...] = CURATED_EVAL_QIDS) -> list[dict[str, Any]]:
    dataframe = load_workbook(path)
    subset = dataframe[dataframe["qid"].isin(qids)].copy()
    examples: list[dict[str, Any]] = []

    for record in subset.to_dict(orient="records"):
        qid = int(record["qid"])
        correct_files = _parse_filename_list(record.get("correct_any_of_files"))
        examples.append(
            {
                "id": str(uuid5(NAMESPACE_URL, f"gtm-diligence-assistant:{qid}")),
                "inputs": {
                    "question": str(record["question"]).strip(),
                    "request_id": f"eval-qid-{qid}",
                },
                "outputs": {
                    "expected_kind": str(record["expected_kind"]).strip(),
                    "expected_value": float(record["expected_value"]),
                    "correct_any_of_files": correct_files,
                },
                "metadata": {
                    "qid": qid,
                    "company": str(record["company"]).strip(),
                    "fiscal_year": int(record["fy"]),
                    "correct_any_of_files": correct_files,
                },
            }
        )

    return examples
