from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .batch import (
    DEFAULT_DATAROOM_ROOT,
    DEFAULT_OUT_JSONL,
    DEFAULT_OUT_SUMMARY_JSON,
    run_batch,
)
from .config import load_local_env
from .dataset import DEFAULT_DATASET_JSONL, load_dataset_records
from .embeddings import create_embedding_model
from .llm import create_chat_model
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR, LocalVectorIndexManager
from .workflow import DiligenceWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or refresh dataset-scoped vector indexes inside Docker, then run the batch workflow."
    )
    parser.add_argument("--dataset-jsonl", default=os.environ.get("DATASET_JSONL", DEFAULT_DATASET_JSONL))
    parser.add_argument("--dataroom-root", default=os.environ.get("DATAROOM_ROOT", DEFAULT_DATAROOM_ROOT))
    parser.add_argument(
        "--vector-index-cache-dir",
        default=os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR),
    )
    parser.add_argument("--out-jsonl", default=os.environ.get("OUT_JSONL", DEFAULT_OUT_JSONL))
    parser.add_argument("--out-summary-json", default=os.environ.get("OUT_SUMMARY_JSON", DEFAULT_OUT_SUMMARY_JSON))
    parser.add_argument("--embedding-provider", help="Optional override for EMBEDDING_PROVIDER=openai|google|fake.")
    parser.add_argument("--force", action="store_true", help="Rebuild indexes even when source PDFs are unchanged.")
    return parser


def _runtime_company_fiscal_year_pairs(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        company = metadata.get("company")
        fiscal_year = metadata.get("fiscal_year")
        if not company or fiscal_year is None:
            continue
        pair = (str(company), int(fiscal_year))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    return pairs


def _resolve_fy_dir(dataroom_root: Path, company: str, fiscal_year: int) -> Path | None:
    company_dir = dataroom_root / company
    if not company_dir.exists() or not company_dir.is_dir():
        return None

    exact_match = company_dir / f"FY {fiscal_year}"
    if exact_match.exists() and exact_match.is_dir():
        return exact_match

    candidates = sorted(path for path in company_dir.iterdir() if path.is_dir() and str(fiscal_year) in path.name)
    return candidates[0] if candidates else None


def prepare_runtime_indexes(
    dataset_jsonl: str | Path,
    dataroom_root: str | Path,
    vector_index_cache_dir: str | Path,
    *,
    embedding_provider: str | None = None,
    force: bool = False,
    create_embedding_model_fn: Callable[[str | None], Any] = create_embedding_model,
    manager_cls: type[LocalVectorIndexManager] = LocalVectorIndexManager,
) -> tuple[Any | None, dict[str, Any]]:
    records = load_dataset_records(dataset_jsonl)
    dataroom_root_path = Path(dataroom_root)
    cache_dir_path = Path(vector_index_cache_dir)

    requested_pairs = _runtime_company_fiscal_year_pairs(records)
    requested_paths = [str(dataroom_root_path / company / f"FY {fiscal_year}") for company, fiscal_year in requested_pairs]
    summary: dict[str, Any] = {
        "requested_fy_folders": requested_paths,
        "indexed_folders": [],
        "skipped_unchanged_folders": [],
        "failed_folders": [],
        "embeddings_available": False,
    }

    try:
        embedding_model = create_embedding_model_fn(embedding_provider)
    except Exception as exc:
        summary["embedding_error"] = str(exc)
        return None, summary

    summary["embeddings_available"] = embedding_model is not None
    if embedding_model is None:
        summary["embedding_error"] = "Embedding model was unavailable."
        return None, summary

    manager = manager_cls(
        embedding_model=embedding_model,
        dataroom_root=dataroom_root_path,
        cache_dir=cache_dir_path,
    )

    for company, fiscal_year in requested_pairs:
        resolved_fy_dir = _resolve_fy_dir(dataroom_root_path, company, fiscal_year)
        if resolved_fy_dir is None:
            summary["failed_folders"].append(
                {
                    "fy_dir": str(dataroom_root_path / company / f"FY {fiscal_year}"),
                    "error": "FY directory not found under the dataroom root.",
                }
            )
            continue

        try:
            result = manager.build_fy_index(resolved_fy_dir, force=force)
        except Exception as exc:
            summary["failed_folders"].append({"fy_dir": str(resolved_fy_dir), "error": str(exc)})
            continue

        if result.get("rebuilt"):
            summary["indexed_folders"].append(str(resolved_fy_dir))
        else:
            summary["skipped_unchanged_folders"].append(str(resolved_fy_dir))

    return embedding_model, summary


def run_docker_runtime(
    dataset_jsonl: str | Path,
    dataroom_root: str | Path,
    vector_index_cache_dir: str | Path,
    out_jsonl: str | Path,
    out_summary_json: str | Path,
    *,
    embedding_provider: str | None = None,
    force: bool = False,
    create_embedding_model_fn: Callable[[str | None], Any] = create_embedding_model,
    manager_cls: type[LocalVectorIndexManager] = LocalVectorIndexManager,
    workflow_factory: Callable[[Path, Path, Any | None], Any] | None = None,
) -> dict[str, Any]:
    embedding_model, index_prep_summary = prepare_runtime_indexes(
        dataset_jsonl=dataset_jsonl,
        dataroom_root=dataroom_root,
        vector_index_cache_dir=vector_index_cache_dir,
        embedding_provider=embedding_provider,
        force=force,
        create_embedding_model_fn=create_embedding_model_fn,
        manager_cls=manager_cls,
    )

    if not index_prep_summary.get("embeddings_available"):
        warning = index_prep_summary.get("embedding_error", "Embedding model was unavailable.")
        print(
            f"Warning: vector index preparation degraded to exact-scan mode. {warning}",
            file=sys.stderr,
        )
    if index_prep_summary.get("failed_folders"):
        print(
            "Warning: some FY folders could not be indexed and will use exact-scan fallback.",
            file=sys.stderr,
        )

    print(json.dumps({"event": "index_prep", **index_prep_summary}, indent=2), flush=True)

    dataroom_root_path = Path(dataroom_root)
    cache_dir_path = Path(vector_index_cache_dir)
    workflow = (
        workflow_factory(dataroom_root_path, cache_dir_path, embedding_model)
        if workflow_factory is not None
        else DiligenceWorkflow(
            create_chat_model(),
            dataroom_root=dataroom_root_path,
            embedding_model=embedding_model,
            vector_index_cache_dir=cache_dir_path,
        )
    )

    return run_batch(
        dataset_jsonl=dataset_jsonl,
        dataroom_root=dataroom_root,
        out_jsonl=out_jsonl,
        out_summary_json=out_summary_json,
        workflow=workflow,
        index_prep_summary=index_prep_summary,
    )


def main() -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    summary = run_docker_runtime(
        dataset_jsonl=args.dataset_jsonl,
        dataroom_root=args.dataroom_root,
        vector_index_cache_dir=args.vector_index_cache_dir,
        out_jsonl=args.out_jsonl,
        out_summary_json=args.out_summary_json,
        embedding_provider=args.embedding_provider,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
