from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_local_env
from .embeddings import create_embedding_model
from .vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR, LocalVectorIndexManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute local page-window vector indexes for the dataroom.")
    parser.add_argument("--dataroom-root", default=os.environ.get("DATAROOM_ROOT", "dataroom"))
    parser.add_argument("--cache-dir", default=os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR))
    parser.add_argument("--embedding-provider", help="Optional override for EMBEDDING_PROVIDER=openai|google|fake.")
    parser.add_argument("--company", help="Optional company folder to build, e.g. Adobe.")
    parser.add_argument("--fy", type=int, help="Optional fiscal year to build, e.g. 2024.")
    parser.add_argument("--force", action="store_true", help="Rebuild indexes even when the source PDFs are unchanged.")
    return parser


def _filter_fy_dirs(all_fy_dirs: list[Path], company: str | None, fiscal_year: int | None) -> list[Path]:
    filtered = all_fy_dirs
    if company:
        filtered = [path for path in filtered if path.parent.name.lower() == company.lower()]
    if fiscal_year is not None:
        filtered = [path for path in filtered if str(fiscal_year) in path.name]
    return filtered


def main() -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    manager = LocalVectorIndexManager(
        embedding_model=create_embedding_model(args.embedding_provider),
        dataroom_root=Path(args.dataroom_root),
        cache_dir=Path(args.cache_dir),
    )
    fy_dirs = _filter_fy_dirs(manager.iter_fy_directories(), args.company, args.fy)
    if not fy_dirs:
        parser.exit(status=1, message="No matching FY directories were found for index building.\n")
        return 1

    results = [manager.build_fy_index(fy_dir, force=args.force) for fy_dir in fy_dirs]
    payload = {
        "dataroom_root": str(Path(args.dataroom_root)),
        "cache_dir": str(Path(args.cache_dir)),
        "total_fy_dirs": len(results),
        "rebuilt_count": sum(1 for result in results if result["rebuilt"]),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
