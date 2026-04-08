from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gtm_diligence_assistant import (
    DiligenceRequest,
    DiligenceWorkflow,
    create_chat_model,
    create_embedding_model,
    load_local_env,
)
from gtm_diligence_assistant.dataset import DEFAULT_DATASET_JSONL, load_request_from_dataset
from gtm_diligence_assistant.vector_index import DEFAULT_VECTOR_INDEX_CACHE_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local-only GTM diligence assistant workflow.")
    parser.add_argument("--qid", type=int, help="Load a real example from the JSONL dataset by qid.")
    parser.add_argument("--dataset-jsonl", default=os.environ.get("DATASET_JSONL", DEFAULT_DATASET_JSONL))
    parser.add_argument("--company", help="Optional company override for an ad hoc request.")
    parser.add_argument("--fy", type=int, help="Optional fiscal year override for an ad hoc request.")
    parser.add_argument("--question", help="Question for an ad hoc request.")
    parser.add_argument("--dataroom-root", default=os.environ.get("DATAROOM_ROOT", "dataroom"))
    parser.add_argument("--model-provider", help="Optional override for MODEL=openai|anthropic|google.")
    parser.add_argument("--embedding-provider", help="Optional override for EMBEDDING_PROVIDER=openai|google|fake.")
    parser.add_argument(
        "--vector-index-cache-dir",
        default=os.environ.get("VECTOR_INDEX_CACHE_DIR", DEFAULT_VECTOR_INDEX_CACHE_DIR),
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON response.")
    return parser


def build_request_from_args(args: argparse.Namespace) -> DiligenceRequest:
    if args.qid is not None:
        return load_request_from_dataset(args.dataset_jsonl, args.qid)

    if not args.question:
        raise ValueError("For ad hoc runs, --question is required.")

    return DiligenceRequest(
        question=args.question,
        company=args.company,
        fiscal_year=args.fy,
    )


def main() -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    try:
        request = build_request_from_args(args)
        workflow = DiligenceWorkflow(
            llm=create_chat_model(args.model_provider),
            dataroom_root=Path(args.dataroom_root),
            embedding_model=create_embedding_model(args.embedding_provider),
            vector_index_cache_dir=Path(args.vector_index_cache_dir),
        )
        response = workflow.run_request(request)
    except Exception as exc:
        parser.exit(status=1, message=f"{exc}\n")
        return 1

    payload = {
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
