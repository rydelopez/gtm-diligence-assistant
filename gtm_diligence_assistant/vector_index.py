from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


DEFAULT_VECTOR_INDEX_CACHE_DIR = ".vector_indexes"
DEFAULT_EMBEDDING_DOC_MAX_CHARS = 12000
DEFAULT_EMBEDDING_ADD_BATCH_SIZE = 8
DEFAULT_EMBEDDING_BATCH_MAX_RETRIES = 4
DEFAULT_EMBEDDING_BATCH_BACKOFF_SECONDS = 2.0


@dataclass
class LoadedVectorIndex:
    vector_store: InMemoryVectorStore
    manifest: dict[str, Any]
    index_path: Path
    manifest_path: Path


def _normalize_cache_segment(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


def _parse_fiscal_year(value: str) -> int | None:
    match = re.search(r"(20\d{2})", value)
    return int(match.group(1)) if match else None


def _file_fingerprint(pdf_path: Path) -> dict[str, Any]:
    stat = pdf_path.stat()
    digest = hashlib.sha1()
    digest.update(str(pdf_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return {
        "file_path": str(pdf_path),
        "file_name": pdf_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint": digest.hexdigest(),
    }


def _fy_cache_directory(cache_root: Path, dataroom_root: Path, fy_dir: Path) -> Path:
    try:
        relative = fy_dir.resolve().relative_to(dataroom_root.resolve())
    except ValueError:
        relative = Path(fy_dir.name)
    slug = "__".join(_normalize_cache_segment(part) for part in relative.parts)
    key = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:10]
    return cache_root / f"{slug}-{key}"


class LocalVectorIndexManager:
    def __init__(
        self,
        embedding_model: Any,
        dataroom_root: str | Path = "dataroom",
        cache_dir: str | Path = DEFAULT_VECTOR_INDEX_CACHE_DIR,
    ) -> None:
        self.embedding_model = embedding_model
        self.dataroom_root = Path(dataroom_root)
        self.cache_dir = Path(cache_dir)
        self.embedding_doc_max_chars = int(
            os.environ.get("EMBEDDING_DOC_MAX_CHARS", str(DEFAULT_EMBEDDING_DOC_MAX_CHARS))
        )
        self.embedding_add_batch_size = max(
            1,
            int(os.environ.get("EMBEDDING_ADD_BATCH_SIZE", str(DEFAULT_EMBEDDING_ADD_BATCH_SIZE))),
        )
        self.embedding_batch_max_retries = max(
            1,
            int(os.environ.get("EMBEDDING_BATCH_MAX_RETRIES", str(DEFAULT_EMBEDDING_BATCH_MAX_RETRIES))),
        )
        self.embedding_batch_backoff_seconds = max(
            0.0,
            float(os.environ.get("EMBEDDING_BATCH_BACKOFF_SECONDS", str(DEFAULT_EMBEDDING_BATCH_BACKOFF_SECONDS))),
        )

    def iter_fy_directories(self) -> list[Path]:
        if not self.dataroom_root.exists():
            return []
        fy_dirs: list[Path] = []
        for company_dir in sorted(path for path in self.dataroom_root.iterdir() if path.is_dir()):
            for fy_dir in sorted(path for path in company_dir.iterdir() if path.is_dir()):
                fy_dirs.append(fy_dir)
        return fy_dirs

    def cache_paths_for_fy_dir(self, fy_dir: str | Path) -> tuple[Path, Path]:
        fy_path = Path(fy_dir)
        cache_dir = _fy_cache_directory(self.cache_dir, self.dataroom_root, fy_path)
        return cache_dir / "index.json", cache_dir / "manifest.json"

    def _current_manifest(self, fy_dir: Path) -> dict[str, Any]:
        company = fy_dir.parent.name
        fiscal_year = _parse_fiscal_year(fy_dir.name)
        pdfs = sorted(path for path in fy_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        return {
            "company": company,
            "fiscal_year": fiscal_year,
            "fy_dir": str(fy_dir),
            "pdfs": [_file_fingerprint(pdf_path) for pdf_path in pdfs],
        }

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any] | None:
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _page_window_documents(self, fy_dir: Path) -> tuple[list[Document], list[str]]:
        company = fy_dir.parent.name
        fiscal_year = _parse_fiscal_year(fy_dir.name)
        documents: list[Document] = []
        ids: list[str] = []

        pdfs = sorted(path for path in fy_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        for pdf_path in pdfs:
            fingerprint = _file_fingerprint(pdf_path)
            with pymupdf.open(pdf_path) as document:
                page_texts = [document.load_page(page_index).get_text("text").strip() for page_index in range(document.page_count)]

            for center_page in range(1, len(page_texts) + 1):
                window_start = max(1, center_page - 1)
                window_end = min(len(page_texts), center_page + 1)
                window_parts = []
                for page_number in range(window_start, window_end + 1):
                    page_text = page_texts[page_number - 1]
                    window_parts.append(f"Page {page_number}:\n{page_text}")
                page_content = (
                    f"Company: {company}\n"
                    f"Fiscal year: FY {fiscal_year or 'unknown'}\n"
                    f"Source file: {pdf_path.name}\n"
                    f"Center page: {center_page}\n"
                    f"Window pages: {window_start}-{window_end}\n\n"
                    + "\n\n".join(window_parts)
                )
                if len(page_content) > self.embedding_doc_max_chars:
                    page_content = page_content[: self.embedding_doc_max_chars].rstrip() + "\n\n[truncated for embedding index]"
                metadata = {
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "center_page": center_page,
                    "window_start": window_start,
                    "window_end": window_end,
                    "source_mtime_ns": fingerprint["mtime_ns"],
                    "source_fingerprint": fingerprint["fingerprint"],
                }
                relative_pdf = pdf_path.relative_to(fy_dir)
                doc_id = f"{relative_pdf.as_posix()}::page::{center_page}"
                documents.append(Document(page_content=page_content, metadata=metadata))
                ids.append(doc_id)

        return documents, ids

    def _add_documents_with_retries(
        self,
        vector_store: InMemoryVectorStore,
        documents: list[Document],
        ids: list[str],
    ) -> None:
        if not documents:
            return

        def _add_batch(batch_documents: list[Document], batch_ids: list[str]) -> None:
            last_error: Exception | None = None
            for attempt in range(1, self.embedding_batch_max_retries + 1):
                try:
                    vector_store.add_documents(documents=batch_documents, ids=batch_ids)
                    return
                except Exception as exc:  # pragma: no cover - exercised via tests with mock
                    last_error = exc
                    if attempt >= self.embedding_batch_max_retries:
                        break
                    sleep_seconds = self.embedding_batch_backoff_seconds * (2 ** (attempt - 1))
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

            if len(batch_documents) > 1:
                midpoint = max(1, len(batch_documents) // 2)
                _add_batch(batch_documents[:midpoint], batch_ids[:midpoint])
                _add_batch(batch_documents[midpoint:], batch_ids[midpoint:])
                return

            assert last_error is not None
            raise RuntimeError(
                f"Failed to add embedding batch for {batch_ids[0]} after "
                f"{self.embedding_batch_max_retries} retries."
            ) from last_error

        for start in range(0, len(documents), self.embedding_add_batch_size):
            end = start + self.embedding_add_batch_size
            _add_batch(documents[start:end], ids[start:end])

    def build_fy_index(self, fy_dir: str | Path, force: bool = False) -> dict[str, Any]:
        fy_path = Path(fy_dir)
        if not fy_path.exists():
            raise ValueError(f"FY directory not found: {fy_path}")

        index_path, manifest_path = self.cache_paths_for_fy_dir(fy_path)
        current_manifest = self._current_manifest(fy_path)
        existing_manifest = self._load_manifest(manifest_path)
        comparable_existing_manifest = (
            {key: existing_manifest.get(key) for key in ("company", "fiscal_year", "fy_dir", "pdfs")}
            if existing_manifest is not None
            else None
        )

        if not force and index_path.exists() and comparable_existing_manifest == current_manifest:
            return {
                "fy_dir": str(fy_path),
                "index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "rebuilt": False,
                "document_count": int(existing_manifest.get("document_count", 0)),
            }

        documents, ids = self._page_window_documents(fy_path)
        vector_store = InMemoryVectorStore(self.embedding_model)
        if documents:
            self._add_documents_with_retries(vector_store, documents, ids)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.dump(str(index_path))

        current_manifest["document_count"] = len(documents)
        current_manifest["index_backend"] = "InMemoryVectorStore"
        manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
        return {
            "fy_dir": str(fy_path),
            "index_path": str(index_path),
            "manifest_path": str(manifest_path),
            "rebuilt": True,
            "document_count": len(documents),
        }

    def load_fy_index(self, fy_dir: str | Path) -> LoadedVectorIndex | None:
        fy_path = Path(fy_dir)
        index_path, manifest_path = self.cache_paths_for_fy_dir(fy_path)
        if not index_path.exists() or not manifest_path.exists():
            return None
        manifest = self._load_manifest(manifest_path)
        if manifest is None:
            return None
        current_manifest = self._current_manifest(fy_path)
        comparable_manifest = {key: manifest.get(key) for key in ("company", "fiscal_year", "fy_dir", "pdfs")}
        if comparable_manifest != current_manifest:
            return None
        vector_store = InMemoryVectorStore.load(str(index_path), self.embedding_model)
        return LoadedVectorIndex(
            vector_store=vector_store,
            manifest=manifest,
            index_path=index_path,
            manifest_path=manifest_path,
        )
