from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from gtm_diligence_assistant.vector_index import LocalVectorIndexManager


def _create_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class KeywordEmbeddings(Embeddings):
    def _encode(self, text: str) -> list[float]:
        lower = text.lower()
        groups = [
            ("debt", "borrowings", "notes"),
            ("cash", "equivalents", "investments"),
            ("lease", "obligations", "liabilities"),
            ("revenue", "sales"),
        ]
        return [float(sum(lower.count(term) for term in group)) for group in groups]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class VectorIndexTests(unittest.TestCase):
    def test_build_and_load_page_window_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            cache_dir = Path(tmpdir) / "vector_indexes"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(
                pdf_path,
                [
                    "Overview and business discussion.",
                    "Borrowings were $ 5,000 and lease liabilities were $ 200.",
                    "Cash and cash equivalents were $ 300.",
                ],
            )

            manager = LocalVectorIndexManager(
                embedding_model=KeywordEmbeddings(),
                dataroom_root=dataroom_root,
                cache_dir=cache_dir,
            )
            first_build = manager.build_fy_index(fy_dir)
            second_build = manager.build_fy_index(fy_dir)
            loaded = manager.load_fy_index(fy_dir)

            self.assertTrue(first_build["rebuilt"])
            self.assertFalse(second_build["rebuilt"])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.manifest["document_count"], 3)

            hits = loaded.vector_store.similarity_search("borrowings and cash", k=2)
            self.assertTrue(hits)
            self.assertIn("center_page", hits[0].metadata)
            self.assertEqual(Path(hits[0].metadata["file_path"]).name, "Acme 2024 10-K.pdf")

    def test_rebuilds_when_pdf_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataroom_root = Path(tmpdir) / "dataroom"
            cache_dir = Path(tmpdir) / "vector_indexes"
            fy_dir = dataroom_root / "Acme" / "FY 2024"
            fy_dir.mkdir(parents=True)
            pdf_path = fy_dir / "Acme 2024 10-K.pdf"
            _create_pdf(pdf_path, ["Cash and cash equivalents were $ 300."])

            manager = LocalVectorIndexManager(
                embedding_model=KeywordEmbeddings(),
                dataroom_root=dataroom_root,
                cache_dir=cache_dir,
            )
            first_build = manager.build_fy_index(fy_dir)
            _create_pdf(pdf_path, ["Borrowings were $ 5,000 and cash was $ 300."])
            second_build = manager.build_fy_index(fy_dir)

            self.assertTrue(first_build["rebuilt"])
            self.assertTrue(second_build["rebuilt"])

    def test_add_documents_with_retries_splits_failing_batches(self) -> None:
        manager = LocalVectorIndexManager(
            embedding_model=KeywordEmbeddings(),
            dataroom_root="dataroom",
            cache_dir=".vector_indexes-test",
        )
        manager.embedding_add_batch_size = 4
        manager.embedding_batch_max_retries = 1
        manager.embedding_batch_backoff_seconds = 0.0

        documents = [Document(page_content=f"doc {index}", metadata={}) for index in range(4)]
        ids = [f"doc-{index}" for index in range(4)]

        class FlakyVectorStore:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def add_documents(self, documents, ids):  # noqa: ANN001
                self.batch_sizes.append(len(documents))
                if len(documents) > 1:
                    raise RuntimeError("Simulated provider failure for large batch")
                return ids

        vector_store = FlakyVectorStore()
        manager._add_documents_with_retries(vector_store, documents, ids)
        self.assertEqual(vector_store.batch_sizes[0], 4)
        self.assertIn(2, vector_store.batch_sizes[1:])
        self.assertGreaterEqual(vector_store.batch_sizes.count(1), 4)
