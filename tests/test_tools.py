from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from gtm_diligence_assistant.tools import (
    get_full_pdf_text_impl,
    read_pdf_pages_impl,
    scan_pdf_pages_impl,
    search_document_pages_impl,
)


def _create_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class ToolTests(unittest.TestCase):
    def test_search_document_pages_returns_best_matching_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "example.pdf"
            _create_pdf(
                pdf_path,
                [
                    "Overview page with general company context.",
                    "Net debt equals short-term debt plus long-term debt plus operating lease liabilities minus cash and equivalents.",
                    "Appendix page.",
                ],
            )
            hits = search_document_pages_impl(str(pdf_path), "What is net debt including operating lease liabilities and cash?")
            self.assertTrue(hits)
            self.assertEqual(hits[0]["page_number"], 2)

    def test_read_pdf_pages_returns_requested_one_based_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "example.pdf"
            _create_pdf(pdf_path, ["First page", "Second page", "Third page"])
            extracted = read_pdf_pages_impl(str(pdf_path), [2, 3], max_chars=2000)
            self.assertEqual([page["page_number"] for page in extracted], [2, 3])
            self.assertIn("Second page", extracted[0]["text"])

    def test_get_full_pdf_text_returns_page_backed_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "example.pdf"
            _create_pdf(pdf_path, ["First page", "Second page"])
            extracted = get_full_pdf_text_impl(str(pdf_path), max_chars_per_page=2000)
            self.assertEqual([page["page_number"] for page in extracted], [1, 2])
            self.assertIn("Second page", extracted[1]["text"])

    def test_scan_pdf_pages_returns_phrase_and_regex_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "example.pdf"
            _create_pdf(
                pdf_path,
                [
                    "Overview page.",
                    "Total debt outstanding was 500 and borrowings were 200.",
                    "Operating lease liabilities were 50.",
                ],
            )
            hits = scan_pdf_pages_impl(
                str(pdf_path),
                search_terms=["long-term debt", "operating lease liabilities"],
                regex_patterns=[r"\b(total[-\s]+debt\s+outstanding|borrowings?)\b"],
                token_bundle_queries=["debt lease liabilities"],
                snippet_chars=200,
            )
            self.assertEqual([hit["page_number"] for hit in hits[:2]], [3, 2])
            self.assertTrue(any("regex:" in reason for reason in hits[1]["match_reasons"]))

    def test_read_pdf_pages_is_unclipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "long.pdf"
            long_text = "Debt table " + ("1234567890 " * 60)
            _create_pdf(pdf_path, [long_text])
            clipped = read_pdf_pages_impl(str(pdf_path), [1], max_chars=20)
            extracted = read_pdf_pages_impl(str(pdf_path), [1])
            self.assertTrue(clipped[0]["text"].endswith("... [truncated]"))
            self.assertGreater(len(extracted[0]["text"]), len(clipped[0]["text"]))
            self.assertFalse(extracted[0]["text"].endswith("... [truncated]"))

    def test_real_adobe_debt_note_page_is_found_by_exact_scan(self) -> None:
        pdf_path = Path("dataroom/Adobe/FY 2024/adbe-10k-fy24-final.pdf")
        if not pdf_path.exists():
            self.skipTest("Adobe 10-K fixture not present.")
        heuristic_hits = search_document_pages_impl(
            str(pdf_path),
            "What is Adobe's net debt as of the FY 2024 10-K? cash and cash equivalents short-term investments long-term debt operating lease liabilities",
            top_k=8,
            snippet_chars=200,
        )
        exact_hits = scan_pdf_pages_impl(
            str(pdf_path),
            search_terms=["long-term debt", "borrowings", "debt outstanding"],
            regex_patterns=[r"\b(total[-\s]+debt(?:\s+outstanding)?|borrowings?)\b"],
            token_bundle_queries=["long-term debt borrowings debt due after one year"],
            snippet_chars=200,
        )
        self.assertNotIn(88, [hit["page_number"] for hit in heuristic_hits[:3]])
        self.assertIn(88, [hit["page_number"] for hit in exact_hits[:10]])


if __name__ == "__main__":
    unittest.main()
