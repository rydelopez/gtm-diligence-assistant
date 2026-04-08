from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from gtm_diligence_assistant.models import Citation, DiligenceResponse
from gtm_diligence_assistant.web_app import DemoExample, WebAppRuntime, create_app


class FakeRuntime(WebAppRuntime):
    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(
            workflow=None,
            examples=[
                DemoExample(
                    id="example-1",
                    qid=6,
                    company="Adobe",
                    fiscal_year=2024,
                    question="What is Adobe's net debt?",
                    request_id="eval-qid-6",
                    label="Adobe",
                )
            ],
            startup_error="workflow unavailable" if should_fail else None,
            index_prep_summary={"embeddings_available": True},
        )
        self.should_fail = should_fail
        self.requests = []

    def run_request(self, request):
        if self.should_fail:
            raise RuntimeError(self.startup_error or "workflow unavailable")
        self.requests.append(request)
        return (
            DiligenceResponse(
                final_answer="7682.00",
                answer_kind="number",
                explanation="Net debt was calculated from the cited pages.",
                citations=[
                    Citation(
                        citation_id="local_pdf:Roper 2024 10K:29",
                        source_type="local_pdf",
                        source_label="Roper 2024 10K.pdf",
                        source_path="/data/Roper Technologies/FY 2024/Roper 2024 10K.pdf",
                        page_number=29,
                    )
                ],
                confidence=0.9,
                needs_human_review=False,
                errors=[],
            ),
            {"retrieval_stop_reason": "coverage_complete"},
        )


class WebAppTests(unittest.TestCase):
    def test_examples_endpoint_returns_runtime_safe_examples(self) -> None:
        client = TestClient(create_app(runtime=FakeRuntime()))

        response = client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(len(payload["examples"]), 1)
        self.assertEqual(payload["examples"][0]["company"], "Adobe")
        self.assertNotIn("outputs", payload["examples"][0])
        self.assertNotIn("metadata", payload["examples"][0])

    def test_run_endpoint_returns_response_and_telemetry(self) -> None:
        runtime = FakeRuntime()
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/run",
            json={
                "question": "What is Adobe's net debt?",
                "request_id": "adhoc-adobe-2024",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"]["final_answer"], "7682.00")
        self.assertEqual(payload["telemetry"]["retrieval_stop_reason"], "coverage_complete")
        self.assertEqual(len(runtime.requests), 1)

    def test_run_endpoint_surfaces_runtime_failures_cleanly(self) -> None:
        client = TestClient(create_app(runtime=FakeRuntime(should_fail=True)))

        response = client.post(
            "/api/run",
            json={
                "question": "What is Adobe's net debt?",
                "request_id": "adhoc-adobe-2024",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "workflow unavailable")

    def test_health_endpoint_reports_runtime_state(self) -> None:
        client = TestClient(create_app(runtime=FakeRuntime()))
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["workflow_ready"])
        self.assertEqual(payload["examples_loaded"], 1)


if __name__ == "__main__":
    unittest.main()
