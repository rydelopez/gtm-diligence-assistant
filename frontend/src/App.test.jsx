import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

describe("App", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(Date, "now").mockReturnValue(1712563200000);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    fetchMock.mockReset();
  });

  it("renders a single title and only the visible example and question inputs", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        examples: [
          {
            id: "example-1",
            qid: 6,
            company: "Adobe",
            fiscal_year: 2024,
            question: "What is Adobe's net debt?",
            request_id: "eval-qid-6",
            label: "Adobe",
          },
        ],
      }),
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("Diligence Assistant")).toBeInTheDocument());
    expect(screen.getAllByText("Diligence Assistant")).toHaveLength(1);
    expect(screen.getByLabelText("Example")).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toBeInTheDocument();
    expect(screen.queryByLabelText("Company")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Fiscal year")).not.toBeInTheDocument();
  });

  it("submits only question and request_id and renders a chat-style answer with explanation and citations", async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          examples: [
            {
              id: "example-1",
              qid: 6,
              company: "Adobe",
              fiscal_year: 2024,
              question: "What is Adobe's net debt?",
              request_id: "eval-qid-6",
              label: "Adobe",
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          response: {
            final_answer: "7682.00",
            answer_kind: "number",
            explanation: "Net debt was calculated from the cited pages.",
            confidence: 0.92,
            needs_human_review: false,
            errors: ["internal detail should stay hidden"],
            citations: [
              {
                citation_id: "local_pdf:Roper 2024 10K:29",
                source_type: "local_pdf",
                source_label: "Roper 2024 10K.pdf",
                source_path: "/data/Roper Technologies/FY 2024/Roper 2024 10K.pdf",
                page_number: 29,
                source_url: null,
              },
            ],
          },
          telemetry: { retrieval_stop_reason: "coverage_complete" },
        }),
      });

    render(<App />);

    await waitFor(() =>
      expect(screen.getByRole("option", { name: "Adobe · FY 2024 · Net debt" })).toBeInTheDocument(),
    );

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "example-1" } });
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What is net debt?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const runPayload = JSON.parse(fetchMock.mock.calls[1][1].body);

    expect(runPayload.question).toBe("What is net debt?");
    expect(runPayload.request_id).toMatch(/^eval-qid-6-/);

    await waitFor(() => expect(screen.getByText("7682.00")).toBeInTheDocument());
    expect(screen.getAllByText("What is net debt?")).toHaveLength(2);
    expect(screen.getByText("Net debt was calculated from the cited pages.")).toBeInTheDocument();
    expect(screen.getByText("Roper 2024 10K.pdf · Page 29")).toBeInTheDocument();
    expect(screen.queryByText("internal detail should stay hidden")).not.toBeInTheDocument();
  });

  it("shows a thinking placeholder and disables submit while the request is running", async () => {
    let resolveRun;
    const runPromise = new Promise((resolve) => {
      resolveRun = resolve;
    });

    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          examples: [
            {
              id: "example-1",
              qid: 9,
              company: "Adobe",
              fiscal_year: 2024,
              question: "What is Adobe's net debt?",
              request_id: "eval-qid-9",
              label: "Adobe",
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => runPromise,
      });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Question")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What is Adobe's net debt?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText((content, element) => element?.tagName === "P" && content === "Thinking...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Thinking..." })).toBeDisabled();

    resolveRun({
      response: {
        final_answer: "7682.00",
        answer_kind: "number",
        explanation: "Calculated from filings.",
        confidence: 0.9,
        needs_human_review: false,
        errors: [],
        citations: [],
      },
      telemetry: {},
    });

    await waitFor(() => expect(screen.getByText("7682.00")).toBeInTheDocument());
  });

  it("uses a generated request id when no example is selected", async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          examples: [
            {
              id: "example-1",
              qid: 5,
              company: "Roper Technologies",
              fiscal_year: 2024,
              question: "What is Roper Technologies's net debt?",
              request_id: "eval-qid-5",
              label: "Roper Technologies",
            },
            {
              id: "example-2",
              qid: 9,
              company: "Adobe",
              fiscal_year: 2024,
              question: "What is Adobe's net debt?",
              request_id: "eval-qid-9",
              label: "Adobe",
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          response: {
            final_answer: "7682.00",
            answer_kind: "number",
            explanation: "Calculated from filings.",
            confidence: 0.9,
            needs_human_review: false,
            errors: [],
            citations: [],
          },
          telemetry: {},
        }),
      });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Question")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What is Adobe's net debt in FY 2024?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const runPayload = JSON.parse(fetchMock.mock.calls[1][1].body);

    expect(runPayload.question).toBe("What is Adobe's net debt in FY 2024?");
    expect(runPayload.request_id).toMatch(/^web-/);
  });

  it("shows a friendly assistant failure message when the run fails", async () => {
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          examples: [
            {
              id: "example-1",
              qid: 9,
              company: "Adobe",
              fiscal_year: 2024,
              question: "What is Adobe's net debt?",
              request_id: "eval-qid-9",
              label: "Adobe",
            },
          ],
        }),
      })
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: "Workflow execution failed." }),
      });

    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("Question")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What is Adobe's net debt?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(screen.getByText("Workflow execution failed.")).toBeInTheDocument(),
    );
  });
});
