import { useEffect, useMemo, useState } from "react";

const emptyForm = {
  question: "",
};

function summarizeQuestion(question) {
  const normalized = question.toLowerCase();
  if (normalized.includes("share repurchase")) {
    return "Share repurchase";
  }
  if (normalized.includes("dividend")) {
    return "Dividend paydown";
  }
  if (normalized.includes("net debt")) {
    return "Net debt";
  }
  if (normalized.includes("roe")) {
    return "ROE";
  }
  if (normalized.includes("revenue")) {
    return "Revenue";
  }
  return "Question";
}

function formatExampleLabel(example) {
  const company = example.company || "Example";
  const details = [];
  if (example.fiscal_year) {
    details.push(`FY ${example.fiscal_year}`);
  }
  details.push(summarizeQuestion(example.question || ""));
  return [company, ...details].join(" · ");
}

function formatCitationLabel(citation) {
  const fileName = citation.source_label || citation.source_path?.split("/").at(-1) || "Source";
  return `${fileName} · Page ${citation.page_number}`;
}

function createRequestId(examples, selectedExampleId) {
  const selectedExample = examples.find((example) => example.id === selectedExampleId) || null;
  const base = selectedExample?.request_id || "web";
  return `${base}-${Date.now().toString(36)}`;
}

function buildRequestPayload(form, examples, selectedExampleId) {
  return {
    question: form.question.trim(),
    request_id: createRequestId(examples, selectedExampleId),
  };
}

function displayAnswerValue(answer) {
  if (answer.final_answer) {
    return answer.final_answer;
  }
  if (answer.needs_human_review) {
    return "Review required";
  }
  return "No answer available";
}

function LoadingBubble() {
  return (
    <div className="message-row assistant-row" aria-live="polite">
      <div className="avatar assistant-avatar" aria-hidden="true">
        AI
      </div>
      <div className="message-bubble assistant-bubble thinking-bubble">
        <div className="thinking-indicator" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <p>Thinking...</p>
      </div>
    </div>
  );
}

function App() {
  const [form, setForm] = useState(emptyForm);
  const [examples, setExamples] = useState([]);
  const [selectedExampleId, setSelectedExampleId] = useState("");
  const [examplesError, setExamplesError] = useState("");
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);
  const [submittedQuestion, setSubmittedQuestion] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadExamples() {
      try {
        const response = await fetch("/api/examples");
        if (!response.ok) {
          throw new Error(`Example loading failed with status ${response.status}.`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setExamples(payload.examples || []);
        }
      } catch (error) {
        if (!cancelled) {
          setExamplesError(error instanceof Error ? error.message : "Could not load examples.");
        }
      }
    }

    loadExamples();
    return () => {
      cancelled = true;
    };
  }, []);

  const renderedExamples = useMemo(
    () =>
      examples.map((example) => ({
        ...example,
        renderedLabel: formatExampleLabel(example),
      })),
    [examples],
  );

  function updateQuestion(value) {
    setForm({ question: value });
  }

  function handleExampleChange(event) {
    const nextId = event.target.value;
    setSelectedExampleId(nextId);
    const nextExample = examples.find((example) => example.id === nextId);
    if (!nextExample) {
      return;
    }
    setForm({
      question: nextExample.question || "",
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmedQuestion = form.question.trim();
    setApiError("");
    setLoading(true);
    setResult(null);
    setSubmittedQuestion(trimmedQuestion);

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildRequestPayload({ question: trimmedQuestion }, examples, selectedExampleId)),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Unable to answer right now.");
      }
      setResult(payload);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Unable to answer right now.");
    } finally {
      setLoading(false);
    }
  }

  const answer = result?.response;
  const hasQuestion = Boolean(submittedQuestion);

  return (
    <div className="app-shell">
      <div className="background-glow background-glow-left" />
      <div className="background-glow background-glow-right" />

      <main className="layout">
        <header className="hero">
          <h1>Diligence Assistant</h1>
        </header>

        <section className="chat-shell">
          <div className="chat-toolbar">
            <label className="field picker-field">
              <span>Example</span>
              <select value={selectedExampleId} onChange={handleExampleChange}>
                <option value="">Choose an example</option>
                {renderedExamples.map((example) => (
                  <option key={example.id || example.renderedLabel} value={example.id || ""}>
                    {example.renderedLabel}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <section className="chat-thread" aria-label="Conversation">
            {!hasQuestion && !loading && !apiError ? (
              <div className="empty-state">
                <p>Select an example or ask a question about a filing to get started.</p>
              </div>
            ) : null}

            {hasQuestion ? (
              <div className="message-row user-row">
                <div className="message-bubble user-bubble">
                  <p>{submittedQuestion}</p>
                </div>
              </div>
            ) : null}

            {loading ? <LoadingBubble /> : null}

            {!loading && answer ? (
              <div className="message-row assistant-row">
                <div className="avatar assistant-avatar" aria-hidden="true">
                  AI
                </div>
                <div className="message-bubble assistant-bubble">
                  <div className="answer-value">{displayAnswerValue(answer)}</div>
                  <p className="answer-explanation">{answer.explanation}</p>

                  <div className="citation-section">
                    <h2>Citations</h2>
                    {answer.citations.length === 0 ? (
                      <p className="subtle-note">No citations were returned for this answer.</p>
                    ) : (
                      <ul className="citation-list">
                        {answer.citations.map((citation) => (
                          <li key={citation.citation_id} className="citation-item">
                            <span>{formatCitationLabel(citation)}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            ) : null}

            {!loading && apiError ? (
              <div className="message-row assistant-row">
                <div className="avatar assistant-avatar" aria-hidden="true">
                  AI
                </div>
                <div className="message-bubble assistant-bubble failure-bubble">
                  <p>{apiError}</p>
                </div>
              </div>
            ) : null}
          </section>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="field field-question">
              <span>Question</span>
              <textarea
                value={form.question}
                onChange={(event) => updateQuestion(event.target.value)}
                placeholder="What is Adobe's net debt as of the FY 2024 10-K?"
                rows={4}
                required
              />
            </label>

            <div className="composer-actions">
              {examplesError ? <p className="subtle-note">{examplesError}</p> : <span />}
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "Thinking..." : "Send"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;
