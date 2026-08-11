# CLAUDE.md

Conventions and architecture notes for working on DocLens. Read this before making changes.

## What this is

DocLens is a small AI document analyzer: upload a PDF, get an AI-generated summary, then ask
questions about it. Answers are grounded with a lightweight retrieval-augmented-generation (RAG)
step over the document's own text.

## Architecture

Two processes, talking over plain HTTP:

- **Backend** (`src/backend/`) — FastAPI. Owns PDF parsing, chunking, retrieval, and all calls to
  the Claude API. Stateful only in memory (`DOCUMENT_STORE` in `api.py`) — there is no database.
  Restarting the backend drops every uploaded document.
- **Frontend** (`src/frontend/`) — Streamlit. A local/standalone thin client that calls the
  backend's `/api/*` endpoints via `requests`. It holds no document text itself, only
  `document_id` and display state in `st.session_state`. The deployed, public-facing UI is a
  separate React app ([doclens-ai-chat](https://github.com/emdej111/doclens-ai-chat)) that talks
  to this same backend — Streamlit is kept for local development and standalone use.

The frontend never imports from `src/backend` and never calls the Anthropic SDK directly — every
AI call goes through the backend. Keep it that way; it's what lets the backend be swapped for a
different frontend (or a CLI) later without touching AI logic.

### Module map (backend)

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` (pydantic-settings), loaded once from `.env` / env vars |
| `pdf_utils.py` | `extract_text_from_pdf` — pdfplumber wrapper, page-by-page text extraction |
| `rag.py` | `chunk_text`, `TfidfIndex` — chunking and retrieval, pure numpy, no external service |
| `claude_client.py` | `generate_summary`, `answer_question` — the only place that calls Anthropic |
| `schemas.py` | Pydantic request/response models shared by the API routes |
| `api.py` | FastAPI routes + the in-memory `DOCUMENT_STORE` |

## RAG approach — deliberately simple

Retrieval is TF-IDF cosine similarity computed with numpy (`rag.py`), not an embedding API call
or a vector database. This is intentional, not a placeholder to "upgrade later" without cause:

- Anthropic doesn't offer an embeddings endpoint, so "real" embeddings would mean adding another
  provider (Voyage AI, OpenAI, etc.) for what is a single-user, single-document demo.
- TF-IDF is exact, deterministic, and needs zero extra API calls or infrastructure — good tradeoffs
  at this scale (one document, a few dozen to a few hundred chunks).

If the project grows to multi-document corpora, persistent storage, or needs semantic (not just
lexical) matching, that's the point to introduce real embeddings + a vector store — don't do it
preemptively.

## Model

Configured via `CLAUDE_MODEL` in `.env`, default `claude-sonnet-4-6`. Both AI calls
(`claude_client.py`) are plain non-streaming `messages.create()` — no extended thinking, no tool
use. `max_tokens` is capped at 1024 for both summary and Q&A, well under the threshold where the
SDK requires streaming to avoid HTTP timeouts, so keep new calls non-streaming unless you raise
`max_tokens` significantly.

## Conventions

- **Python 3.11+**, type hints everywhere, `dataclass` for internal data, Pydantic for
  request/response models crossing the HTTP boundary.
- **No comments explaining what code does** — names should carry that. A comment is only for a
  non-obvious *why* (see `rag.py`'s smoothed-idf and normalization notes for the pattern).
- **Errors surface to the user, not swallowed.** Backend routes raise `HTTPException` with a
  specific `detail` message; the frontend reads `response.json()["detail"]` and shows it with
  `st.error`, rather than a generic "something went wrong."
- **Don't silently truncate.** Long documents get their summary generated from a prefix
  (`MAX_SUMMARY_CHARS`), and the API response's `truncated_for_summary` flag is surfaced in the UI
  — see `render_summary()` in `app.py`. If you add another place that trims input for length,
  follow the same pattern: flag it back to the caller.
- **Config lives in `.env`, not hardcoded.** Add new settings to `Settings` in `config.py` and to
  `.env.example` with a comment, never as a bare literal in application code.
- **Keep the frontend dumb.** UI logic and display formatting belong in `src/frontend/app.py`.
  Retrieval, chunking, and prompting belong in the backend. If you're about to write an `if` that
  branches on document content in `app.py`, it probably belongs server-side instead.

## Running locally

```bash
cp .env.example .env          # then add your ANTHROPIC_API_KEY
pip install -r requirements.txt

# terminal 1
uvicorn src.backend.api:app --reload

# terminal 2
streamlit run src/frontend/app.py
```

## Tests

Five test files cover the backend end to end:

- `tests/test_rag.py` — chunking and TF-IDF retrieval
- `tests/test_pdf_utils.py` — PDF text extraction
- `tests/test_claude_client.py` — summary/Q&A prompt construction, with the Anthropic client
  mocked (no real API calls, no cost)
- `tests/test_schemas.py` — Pydantic request/response validation
- `tests/test_api.py` — the FastAPI endpoints themselves via `TestClient`, covering the full
  upload → ask → list → delete flow and error cases (non-PDF upload, unknown document, etc.)

Nothing here calls the live Anthropic API or needs a running server — `claude_client` is mocked
in `test_api.py` and `test_claude_client.py` via `monkeypatch`. Run with:

```bash
PYTHONPATH=. pytest tests/ -v
```
