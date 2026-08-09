# DocLens

An AI document analyzer: upload a PDF, get a summary, then ask questions about it. Answers are
grounded in the document via a lightweight retrieval step over its own text — no vector database
required.

## Features

- Upload a PDF and extract its text (`pdfplumber`)
- Generate an AI summary with Claude (`claude-sonnet-4-6` by default)
- Ask free-form questions about the document — answered using only the most relevant excerpts
  (retrieval-augmented generation via TF-IDF chunk similarity, computed with numpy)
- Clean Streamlit UI with chat-style Q&A and expandable source excerpts

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF processing | pdfplumber |
| AI | Anthropic Python SDK (Claude) |
| Retrieval | TF-IDF cosine similarity, pure numpy (no external vector DB) |

## Project structure

```
doclens/
├── requirements.txt
├── .env.example
├── CLAUDE.md              # architecture & conventions for contributors
├── README.md
├── src/
│   ├── backend/
│   │   ├── api.py         # FastAPI app and routes
│   │   ├── config.py      # env-based settings
│   │   ├── pdf_utils.py   # PDF text extraction
│   │   ├── rag.py         # chunking + TF-IDF retrieval
│   │   ├── claude_client.py  # summary + Q&A calls to Claude
│   │   └── schemas.py     # request/response models
│   └── frontend/
│       └── app.py         # Streamlit UI
└── tests/
    └── test_rag.py
```

## Setup

1. **Clone and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `ANTHROPIC_API_KEY` to a key from
   [console.anthropic.com](https://console.anthropic.com/).

3. **Run the backend** (terminal 1)

   ```bash
   uvicorn src.backend.api:app --reload
   ```

   The API is now live at `http://localhost:8000` (docs at `/docs`).

4. **Run the frontend** (terminal 2)

   ```bash
   streamlit run src/frontend/app.py
   ```

   Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Upload a PDF from the sidebar and click **Analyze document**.
2. Read the generated summary.
3. Ask questions in the chat box — each answer shows the source excerpts it was grounded in.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required, your Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | model used for summaries and Q&A |
| `BACKEND_URL` | `http://localhost:8000` | where the frontend looks for the backend |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | character-based chunking for retrieval |
| `TOP_K_CHUNKS` | `4` | how many chunks are retrieved per question |
| `MAX_SUMMARY_CHARS` | `60000` | prefix length used when a document is too long to summarize whole |

## How retrieval works

There's no vector database. On upload, the backend splits the document into overlapping chunks
and builds a TF-IDF matrix in memory with numpy. Each question is vectorized the same way and
scored against every chunk by cosine similarity; the top-scoring chunks are passed to Claude as
context. This is exact, fast, and dependency-free at the scale of one document — see `CLAUDE.md`
for the reasoning and when it'd be worth swapping in real embeddings.

## Notes

- Documents are stored **in memory only** — restarting the backend clears everything uploaded.
- Scanned PDFs with no text layer (image-only) aren't supported; there's no OCR step.
- Very long documents are summarized from a truncated prefix (`MAX_SUMMARY_CHARS`); the UI tells
  you when this happens. Q&A retrieval always searches the *full* document, not just the prefix.
