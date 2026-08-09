"""DocLens FastAPI backend.

Endpoints:
  POST   /api/upload             - upload a PDF, extract + chunk + summarize
  POST   /api/ask                - ask a question about a previously uploaded PDF
  GET    /api/documents          - list uploaded documents (this process's lifetime)
  DELETE /api/documents/{doc_id} - remove a document from memory
  GET    /api/health             - liveness check

Documents live in an in-memory store for the lifetime of the process — there
is no database. Restarting the backend clears everything, which is expected
for this single-user demo app.
"""

import uuid
from dataclasses import dataclass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.backend import claude_client
from src.backend.config import settings
from src.backend.pdf_utils import extract_text_from_pdf
from src.backend.rag import TfidfIndex, chunk_text
from src.backend.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentUploadResponse,
    SourceChunk,
)

app = FastAPI(title="DocLens API", version="1.0.0")

# Streamlit runs on a different port during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class DocumentRecord:
    filename: str
    num_pages: int
    chunks: list[str]
    index: TfidfIndex
    summary: str


DOCUMENT_STORE: dict[str, DocumentRecord] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        extracted = extract_text_from_pdf(file_bytes)
    except Exception as exc:  # pdfplumber can raise a variety of parser errors
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    if not extracted.full_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text found in this PDF. It may be a scanned "
                "image without an OCR text layer, which this app doesn't support."
            ),
        )

    chunks = chunk_text(extracted.full_text, settings.chunk_size, settings.chunk_overlap)
    index = TfidfIndex.build(chunks)

    truncated = len(extracted.full_text) > settings.max_summary_chars
    summary_source = extracted.full_text[: settings.max_summary_chars]

    try:
        summary = claude_client.generate_summary(summary_source)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    document_id = str(uuid.uuid4())
    DOCUMENT_STORE[document_id] = DocumentRecord(
        filename=file.filename or "document.pdf",
        num_pages=extracted.num_pages,
        chunks=chunks,
        index=index,
        summary=summary,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "document.pdf",
        num_pages=extracted.num_pages,
        num_chunks=len(chunks),
        summary=summary,
        truncated_for_summary=truncated,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    record = DOCUMENT_STORE.get(request.document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again.")

    top_k = request.top_k or settings.top_k_chunks
    results = record.index.query(request.question, top_k)

    if not results:
        raise HTTPException(status_code=422, detail="Document has no retrievable content.")

    context_chunks = [chunk for _, _, chunk in results]

    try:
        answer = claude_client.answer_question(request.question, context_chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceChunk(chunk_index=idx, score=round(score, 4), text=chunk)
        for idx, score, chunk in results
    ]
    return AskResponse(answer=answer, sources=sources)


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [
        DocumentInfo(
            document_id=doc_id,
            filename=record.filename,
            num_pages=record.num_pages,
            num_chunks=len(record.chunks),
        )
        for doc_id, record in DOCUMENT_STORE.items()
    ]


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    if document_id not in DOCUMENT_STORE:
        raise HTTPException(status_code=404, detail="Document not found.")
    del DOCUMENT_STORE[document_id]
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.backend.api:app", host=settings.backend_host, port=settings.backend_port)
