"""Pydantic request/response models for the DocLens API."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    summary: str
    truncated_for_summary: bool


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int


class AskRequest(BaseModel):
    document_id: str
    question: str = Field(..., min_length=1)
    top_k: int | None = None


class SourceChunk(BaseModel):
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
