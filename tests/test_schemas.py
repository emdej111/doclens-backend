import pytest
from pydantic import ValidationError

from src.backend.schemas import AskRequest, AskResponse, DocumentUploadResponse, SourceChunk


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(document_id="abc-123", question="")


def test_ask_request_accepts_valid_question():
    request = AskRequest(document_id="abc-123", question="What is this about?")
    assert request.question == "What is this about?"
    assert request.top_k is None


def test_ask_request_top_k_is_optional():
    request = AskRequest(document_id="abc-123", question="Q?", top_k=6)
    assert request.top_k == 6


def test_document_upload_response_requires_all_fields():
    with pytest.raises(ValidationError):
        DocumentUploadResponse(document_id="abc", filename="f.pdf")  # type: ignore[call-arg]


def test_ask_response_serializes_sources_list():
    response = AskResponse(
        answer="Yes.",
        sources=[SourceChunk(chunk_index=0, score=0.91, text="Relevant excerpt.")],
    )
    assert response.sources[0].chunk_index == 0
    assert response.sources[0].score == 0.91
