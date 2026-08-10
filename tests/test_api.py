from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.backend import api

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture(autouse=True)
def clear_document_store():
    """The API keeps documents in a module-level in-memory dict. Reset it
    around every test so upload/list/delete tests don't leak into each
    other regardless of run order."""
    api.DOCUMENT_STORE.clear()
    yield
    api.DOCUMENT_STORE.clear()


@pytest.fixture(autouse=True)
def mock_claude(monkeypatch):
    """Every test in this file runs against a mocked Claude client so the
    test suite never makes a real, billed API call."""
    monkeypatch.setattr(api.claude_client, "generate_summary", lambda text: "A short summary.")
    monkeypatch.setattr(
        api.claude_client, "answer_question", lambda question, chunks: "A grounded answer."
    )


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_pdf_returns_summary_and_metadata(client):
    with FIXTURE_PDF.open("rb") as f:
        response = client.post(
            "/api/upload", files={"file": ("sample.pdf", f, "application/pdf")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["num_pages"] == 2
    assert body["summary"] == "A short summary."
    assert body["truncated_for_summary"] is False
    assert "document_id" in body


def test_upload_rejects_non_pdf_file(client):
    response = client.post(
        "/api/upload", files={"file": ("notes.txt", b"just some text", "text/plain")}
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/upload", files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400


def test_full_upload_then_ask_flow(client):
    with FIXTURE_PDF.open("rb") as f:
        upload_response = client.post(
            "/api/upload", files={"file": ("sample.pdf", f, "application/pdf")}
        )
    document_id = upload_response.json()["document_id"]

    ask_response = client.post(
        "/api/ask", json={"document_id": document_id, "question": "What is the revenue growth?"}
    )

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["answer"] == "A grounded answer."
    assert len(body["sources"]) > 0
    assert "chunk_index" in body["sources"][0]
    assert "score" in body["sources"][0]


def test_ask_returns_404_for_unknown_document(client):
    response = client.post(
        "/api/ask", json={"document_id": "does-not-exist", "question": "Anything?"}
    )
    assert response.status_code == 404


def test_list_documents_reflects_uploads(client):
    assert client.get("/api/documents").json() == []

    with FIXTURE_PDF.open("rb") as f:
        client.post("/api/upload", files={"file": ("sample.pdf", f, "application/pdf")})

    documents = client.get("/api/documents").json()
    assert len(documents) == 1
    assert documents[0]["filename"] == "sample.pdf"


def test_delete_document_removes_it_from_the_store(client):
    with FIXTURE_PDF.open("rb") as f:
        upload_response = client.post(
            "/api/upload", files={"file": ("sample.pdf", f, "application/pdf")}
        )
    document_id = upload_response.json()["document_id"]

    delete_response = client.delete(f"/api/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted"}
    assert client.get("/api/documents").json() == []


def test_delete_returns_404_for_unknown_document(client):
    response = client.delete("/api/documents/does-not-exist")
    assert response.status_code == 404
