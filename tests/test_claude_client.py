from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backend import claude_client


@pytest.fixture(autouse=True)
def reset_client_singleton():
    """claude_client caches its Anthropic client in a module-level global;
    reset it around each test so tests don't leak state into each other."""
    claude_client._client = None
    yield
    claude_client._client = None


def _fake_text_response(text: str):
    """Build an object shaped like anthropic's Message response, just enough
    for _extract_text to work: a list of content blocks with type/text."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


def test_get_client_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(claude_client.settings, "anthropic_api_key", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        claude_client.get_client()


def test_generate_summary_sends_document_text_and_returns_response_text(monkeypatch):
    monkeypatch.setattr(claude_client.settings, "anthropic_api_key", "sk-ant-fake-key")

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_text_response("This document is about X.")
    monkeypatch.setattr(claude_client, "_client", mock_client)

    result = claude_client.generate_summary("Some extracted document text.")

    assert result == "This document is about X."
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "Some extracted document text." in call_kwargs["messages"][0]["content"]


def test_answer_question_includes_all_context_chunks_in_prompt(monkeypatch):
    monkeypatch.setattr(claude_client.settings, "anthropic_api_key", "sk-ant-fake-key")

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_text_response("The answer is 12%.")
    monkeypatch.setattr(claude_client, "_client", mock_client)

    result = claude_client.answer_question(
        "What was the growth?", ["Chunk about growth.", "Unrelated chunk."]
    )

    assert result == "The answer is 12%."
    prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Chunk about growth." in prompt
    assert "Unrelated chunk." in prompt
    assert "What was the growth?" in prompt
