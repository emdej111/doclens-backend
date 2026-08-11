"""Thin wrapper around the Anthropic SDK for DocLens' two AI calls:
document summarization and RAG-grounded question answering.
"""

import anthropic

from src.backend.config import settings

SUMMARY_MAX_TOKENS = 1024
ANSWER_MAX_TOKENS = 1024

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_summary(document_text: str) -> str:
    """Summarize a document's extracted text into a concise overview."""
    client = get_client()

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=(
            "You are a document analysis assistant. Summarize the document the "
            "user provides in 3-5 short paragraphs or a tight bullet list. Lead "
            "with what the document is and its purpose, then cover the key "
            "points, findings, or sections. Be concise and factual — don't "
            "speculate about content that isn't in the text. Respond in plain "
            "text only: no markdown formatting (no #, ##, **, -, or similar "
            "syntax), since the output is displayed as-is without a markdown "
            "renderer."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Summarize this document:\n\n{document_text}",
            }
        ],
    )

    return _extract_text(response)


def answer_question(question: str, context_chunks: list[str]) -> str:
    """Answer a question about a document using only the retrieved chunks as context."""
    client = get_client()

    context = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=ANSWER_MAX_TOKENS,
        system=(
            "You are a document Q&A assistant. Answer the user's question using "
            "only the excerpts provided below — they were retrieved from a larger "
            "document as the most relevant sections. If the excerpts don't contain "
            "enough information to answer, say so plainly rather than guessing. "
            "Respond in plain text only: no markdown formatting (no #, ##, **, -, "
            "or similar syntax), since the output is displayed as-is without a "
            "markdown renderer."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return _extract_text(response)


def _extract_text(response: anthropic.types.Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()

