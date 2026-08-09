"""PDF text extraction using pdfplumber."""

from dataclasses import dataclass
from io import BytesIO

import pdfplumber


@dataclass
class ExtractedPdf:
    full_text: str
    page_texts: list[str]
    num_pages: int


def extract_text_from_pdf(file_bytes: bytes) -> ExtractedPdf:
    """Extract text page-by-page from a PDF's raw bytes.

    Pages that yield no extractable text (e.g. scanned images) contribute an
    empty string rather than being dropped, so page numbering in citations
    stays aligned with the source document.
    """
    page_texts: list[str] = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(text.strip())

    full_text = "\n\n".join(page_texts).strip()
    return ExtractedPdf(full_text=full_text, page_texts=page_texts, num_pages=len(page_texts))
