from pathlib import Path

from src.backend.pdf_utils import extract_text_from_pdf

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_extract_text_from_pdf_returns_correct_page_count():
    result = extract_text_from_pdf(FIXTURE_PDF.read_bytes())
    assert result.num_pages == 2
    assert len(result.page_texts) == 2


def test_extract_text_from_pdf_preserves_page_order():
    result = extract_text_from_pdf(FIXTURE_PDF.read_bytes())
    assert "page one" in result.page_texts[0]
    assert "Page two" in result.page_texts[1]


def test_extract_text_from_pdf_joins_pages_into_full_text():
    result = extract_text_from_pdf(FIXTURE_PDF.read_bytes())
    assert "DocLens Test Fixture" in result.full_text
    assert "revenue growth" in result.full_text
    # Pages are joined with a blank line so text from different pages doesn't run together.
    assert "\n\n" in result.full_text
