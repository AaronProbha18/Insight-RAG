from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from insight_rag.connectors import ParseError
from insight_rag.connectors.builtin.pdf import PdfConnector
from insight_rag.connectors.registry import load_connectors
from insight_rag.connectors.testing import run_conformance

CORPUS = Path(__file__).parents[2] / "evals" / "connectors" / "pdf" / "corpus"


def test_pdf_conformance():
    run_conformance(PdfConnector(), CORPUS)


def test_page_locators():
    doc = PdfConnector().parse(
        (CORPUS / "trail-ebike-manual.pdf").read_bytes(), "m.pdf"
    )

    assert [s.locator for s in doc.sections] == ["p. 1", "p. 2", "p. 3"]
    assert "Charging" in doc.sections[1].text
    assert doc.title == "Lumen Ridge Trail E-Bike Owner Manual"


def test_no_text_pdf_is_parse_error():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    with pytest.raises(ParseError, match="no extractable text"):
        PdfConnector().parse(buf.getvalue(), "blank.pdf")


def test_pdf_registered_via_entry_point():
    assert isinstance(load_connectors()["pdf"], PdfConnector)
