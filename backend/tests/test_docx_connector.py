from io import BytesIO
from pathlib import Path

import docx
import pytest

from plumbline.connectors import ParseError
from plumbline.connectors.builtin.docx import DocxConnector
from plumbline.connectors.registry import load_connectors
from plumbline.connectors.testing import run_conformance

CORPUS = Path(__file__).parents[2] / "evals" / "connectors" / "docx" / "corpus"


def _sections(name: str) -> dict[str, str]:
    doc = DocxConnector().parse((CORPUS / name).read_bytes(), name)
    return {s.locator: s.text for s in doc.sections}


def test_docx_conformance():
    run_conformance(DocxConnector(), CORPUS)


def test_heading_path_locators():
    sections = _sections("employee-handbook.docx")

    assert "Working Hours > Store Staff" in sections
    assert "25 days of annual leave" in sections["Leave > Annual Leave"]
    assert not any(loc.startswith("Lumen Ridge") and " > " in loc for loc in sections)


def test_table_text_under_its_heading():
    sections = _sections("trail-ebike-manual.docx")

    assert "500 Wh removable lithium-ion" in sections["Specifications"]
    assert "E04 | Battery not detected" in sections["Display Error Codes"]


def test_bullets_kept():
    assert "- Gift cards" in _sections("returns-policy.docx")["Exceptions"]


def test_empty_docx_is_parse_error():
    buf = BytesIO()
    docx.Document().save(buf)

    with pytest.raises(ParseError, match="no text"):
        DocxConnector().parse(buf.getvalue(), "empty.docx")


def test_docx_registered_via_entry_point():
    assert isinstance(load_connectors()["docx"], DocxConnector)
