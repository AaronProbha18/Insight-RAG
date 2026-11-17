from itertools import pairwise
from pathlib import Path

import pytest

from insight_rag.chunking import chunk
from insight_rag.connectors import ParsedDocument, Section
from insight_rag.connectors.builtin.docx import DocxConnector

CORPUS = Path(__file__).parents[2] / "evals" / "connectors" / "docx" / "corpus"


def _doc(*sections: tuple[str, str]) -> ParsedDocument:
    return ParsedDocument(
        title="t", sections=[Section(text=t, locator=loc) for t, loc in sections]
    )


def test_short_sections_one_chunk_each():
    chunks = chunk(_doc(("one", "p. 1"), ("two", "p. 2"), ("three", "p. 3")))

    assert [(c.ordinal, c.text, c.locator) for c in chunks] == [
        (0, "one", "p. 1"),
        (1, "two", "p. 2"),
        (2, "three", "p. 3"),
    ]


def test_long_section_splits_with_overlap():
    words = [f"w{i}" for i in range(500)]
    text = " ".join(words)  # ~2,400 characters

    chunks = chunk(_doc((text, "p. 4")), size=300, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 and c.locator == "p. 4" for c in chunks)
    for a, b in pairwise(chunks):
        assert a.text.split()[-1] in b.text.split()  # neighbours overlap
    assert {w for c in chunks for w in c.text.split()} == set(words)  # nothing lost


def test_never_crosses_sections():
    chunks = chunk(
        _doc(("alpha " * 100, "A"), ("beta " * 100, "B")), size=200, overlap=20
    )

    for c in chunks:
        assert ("alpha" in c.text) != ("beta" in c.text)
        assert c.locator == ("A" if "alpha" in c.text else "B")


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_params_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk(_doc(("x", "p. 1")), size=size, overlap=overlap)


def test_real_corpus_locators():
    doc = DocxConnector().parse((CORPUS / "employee-handbook.docx").read_bytes(), "h")

    chunks = chunk(doc, size=200, overlap=30)

    assert {c.locator for c in chunks} == {s.locator for s in doc.sections}
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
