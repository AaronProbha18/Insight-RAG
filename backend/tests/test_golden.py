from pathlib import Path

import pytest
import yaml

from plumbline.connectors.builtin.docx import DocxConnector
from plumbline.connectors.builtin.pdf import PdfConnector
from plumbline.evals.golden import load_golden, load_labelled

EVALS = Path(__file__).parents[2] / "evals"
CONNECTORS = {"pdf": PdfConnector(), "docx": DocxConnector()}


def _golden(kind):
    return load_golden(EVALS / "connectors" / kind / "golden.yaml")


def _sections(kind) -> dict[tuple[str, str], str]:
    """(document stem, locator) -> section text, from the real connector."""
    out = {}
    for f in (EVALS / "connectors" / kind / "corpus").glob(f"*.{kind}"):
        for s in CONNECTORS[kind].parse(f.read_bytes(), f.name).sections:
            out[(f.stem, s.locator)] = s.text
    return out


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_golden_sets_load(kind):
    items = _golden(kind)

    assert len(items) >= 30
    assert sum(not i.answerable for i in items) / len(items) >= 0.20


def test_pdf_and_docx_ask_the_same_questions():
    def key(i):
        return (i.id, i.question, i.answerable)

    assert [key(i) for i in _golden("pdf")] == [key(i) for i in _golden("docx")]


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_supporting_locators_exist(kind):
    sections = _sections(kind)

    for item in _golden(kind):
        for s in item.supporting:
            assert (s.document, s.locator) in sections, f"{item.id}: {s}"


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_supporting_text_contains_key_fact(kind):
    sections = _sections(kind)

    for item in (i for i in _golden(kind) if i.answerable):
        texts = [_flat(sections[(s.document, s.locator)]) for s in item.supporting]
        assert any(_flat(item.key_fact) in t for t in texts), (
            f"{item.id}: {item.key_fact!r}"
        )


def _flat(text: str) -> str:
    """Ignore layout: PDF wraps lines and splits table cells, DOCX joins cells with |."""
    return " ".join(text.replace("|", " ").split())


def _write(tmp_path, items):
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump(items))
    return path


def _item(n, answerable=True):
    return {
        "id": f"q{n}",
        "question": "q?",
        "answerable": answerable,
        "reference_answer": "a",
        "supporting": [{"document": "d", "locator": "p. 1"}] if answerable else [],
    }


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([{"id": "q1"}] + [_item(n) for n in range(2, 40)], "question"),
        ([_item(1)] * 2 + [_item(n, False) for n in range(2, 40)], "duplicate"),
        (
            [{**_item(1, False), "supporting": [{"document": "d", "locator": "x"}]}],
            "unanswerable",
        ),
        ([_item(n) for n in range(10)], "at least 30"),
        (
            [_item(n) for n in range(36)] + [_item(n, False) for n in range(36, 40)],
            "unanswerable",
        ),
    ],
)
def test_loader_rejects_bad_shapes(tmp_path, items, message):
    with pytest.raises(ValueError, match=message):
        load_golden(_write(tmp_path, items))


def test_labelled_loads_and_is_balanced():
    labelled = load_labelled(EVALS / "judge" / "labelled.yaml")

    for cases in (labelled.correctness, labelled.faithfulness):
        assert len(cases) == 10
        assert {c.label for c in cases} == {True, False}
    assert all(c.reference_answer for c in labelled.correctness)
    assert all(c.sources for c in labelled.faithfulness)
