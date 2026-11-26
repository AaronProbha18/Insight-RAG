# Write a connector

A connector turns one kind of file into text sections that Plumbline can
chunk, embed and cite. PDF and DOCX ship built in; anything else (Markdown,
HTML, EPUB, …) can be a separate Python package that you install next to
Plumbline. Core never needs to change.

This guide covers **file connectors**: they parse uploaded bytes. Query
connectors for live sources such as SQL databases are planned, under a later
contract version.

## 1. The contract

Everything you need is importable from `plumbline.connectors`:

```python
CONTRACT_VERSION = 1

class Section(BaseModel):          # pydantic
    text: str                      # non-empty
    locator: str                   # non-empty, e.g. "p. 3" or "Install > Linux"

class ParsedDocument(BaseModel):
    title: str
    sections: list[Section]
    metadata: dict[str, str] = {}

class ParseError(Exception): ...   # raise this for input you cannot read

class FileConnector(Protocol):
    name: ClassVar[str]                    # unique, e.g. "markdown"
    contract_version: ClassVar[int]        # must equal CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]]
    extensions: ClassVar[tuple[str, ...]]  # lower-case, with the dot

    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...
```

Rules the conformance kit enforces:

- **Every section has a locator.** It's what users see in a citation, so
  make it something a person can follow back to the source: a page, a
  heading path, a sheet and row range.
- **At least one section per document.** If a file has no extractable
  text, raise `ParseError` instead of returning an empty document.
- **Deterministic.** The same bytes must always give the same output.
- **Unreadable input raises `ParseError`**, never another exception. Wrap
  your parser's own errors. The API turns `ParseError` into a 422 for that
  file; anything else would be a 500.

## 2. A complete example

A plain-text connector with one section per paragraph:

```python
# my_connectors/text.py
from typing import ClassVar

from plumbline.connectors import (
    CONTRACT_VERSION,
    ParsedDocument,
    ParseError,
    Section,
)


class TextConnector:
    name: ClassVar[str] = "text"
    contract_version: ClassVar[int] = CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]] = ("text/plain",)
    extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(f"not UTF-8 text: {exc}") from exc
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            raise ParseError("no text found")
        sections = [
            Section(text=p, locator=f"paragraph {n}") for n, p in enumerate(paragraphs, 1)
        ]
        return ParsedDocument(title=filename.rsplit(".", 1)[0], sections=sections)
```

## 3. Register it

Plumbline discovers connectors through the `plumbline.connectors`
entry-point group, which is the same mechanism its built-in PDF and DOCX
connectors use. In your package's `pyproject.toml`:

```toml
[project.entry-points."plumbline.connectors"]
text = "my_connectors.text:TextConnector"
```

Install your package into the same environment as Plumbline
(`uv pip install -e .` during development) and restart the backend. Its
extensions are accepted by `POST /documents` from then on.

At startup, a connector whose `contract_version` differs from the
installed core is **skipped with a warning** naming it and both versions;
startup carries on. So does a connector that fails to import.

## 4. Test it with the conformance kit

```python
# tests/test_text_connector.py
from pathlib import Path

from plumbline.connectors.testing import run_conformance
from my_connectors.text import TextConnector


def test_conformance():
    run_conformance(TextConnector(), Path(__file__).parent / "fixtures")
```

Fixture layout:

```
fixtures/
  notes.txt            valid files: must parse
  manual.txt
  corrupt/
    binary.txt         must raise ParseError
```

Only files whose extension is in your connector's `extensions` are used.
The kit fails with a message naming the file and the rule it broke:
contract version, no fixtures, no sections, an empty locator,
non-deterministic output, or a corrupt file that didn't raise `ParseError`.

## 5. From "works" to "supported"

Passing conformance proves your connector follows the contract. To be
listed as **supported**, it must also show its answers can be trusted:

1. Add a fixture corpus under `evals/connectors/<name>/corpus/`, written
   for this purpose or permissively licensed, with no real personal data.
2. Add `evals/connectors/<name>/golden.yaml`: at least 30 questions, at
   least 20% of them unanswerable from the corpus. Copy the shape from
   `evals/connectors/pdf/golden.yaml`.
3. Run `uv run plumbline-eval --connectors <name>` (see
   [evals/README.md](https://github.com/AaronProbha18/Plumbline/blob/main/evals/README.md)). Correctness, faithfulness and
   abstention must each reach **0.80**.
