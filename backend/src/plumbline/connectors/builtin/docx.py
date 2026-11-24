"""Built-in DOCX connector: one section per heading path, e.g. "Leave > Annual Leave".

Known gaps (v1): images, footnotes, headers/footers, text boxes, nested
tables, and headings that do not use the built-in "Heading N" styles.
"""

from io import BytesIO
from pathlib import Path
from typing import ClassVar

import docx
from docx.table import Table

from plumbline.connectors.contract import (
    CONTRACT_VERSION,
    ParsedDocument,
    ParseError,
    Section,
)


def _heading_level(style_name: str) -> int | None:
    name, _, level = style_name.rpartition(" ")
    return int(level) if name == "Heading" and level.isdigit() else None


class DocxConnector:
    name: ClassVar[str] = "docx"
    contract_version: ClassVar[int] = CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    extensions: ClassVar[tuple[str, ...]] = (".docx",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            document = docx.Document(BytesIO(data))
            blocks = list(document.iter_inner_content())
            core_title = document.core_properties.title
        except Exception as exc:  # any malformed upload is a ParseError, never a 500
            raise ParseError(f"unreadable DOCX: {exc}") from exc

        h1 = [
            b.text.strip()
            for b in blocks
            if not isinstance(b, Table) and _heading_level(b.style.name) == 1
        ]
        single_h1 = h1[0] if len(h1) == 1 else None  # one H1 = document title
        title = core_title or single_h1 or Path(filename).stem

        sections: dict[str, list[str]] = {}
        path: list[str] = []
        for block in blocks:
            if isinstance(block, Table):
                lines = [
                    " | ".join(c.text.strip() for c in row.cells) for row in block.rows
                ]
            else:
                text = block.text.strip()
                level = _heading_level(block.style.name)
                if level is not None:
                    if not (level == 1 and text == single_h1):
                        # Under a single-H1 title, H2 is the top of the path.
                        depth = level - 1 if single_h1 else level
                        path = path[: depth - 1] + [text]
                    continue
                if not text:
                    continue
                lines = [f"- {text}" if block.style.name == "List Bullet" else text]
            sections.setdefault(" > ".join(path) or title, []).extend(lines)

        if not sections:
            raise ParseError("no text found in document")
        return ParsedDocument(
            title=title,
            sections=[
                Section(text="\n".join(v), locator=k) for k, v in sections.items()
            ],
        )
