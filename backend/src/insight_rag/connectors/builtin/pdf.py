"""Built-in PDF connector: one section per page, located as "p. N"."""

import re
from io import BytesIO
from pathlib import Path
from typing import ClassVar

from pypdf import PdfReader

from insight_rag.connectors.contract import (
    CONTRACT_VERSION,
    ParsedDocument,
    ParseError,
    Section,
)

# Control characters except \t and \n (e.g. bullet glyphs extracted as \x7f).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class PdfConnector:
    name: ClassVar[str] = "pdf"
    contract_version: ClassVar[int] = CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]] = ("application/pdf",)
    extensions: ClassVar[tuple[str, ...]] = (".pdf",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            reader = PdfReader(BytesIO(data))
            pages = [
                _CONTROL.sub("", p.extract_text() or "").strip() for p in reader.pages
            ]
            title = (reader.metadata or {}).get("/Title")
        except Exception as exc:  # any malformed upload is a ParseError, never a 500
            raise ParseError(f"unreadable PDF: {exc}") from exc
        sections = [
            Section(text=text, locator=f"p. {n}")
            for n, text in enumerate(pages, 1)
            if text
        ]
        if not sections:
            raise ParseError("no extractable text (scanned PDFs are not supported)")
        return ParsedDocument(
            title=str(title or Path(filename).stem), sections=sections
        )
