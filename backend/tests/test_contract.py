from typing import ClassVar

import pytest
from pydantic import ValidationError

from plumbline.connectors import (
    CONTRACT_VERSION,
    FileConnector,
    ParsedDocument,
    ParseError,
    Section,
)


class TxtConnector:
    name: ClassVar[str] = "txt"
    contract_version: ClassVar[int] = CONTRACT_VERSION
    media_types: ClassVar[tuple[str, ...]] = ("text/plain",)
    extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(str(exc)) from exc
        return ParsedDocument(
            title=filename, sections=[Section(text=text, locator="line 1")]
        )


def _parse_with(connector: FileConnector, data: bytes) -> ParsedDocument:
    return connector.parse(data, "a.txt")


def test_empty_locator_rejected():
    with pytest.raises(ValidationError):
        Section(text="x", locator="")


def test_minimal_connector_satisfies_protocol():
    doc = _parse_with(TxtConnector(), b"hello")

    assert doc.sections == [Section(text="hello", locator="line 1")]
    with pytest.raises(ParseError):
        _parse_with(TxtConnector(), b"\xff")
