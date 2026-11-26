"""Connector contract v1. Third-party connectors implement `FileConnector`."""

from typing import ClassVar, Protocol

from pydantic import BaseModel, Field

CONTRACT_VERSION = 1


class Section(BaseModel):
    text: str = Field(min_length=1)
    locator: str = Field(min_length=1)  # e.g. "p. 3" or "Introduction > Scope"


class ParsedDocument(BaseModel):
    title: str
    sections: list[Section]
    metadata: dict[str, str] = {}


class ParseError(Exception):
    """Raised by a connector when it cannot read the input."""


class FileConnector(Protocol):
    name: ClassVar[str]
    contract_version: ClassVar[int]
    media_types: ClassVar[tuple[str, ...]]
    extensions: ClassVar[tuple[str, ...]]

    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...
