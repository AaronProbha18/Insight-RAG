from itertools import count
from typing import ClassVar

from insight_rag.connectors import ParsedDocument, ParseError, Section


class GoodConnector:
    name: ClassVar[str] = "good"
    contract_version: ClassVar[int] = 1
    media_types: ClassVar[tuple[str, ...]] = ("text/plain",)
    extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(str(exc)) from exc
        return ParsedDocument(
            title=filename, sections=[Section(text=text, locator="all")]
        )


class FutureConnector(GoodConnector):
    name: ClassVar[str] = "future"
    contract_version: ClassVar[int] = 2


class BrokenConnector(GoodConnector):
    def __init__(self):
        raise RuntimeError("boom")


class EmptyConnector(GoodConnector):
    def parse(self, data, filename):
        return ParsedDocument(title=filename, sections=[])


class NondeterministicConnector(GoodConnector):
    _calls = count()

    def parse(self, data, filename):
        return ParsedDocument(
            title=f"{filename} #{next(self._calls)}",
            sections=[Section(text="x", locator="all")],
        )


class CrashingConnector(GoodConnector):
    def parse(self, data, filename):
        if data.startswith(b"\xff"):
            raise ValueError("unexpected byte")
        return super().parse(data, filename)
