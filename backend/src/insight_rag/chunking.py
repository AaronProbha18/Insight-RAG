"""Split parsed documents into chunks that never cross a section boundary."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from insight_rag.connectors.contract import ParsedDocument

# Paragraph, line, sentence, word, then a hard character cut.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    locator: str


def chunk(doc: ParsedDocument, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(
            f"need size > 0 and 0 <= overlap < size; got {size=}, {overlap=}"
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        keep_separator="end",
    )
    pieces = [
        (text, section.locator)
        for section in doc.sections
        for text in splitter.split_text(section.text)
    ]
    return [Chunk(i, text, locator) for i, (text, locator) in enumerate(pieces)]
