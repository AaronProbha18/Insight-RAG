from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Any


def make_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with consistent separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_document(text_splitter: RecursiveCharacterTextSplitter, content: str, source: str, metadata: dict[str, Any]) -> list[Document]:
    """Split `content` into chunks and return a list of Document objects.

    Preserves the same metadata structure as original implementation.
    """
    chunks = text_splitter.split_text(content)

    return [
        Document(
            page_content=chunk,
            metadata={
                **metadata,
                "source": source,
                "source_file": source,
                "chunk_id": i,
                "total_chunks": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]
