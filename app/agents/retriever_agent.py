"""
Retriever agent
----------------
Pure retrieval component that abstracts FAISS/Chroma logic and returns
clean text snippets for downstream graders and answer generators.

This module intentionally performs NO LLM inference. It returns a dict
with the key `retrieved_docs` containing the retrieved text snippets in
descending order of relevance.

API:
    retrieve(query: str, k: int = 4, loader: Optional[VectorStoreLoader]=None) -> dict

Behavior:
 - Uses the application's `VectorStoreLoader` if no loader is provided.
 - Calls the vectorstore's similarity search to get top-k documents.
 - If a document is very large it will be split into character-level
   chunks using RecursiveCharacterTextSplitter, but the overall result
   still preserves the original relevance ordering.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.vectorstore_loader import VectorStoreLoader
from langsmith import traceable


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore


@traceable(name="RetrieverAgent.Retrieve")
def retrieve(
    query: str,
    k: int = 4,
    loader: Optional[VectorStoreLoader] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, List[str]]:
    """Retrieve top-K semantically relevant text chunks for a query.

    Args:
        query: The user query.
        k: Number of text snippets to return (top-K overall).
        loader: Optional VectorStoreLoader. If None, a new instance is created.
        chunk_size: If a returned document exceeds this many characters,
            it will be split (only if RecursiveCharacterTextSplitter is available).
        chunk_overlap: Overlap between chunks when splitting.

    Returns:
        A dict with key `retrieved_docs` whose value is a list of text
        snippets (strings) ordered by descending relevance.
    """
    if loader is None:
        loader = VectorStoreLoader()

    raw_results = loader.query(query, k=k)

    # If the loader signals an error, return an empty list (caller may
    # inspect logs or the loader for details).
    if not isinstance(raw_results, list) or (raw_results and "error" in raw_results[0]):
        return {"retrieved_docs": []}

    retrieved: List[str] = []

    for entry in raw_results:
        text = entry.get("page_content", "") if isinstance(entry, dict) else str(entry)
        if not text:
            continue

        # If the document is large, optionally split into chunks but
        # preserve the original ordering. Append chunks until we reach k.
        if (
            RecursiveCharacterTextSplitter is not None
            and isinstance(text, str)
            and len(text) > chunk_size
        ):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            for c in chunks:
                retrieved.append(c)
                if len(retrieved) >= k:
                    break
        else:
            retrieved.append(text)

        if len(retrieved) >= k:
            break

    return {"retrieved_docs": retrieved}


__all__ = ["retrieve"]
