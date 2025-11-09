"""
FAISS-backed vector store loader.

Provides a compact, production-oriented implementation that:
- builds/updates a FAISS index from uploaded text or JSON files
- supports HuggingFace or OpenAI embeddings
- saves and loads the index from disk efficiently

This file implements a class suitable for injecting into FastAPI endpoints.
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable, trace
from fastapi import UploadFile

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Handle HuggingFace embeddings imports
# Try modern location first, then fall back to community
HuggingFaceEmbeddings = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        pass  # Will be None, handle gracefully later

# Handle OpenAI embeddings import
OpenAIEmbeddings = None
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    pass  # Will be None, handle gracefully later
load_dotenv()


class VectorStoreLoader:
    """
    Manage a FAISS vector index: create, update, persist, load, and query.

    Usage:
        loader = VectorStoreLoader(model_name="all-MiniLM-L6-v2", index_path="faiss_index")
        await loader.load_documents(files)
        results = loader.query("search text", k=3)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str | Path = "faiss_index",
    ) -> None:
        self.index_path = Path(index_path)
        self.embeddings = self._init_embeddings(model_name)
        self.vectorstore: FAISS | None = None
        self._load_index()

    def _init_embeddings(self, model_name: str) -> Embeddings:
        """Return an embeddings object for the given model name."""
        if model_name.lower() == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError(
                    "OpenAI embeddings requested but langchain-openai not installed.\n"
                    "Install with: pip install langchain-openai"
                )
            from os import getenv
            api_key = getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI embeddings.")
            return OpenAIEmbeddings(openai_api_key=api_key)

        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "HuggingFace embeddings not available.\n"
                "Install with: pip install langchain-huggingface sentence-transformers"
            )
        return HuggingFaceEmbeddings(model_name=model_name)

    def _load_index(self) -> None:
        """Load a saved FAISS index if it exists on disk."""
        if not self.index_path.exists():
            print(f"ℹ️ No existing FAISS index found at {self.index_path}")
            return
        try:
            # Trace index load to LangSmith
            with trace(name="LoadFAISSIndex", metadata={"component": "vectorstore", "index": str(self.index_path)}):
                self.vectorstore = FAISS.load_local(
                    str(self.index_path), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            print(f"✅ FAISS index loaded from {self.index_path}")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index from {self.index_path}: {e}")
            print("A new index will be created on first document load.")
            self.vectorstore = None

    def _save_index(self) -> None:
        """Persist the FAISS index to disk in the configured directory."""
        if not self.vectorstore:
            print("⚠️ No vectorstore to save")
            return

        self.index_path.mkdir(parents=True, exist_ok=True)
        # Trace index save
        with trace(name="SaveFAISSIndex", metadata={"component": "vectorstore", "index": str(self.index_path)}):
            self.vectorstore.save_local(str(self.index_path))
        print(f"💾 FAISS index saved to {self.index_path}")

    @traceable(name="VectorStore.LoadDocuments")
    async def load_documents(self, files: list[UploadFile]) -> dict[str, Any]:
        """Load uploaded text or JSON files and add them to the FAISS index."""
        docs: list[Document] = []
        
        for file in files:
            try:
                content = (await file.read()).decode("utf-8")
            except Exception as e:
                return {
                    "status": "error", 
                    "message": f"Failed to read {file.filename}: {str(e)}"
                }

            file_path = Path(file.filename or "unknown")

            # Handle JSON files
            if file_path.suffix.lower() == ".json":
                try:
                    payload = json.loads(content)
                    docs.extend(self._parse_json_payload(payload, file_path.name))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON in {file_path.name}, treating as plain text: {e}")
                    docs.append(Document(
                        page_content=content, 
                        metadata={"source": file_path.name, "type": "text"}
                    ))
            else:
                # Plain text files
                docs.append(Document(
                    page_content=content, 
                    metadata={"source": file_path.name, "type": "text"}
                ))

        if not docs:
            return {"status": "no_documents", "count": 0}

        # Add to existing index or create new one
        if self.vectorstore:
            with trace(name="VectorStore.AddDocuments", metadata={"component": "vectorstore", "count": len(docs)}):
                self.vectorstore.add_documents(docs)
            print(f"➕ Added {len(docs)} documents to existing index")
        else:
            with trace(name="VectorStore.CreateIndex", metadata={"component": "vectorstore", "count": len(docs)}):
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"🆕 Created new index with {len(docs)} documents")

        self._save_index()
        return {"status": "ok", "count": len(docs)}

    def _parse_json_payload(self, payload: Any, source: str) -> list[Document]:
        """Parse JSON payload into Document objects."""
        docs: list[Document] = []
        
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or json.dumps(item)
                    metadata = item.get("metadata", {})
                    metadata["source"] = source
                else:
                    text = str(item)
                    metadata = {"source": source}
                docs.append(Document(page_content=str(text), metadata=metadata))
                
        elif isinstance(payload, dict):
            text = payload.get("text") or payload.get("content") or json.dumps(payload)
            metadata = payload.get("metadata", {})
            metadata["source"] = source
            docs.append(Document(page_content=str(text), metadata=metadata))
        else:
            docs.append(Document(
                page_content=str(payload), 
                metadata={"source": source}
            ))
        
        return docs

    @traceable(name="VectorQuery")
    def query(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return top-k most similar documents to the query."""
        if not self.vectorstore:
            return [{"error": "index_not_loaded", "message": "No vectorstore available. Load documents first."}]
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "page_content": doc.page_content, 
                    "metadata": doc.metadata
                } 
                for doc in docs
            ]
        except Exception as e:
            return [{"error": "query_failed", "message": str(e)}]


__all__ = ["VectorStoreLoader"]
