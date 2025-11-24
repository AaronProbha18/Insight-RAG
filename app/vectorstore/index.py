import json
import logging
import shutil
from pathlib import Path
from typing import Any

from langsmith import trace
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class IndexStore:
    """Wrapper around FAISS index operations and metadata management."""

    def __init__(
        self,
        index_path: str | Path = "faiss_index",
        embeddings: Embeddings | None = None,
        index_type: str = "flat",
        model_name: str | None = None,
        version: str | None = None,
        use_icons: bool = True,
        log: logging.Logger | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ) -> None:
        self.index_path = Path(index_path)
        self.embeddings = embeddings
        self.index_type = index_type.lower()
        self.model_name = model_name
        self.version = version
        self.use_icons = use_icons
        self.logger = log or logger
        self.vectorstore: FAISS | None = None

        # Store chunking parameters so they can be saved in metadata
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Track known document hashes from saved metadata
        # This is the canonical in-memory set of content hashes for dedup
        self.document_hashes: set[str] = set()

    def _log(self, level: str, message: str, icon: str = "") -> None:
        if self.use_icons and icon:
            message = f"{icon} {message}"
        getattr(self.logger, level.lower())(message)

    def _get_metadata_path(self) -> Path:
        return self.index_path / "index.meta.json"

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = self._get_metadata_path()
        if not meta_path.exists():
            return {}

        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self._log("warning", f"Failed to load metadata: {e}", "⚠️")
            return {}

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        meta_path = self._get_metadata_path()

        try:
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            self._log("debug", f"Metadata saved to {meta_path}", "💾")
        except Exception as e:
            self._log("error", f"Failed to save metadata: {e}", "❌")

    def load(self) -> dict[str, Any]:
        """Load a saved FAISS index if it exists on disk.

        Returns the loaded metadata (may be empty dict).
        """
        if not self.index_path.exists():
            self._log("info", f"No existing FAISS index found at {self.index_path}", "ℹ️")
            # Ensure document_hashes is empty when there's no saved metadata
            self.document_hashes = set()
            return {}

        metadata = self._load_metadata()

        # Restore any stored document hashes so callers can sync ingestors
        self.document_hashes = set(metadata.get("document_hashes", []))

        # Restore chunk settings if present in metadata so instance reflects saved config
        if "chunk_size" in metadata:
            try:
                self.chunk_size = int(metadata.get("chunk_size", self.chunk_size))
            except Exception:
                pass
        if "chunk_overlap" in metadata:
            try:
                self.chunk_overlap = int(metadata.get("chunk_overlap", self.chunk_overlap))
            except Exception:
                pass

        if metadata:
            stored_model = metadata.get("model_name")
            stored_version = metadata.get("version")

            if stored_model != self.model_name:
                self._log(
                    "warning",
                    f"Model mismatch: stored={stored_model}, current={self.model_name}. Index may be incompatible.",
                    "⚠️",
                )

            if stored_version != self.version:
                self._log("info", f"Version upgrade: {stored_version} -> {self.version}", "🔄")

        try:
            with trace(name="LoadFAISSIndex", metadata={"component": "vectorstore", "index": str(self.index_path)}):
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            self._log("info", f"FAISS index loaded from {self.index_path}", "✅")
        except Exception as e:
            self._log("error", f"Failed to load FAISS index: {e}", "❌")
            self._log("info", "A new index will be created on first document load.", "🆕")
            self.vectorstore = None
        return metadata

    def save(self, document_hashes: set[str] | None = None, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        """Persist the FAISS index and metadata to disk."""
        if not self.vectorstore:
            self._log("warning", "No vectorstore to save", "⚠️")
            return

        self.index_path.mkdir(parents=True, exist_ok=True)

        # Load existing metadata to preserve values if None is passed
        existing_metadata = self._load_metadata()

        # Determine final chunk_size: parameter -> instance -> existing metadata -> defaults
        final_chunk_size = chunk_size if chunk_size is not None else (self.chunk_size if self.chunk_size is not None else existing_metadata.get("chunk_size", 1024))

        # Determine final chunk_overlap: parameter -> instance -> existing metadata -> defaults
        final_chunk_overlap = chunk_overlap if chunk_overlap is not None else (self.chunk_overlap if self.chunk_overlap is not None else existing_metadata.get("chunk_overlap", 128))

        # Determine final document hashes: parameter -> existing metadata -> empty list
        final_hashes = (
            list(document_hashes)
            if document_hashes is not None
            else existing_metadata.get("document_hashes", [])
        )

        try:
            with trace(name="SaveFAISSIndex", metadata={"component": "vectorstore", "index": str(self.index_path)}):
                self.vectorstore.save_local(str(self.index_path))

            metadata = {
                "version": self.version,
                "model_name": self.model_name,
                "index_type": self.index_type,
                "chunk_size": final_chunk_size,
                "chunk_overlap": final_chunk_overlap,
                "document_count": self.vectorstore.index.ntotal if self.vectorstore else 0,
                "document_hashes": list(final_hashes),
            }
            self._save_metadata(metadata)

            # Keep in-memory document_hashes consistent with persisted values
            self.document_hashes = set(final_hashes)

            self._log("info", f"FAISS index saved to {self.index_path}", "💾")
        except Exception as e:
            self._log("error", f"Failed to save index: {e}", "❌")
            raise

    def add_documents(self, batch: list) -> None:
        if self.vectorstore:
            with trace(name="VectorStore.AddDocuments", metadata={"component": "vectorstore", "count": len(batch)}):
                self.vectorstore.add_documents(batch)
        else:
            with trace(name="VectorStore.CreateIndex", metadata={"component": "vectorstore", "count": len(batch), "type": self.index_type}):
                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
            self._log("info", f"Created new {self.index_type.upper()} index", "🆕")

    def query(self, query: str, k: int = 4, score_threshold: float = 0.0, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.vectorstore:
            return [{
                "error": "index_not_loaded",
                "message": "No vectorstore available. Load documents first.",
            }]

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k * 3 if filter_metadata else k)

            if filter_metadata:
                filtered = []
                for doc, score in docs_with_scores:
                    if all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                        filtered.append((doc, score))
                docs_with_scores = filtered[:k]

            results = []
            import math
            for doc, distance in docs_with_scores:
                # Note: FAISS distance metric depends on index configuration.
                # Original code used exponential decay on L2 distance: similarity = exp(-distance).
                # Keep same behavior for parity; consider replacing with cosine conversion if you switch metrics.
                similarity = math.exp(-distance)
                if similarity >= score_threshold:
                    results.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(similarity),
                    })

            self._log("debug", f"Query returned {len(results)} results", "🔍")
            return results
        except Exception as e:
            self._log("error", f"Query failed: {e}", "❌")
            return [{"error": "query_failed", "message": str(e)}]

    def get_stats(self) -> dict[str, Any]:
        if not self.vectorstore:
            return {"status": "empty", "message": "No index loaded"}

        metadata = self._load_metadata()

        return {
            "status": "active",
            "version": self.version,
            "model_name": self.model_name,
            "index_type": self.index_type,
            "total_documents": self.vectorstore.index.ntotal,
            "embedding_dimension": getattr(self.vectorstore.index, "d", None),
            "unique_files": len(metadata.get("document_hashes", [])) if metadata else len(self.document_hashes),
            "chunk_size": metadata.get("chunk_size", self.chunk_size),
            "chunk_overlap": metadata.get("chunk_overlap", self.chunk_overlap),
            "index_path": str(self.index_path),
            "metadata": metadata,
        }

    def reset_index(self) -> dict[str, Any]:
        try:
            self.vectorstore = None
            self.document_hashes = set()
            if self.index_path.exists():
                shutil.rmtree(self.index_path)
                self._log("info", f"Deleted index at {self.index_path}", "🗑️")
            return {"status": "ok", "message": "Index reset successfully"}
        except Exception as e:
            self._log("error", f"Failed to reset index: {e}", "❌")
            return {"status": "error", "message": str(e)}

    def remove_by_metadata(self, filter_metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove documents matching filter criteria and preserve all metadata.

        Returns remaining_hashes in the response so callers (loader/ingestor) can sync dedup state.
        """
        if not self.vectorstore:
            return {"status": "error", "message": "No index loaded"}

        try:
            # Load existing metadata to preserve values
            existing_metadata = self._load_metadata()

            all_docs = self.vectorstore.docstore._dict
            docs_to_keep = []
            removed_count = 0

            # Track which content_hashes to keep
            kept_doc_hashes: set[str] = set()

            for doc_id, doc in all_docs.items():
                matches = all(doc.metadata.get(key) == value for key, value in filter_metadata.items())
                if not matches:
                    docs_to_keep.append(doc)
                    # Preserve content_hash if present in metadata
                    doc_hash = doc.metadata.get("content_hash") or doc.metadata.get("contentHash") or doc.metadata.get("content-hash")
                    if doc_hash:
                        kept_doc_hashes.add(doc_hash)
                else:
                    removed_count += 1

            if removed_count == 0:
                return {"status": "ok", "message": "No documents matched filter", "removed_count": 0}

            self._log("info", f"Rebuilding index without {removed_count} documents", "🔄")

            # Rebuild index with remaining documents (or remove entirely)
            if docs_to_keep:
                self.vectorstore = FAISS.from_documents(docs_to_keep, self.embeddings)
                # Update in-memory hashes and persist using preserved chunk settings when possible
                self.document_hashes = kept_doc_hashes
                self.save(
                    document_hashes=kept_doc_hashes,
                    chunk_size=existing_metadata.get("chunk_size", self.chunk_size),
                    chunk_overlap=existing_metadata.get("chunk_overlap", self.chunk_overlap),
                )
            else:
                # No docs left: remove index files and write empty metadata preserving settings
                if self.index_path.exists():
                    shutil.rmtree(self.index_path)
                self.vectorstore = None
                self.document_hashes = set()
                # Preserve basic metadata so system knows the configuration even when empty
                metadata = {
                    "version": existing_metadata.get("version", self.version),
                    "model_name": existing_metadata.get("model_name", self.model_name),
                    "index_type": existing_metadata.get("index_type", self.index_type),
                    "chunk_size": existing_metadata.get("chunk_size", self.chunk_size),
                    "chunk_overlap": existing_metadata.get("chunk_overlap", self.chunk_overlap),
                    "document_count": 0,
                    "document_hashes": [],
                }
                self._save_metadata(metadata)

            return {
                "status": "ok",
                "message": f"Removed {removed_count} documents",
                "removed_count": removed_count,
                "remaining_hashes": list(self.document_hashes),
            }
        except Exception as e:
            self._log("error", f"Failed to remove documents: {e}", "❌")
            return {"status": "error", "message": str(e)}


__all__ = ["IndexStore"]
