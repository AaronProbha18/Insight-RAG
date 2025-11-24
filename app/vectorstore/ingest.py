import io
import json
import hashlib
import logging
from pathlib import Path
from typing import Any

import pypdf
from fastapi import UploadFile
from langsmith import traceable, trace
from langchain_core.documents import Document
from .chunker import chunk_document

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handle document ingestion, deduplication, parsing, chunking and batching.

    This class preserves the original public behavior and return shapes.
    """

    def __init__(
        self,
        text_splitter,
        index_store,
        use_icons: bool = True,
        log: logging.Logger | None = None,
        batch_size: int = 100,
    ) -> None:
        self.text_splitter = text_splitter
        self.index_store = index_store
        self.use_icons = use_icons
        self.logger = log or logger
        self.batch_size = batch_size
        self.document_hashes: set[str] = set()

    def _log(self, level: str, message: str, icon: str = "") -> None:
        if self.use_icons and icon:
            message = f"{icon} {message}"
        getattr(self.logger, level.lower())(message)

    def _compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _parse_pdf(self, file: UploadFile, source: str) -> list[Document]:
        """Parse PDF file and return chunked documents."""
        try:
            # Read file content
            content_bytes = await file.read()
            pdf_file = io.BytesIO(content_bytes)
            
            # Extract text from all pages
            reader = pypdf.PdfReader(pdf_file)
            page_count = len(reader.pages)
            
            # Combine all page text
            all_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            
            combined_text = "\n\n".join(all_text)
            
            # Compute content hash
            content_hash = self._compute_content_hash(combined_text)
            
            # Build metadata
            metadata = {
                "type": "pdf",
                "source": source,
                "content_hash": content_hash,
                "total_pages": page_count
            }
            
            # Chunk the document
            chunks = chunk_document(self.text_splitter, combined_text, source, metadata)
            
            return chunks
            
        except Exception as e:
            self._log("error", f"Failed to parse PDF {source}: {str(e)}", "❌")
            raise

    def _parse_json_payload(self, payload: Any, source: str, content_hash: str) -> list[Document]:
        docs: list[Document] = []

        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                if isinstance(item, dict):
                    if "text" in item:
                        text = item["text"]
                    elif "content" in item:
                        text = item["content"]
                    else:
                        raise ValueError(f"JSON object at index {idx} missing 'text' or 'content' field")

                    metadata = item.get("metadata", {})
                    metadata["source"] = source
                    metadata["source_file"] = source
                    metadata["content_hash"] = content_hash

                    chunks = chunk_document(self.text_splitter, text, source, metadata)
                    docs.extend(chunks)
                else:
                    text = str(item)
                    metadata = {"source": source, "source_file": source, "content_hash": content_hash}
                    chunks = chunk_document(self.text_splitter, text, source, metadata)
                    docs.extend(chunks)

        elif isinstance(payload, dict):
            if "text" in payload:
                text = payload["text"]
            elif "content" in payload:
                text = payload["content"]
            else:
                raise ValueError("JSON object missing 'text' or 'content' field")

            metadata = payload.get("metadata", {})
            metadata["source"] = source
            metadata["source_file"] = source
            metadata["content_hash"] = content_hash

            chunks = chunk_document(self.text_splitter, text, source, metadata)
            docs.extend(chunks)
        else:
            text = str(payload)
            metadata = {"source": source, "source_file": source, "content_hash": content_hash}
            chunks = chunk_document(self.text_splitter, text, source, metadata)
            docs.extend(chunks)

        return docs

    async def _add_documents_batched(self, docs: list[Document]) -> None:
        total_batches = (len(docs) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            self._log("debug", f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)", "⚙️")

            # Delegate to IndexStore
            self.index_store.add_documents(batch)

        self._log("info", f"Added {len(docs)} documents across {total_batches} batch(es)", "➕")

    @traceable(name="VectorStore.LoadDocuments")
    async def load_documents(self, files: list[UploadFile]) -> dict[str, Any]:
        all_docs: list[Document] = []
        skipped_duplicates = 0
        errors = []

        for file in files:
            file_path = Path(file.filename or "unknown")
            extension = file_path.suffix.lower()
            
            # Handle PDF files
            if extension == ".pdf":
                try:
                    docs = await self._parse_pdf(file, file_path.name)
                    
                    # Check for duplicates using the content hash from the first chunk
                    if docs:
                        content_hash = docs[0].metadata.get("content_hash")
                        if content_hash and content_hash in self.document_hashes:
                            self._log("info", f"Skipping duplicate: {file.filename}", "⏭️")
                            skipped_duplicates += 1
                            continue
                        
                        all_docs.extend(docs)
                        if content_hash:
                            self.document_hashes.add(content_hash)
                            
                except Exception as e:
                    error_msg = f"Failed to process PDF {file.filename}: {str(e)}"
                    self._log("error", error_msg, "❌")
                    errors.append(error_msg)
                continue
            
            # Handle other file types (existing logic)
            try:
                content = (await file.read()).decode("utf-8")
            except Exception as e:
                error_msg = f"Failed to read {file.filename}: {str(e)}"
                self._log("error", error_msg, "❌")
                errors.append(error_msg)
                continue

            content_hash = self._compute_content_hash(content)
            if content_hash in self.document_hashes:
                self._log("info", f"Skipping duplicate: {file.filename}", "⏭️")
                skipped_duplicates += 1
                continue

            # Include content_hash for plain-text files so dedup metadata is consistent
            file_metadata = {"type": "text", "content_hash": content_hash}

            if extension == ".json":
                try:
                    payload = json.loads(content)
                    docs = self._parse_json_payload(payload, file_path.name, content_hash)
                    all_docs.extend(docs)
                except json.JSONDecodeError as e:
                    self._log("warning", f"Invalid JSON in {file_path.name}, treating as plain text: {e}", "⚠️")
                    docs = chunk_document(self.text_splitter, content, file_path.name, file_metadata)
                    all_docs.extend(docs)
            else:
                docs = chunk_document(self.text_splitter, content, file_path.name, file_metadata)
                all_docs.extend(docs)

            self.document_hashes.add(content_hash)

        if not all_docs:
            return {"status": "no_documents", "count": 0, "skipped": skipped_duplicates, "errors": errors}

        try:
            await self._add_documents_batched(all_docs)
        except Exception as e:
            self._log("error", f"Failed to add documents: {e}", "❌")
            return {"status": "error", "message": str(e), "errors": errors}

        # After adding docs, persist index metadata including document hashes
        try:
            self.index_store.save(
                document_hashes=self.document_hashes,
                chunk_size=self.index_store.chunk_size,
                chunk_overlap=self.index_store.chunk_overlap,
            )
        except Exception as e:
            self._log("error", f"Failed to save index after ingest: {e}", "❌")

        return {"status": "ok", "count": len(all_docs), "skipped": skipped_duplicates, "errors": errors if errors else None}


__all__ = ["DocumentIngestor"]