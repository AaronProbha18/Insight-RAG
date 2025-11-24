import logging
from pathlib import Path
from typing import Any, Literal
from os import getenv

from dotenv import load_dotenv
from langsmith import traceable
from fastapi import UploadFile

from langchain_core.embeddings import Embeddings

from .chunker import make_text_splitter
from .index import IndexStore
from .ingest import DocumentIngestor

from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStoreLoader:
    """
    High-level orchestrator that bundles:
    - Embeddings model
    - IndexStore (FAISS)
    - TextSplitter
    - DocumentIngestor

    This is what FastAPI should initialize during lifespan.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "faiss_index",
        index_type: str = "flat",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        version: str = "1.0.0",
        batch_size: int = 100,
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.index_type = index_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.version = version
        self.batch_size = batch_size

        # --------------------------
        # 1. Embeddings
        # --------------------------
        logger.info("🔧 Loading embeddings model: %s", model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # --------------------------
        # 2. Text Splitter
        # --------------------------
        self.text_splitter = make_text_splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # --------------------------
        # 3. FAISS IndexStore
        # --------------------------
        self.index_store = IndexStore(
            index_path=self.index_path,
            embeddings=self.embeddings,
            index_type=self.index_type,
            model_name=self.model_name,
            version=self.version,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Load FAISS index if exists
        logger.info("📦 Loading FAISS index from disk…")
        self.index_store.load()

        # --------------------------
        # 4. DocumentIngestor
        # --------------------------
        self.ingestor = DocumentIngestor(
            text_splitter=self.text_splitter,
            index_store=self.index_store,
            batch_size=self.batch_size,
        )

        logger.info("✅ VectorStoreLoader initialized successfully.")

    # ------------------------------------------------------------------
    # Public helper wrappers used by endpoints
    # ------------------------------------------------------------------

    async def load_documents(self, files):
        """Proxy to ingestor.load_documents"""
        return await self.ingestor.load_documents(files)

    def query(self, query: str, k: int = 4):
        """Proxy to IndexStore query"""
        return self.index_store.query(query, k=k)

    def reset_index(self):
        """Reset FAISS index + clear ingestor dedup state."""
        return self.index_store.reset_index()
