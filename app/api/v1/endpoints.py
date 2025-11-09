"""
Enhanced API Endpoints for Adaptive RAG System

This module defines all HTTP endpoints for interacting with:
1. Vector Store operations (document ingestion & similarity search)
2. Query operations (direct LLM queries and RAG pipeline)

Each endpoint includes detailed OpenAPI metadata for better documentation visibility.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Body
from typing import List

from app.core.vectorstore_loader import VectorStoreLoader
from app.api.v1.schemas import (
    DocumentResponse,
    LoadVectorsResponse,
    ModelQueryRequest,
    ModelQueryResponse,
    QueryRequest,
    QueryResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.core.model_loader import load_gemini_model
from app.core.graph_orchestrator import run_graph

router = APIRouter()

# ======================================================
# Singleton Loader Instance
# ======================================================
_loader_singleton: VectorStoreLoader | None = None

def get_loader() -> VectorStoreLoader:
    """
    Returns a singleton instance of VectorStoreLoader to avoid reloading FAISS index per request.
    """
    global _loader_singleton
    if _loader_singleton is None:
        _loader_singleton = VectorStoreLoader()
    return _loader_singleton


# ======================================================
# Vector Store Operations
# ======================================================

@router.post(
    "/load_vectors",
    response_model=LoadVectorsResponse,
    tags=["Vector Store Operations"],
    summary="Upload documents and build FAISS index",
    description=(
        "Uploads one or more text or JSON files, embeds them using the SentenceTransformer model, "
        "and constructs a FAISS vector index stored on disk for fast semantic retrieval."
    ),
    response_description="Returns confirmation and the number of successfully indexed documents."
)
async def load_vectors(
    files: List[UploadFile] = File(..., description="List of `.txt` or `.json` files to index."),
    loader: VectorStoreLoader = Depends(get_loader),
) -> LoadVectorsResponse:
    try:
        result = await loader.load_documents(files)
        return LoadVectorsResponse(
            status=result.get("status", "ok"),
            count=int(result.get("count", 0))
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/query_vectors",
    response_model=QueryResponse,
    tags=["Vector Store Operations"],
    summary="Query FAISS index for top-k similar documents",
    description=(
        "Performs a semantic similarity search on the FAISS index to retrieve the most relevant documents "
        "for a given query. Useful for testing embedding quality or verifying stored data."
    ),
    response_description="List of retrieved documents with metadata and similarity scores."
)
async def query_vectors(
    request: QueryRequest = Body(
        ...,
        examples={
            "default": {
                "query": "Explain how FAISS indexing works.",
                "k": 3
            }
        }
    ),
    loader: VectorStoreLoader = Depends(get_loader),
) -> QueryResponse:
    try:
        results = loader.query(request.query, k=request.k)
        if results and "error" in results[0]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=results[0]["error"])
        return QueryResponse(results=[DocumentResponse(**r) for r in results])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ======================================================
# Query Operations
# ======================================================

@router.post(
    "/model_query",
    response_model=ModelQueryResponse,
    tags=["Query Operations"],
    summary="Direct LLM interaction",
    description=(
        "Send a free-form prompt directly to the configured Gemini model. "
        "This endpoint is intended for baseline LLM testing without retrieval augmentation."
    ),
    response_description="Raw LLM output text and metadata for debugging."
)
async def model_query(
    request: ModelQueryRequest = Body(
        ...,
        examples={
            "default": {
                "prompt": "What is retrieval-augmented generation?"
            }
        }
    )
) -> ModelQueryResponse:
    llm = load_gemini_model()
    try:
        result = llm.invoke(request.prompt) if hasattr(llm, "invoke") else llm(request.prompt)
        return ModelQueryResponse(response=str(result), metadata={})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/rag_query",
    response_model=RAGQueryResponse,
    tags=["Query Operations"],
    summary="Run full Adaptive RAG pipeline",
    description=(
        "Executes the complete **Adaptive RAG pipeline** combining document retrieval, grading, and grounded "
        "generation. The steps include:\n\n"
        "1. Router determines whether to query internal vectorstore or external web sources.\n"
        "2. Retriever fetches top-k relevant documents.\n"
        "3. Document Grader scores document relevance.\n"
        "4. Answer Generator synthesizes a grounded response.\n"
        "5. Hallucination Grader verifies factual grounding.\n\n"
        "Use this endpoint for full retrieval-augmented question answering."
    ),
    response_description="RAG pipeline output containing final grounded answer, reasoning, and diagnostics."
)
async def rag_query(
    request: RAGQueryRequest = Body(
        ...,
        examples={
            "default": {
                "query": "Explain retrieval-augmented generation.",
                "k": 3
            }
        }
    ),
    loader: VectorStoreLoader = Depends(get_loader),
) -> RAGQueryResponse:
    try:
        result = run_graph(query=request.query, k=request.k, loader=loader)
        return RAGQueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
