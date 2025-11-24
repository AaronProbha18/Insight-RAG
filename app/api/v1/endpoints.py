from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    HTTPException,
    status,
    Body,
    Request
)
from typing import List

from app.vectorstore.loader import DocumentIngestor
from app.api.v1.schemas import (
    DocumentResponse,
    LoadVectorsResponse,
    ModelQueryRequest,
    ModelQueryResponse,
    QueryRequest,
    QueryResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RemoveDocumentsResponse,
    ResetIndexResponse,
    RemoveDocumentsRequest
)
from app.core.model_loader import load_gemini_model
from app.core.graph_orchestrator import run_graph

router = APIRouter()

# ======================================================
# 🔥 Correct Dependency Injection
# ======================================================
def get_loader(request: Request) -> DocumentIngestor:
    """
    Extracts the DocumentIngestor initialized during FastAPI lifespan.
    """
    loader = request.app.state.vectorstore
    if loader is None:
        raise RuntimeError("DocumentIngestor not initialized. Check lifespan in main.py.")
    return loader


# ======================================================
# Vector Store Operations
# ======================================================

@router.post(
    "/load_vectors",
    response_model=LoadVectorsResponse,
    tags=["Vector Store Operations"],
    summary="Upload documents and build FAISS index",
)
async def load_vectors(
    files: List[UploadFile] = File(..., description="Upload TXT, JSON, or PDF files"),
    loader: DocumentIngestor = Depends(get_loader),
) -> LoadVectorsResponse:

    try:
        result = await loader.load_documents(files)

        return LoadVectorsResponse(
            status=result.get("status"),
            count=int(result.get("count", 0)),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@router.post(
    "/query_vectors",
    response_model=QueryResponse,
    tags=["Vector Store Operations"],
)
async def query_vectors(
    request: QueryRequest,
    loader: DocumentIngestor = Depends(get_loader),
) -> QueryResponse:

    try:
        # Correct call to IndexStore
        results = loader.index_store.query(request.query, k=request.k)

        if results and "error" in results[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=results[0]["error"]
            )

        return QueryResponse(results=[DocumentResponse(**r) for r in results])

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


# ======================================================
# Query Operations
# ======================================================

@router.post(
    "/model_query",
    response_model=ModelQueryResponse,
    tags=["Query Operations"],
)
async def model_query(request: ModelQueryRequest):

    llm = load_gemini_model()

    try:
        result = llm.invoke(request.prompt) if hasattr(llm, "invoke") else llm(request.prompt)
        return ModelQueryResponse(response=str(result), metadata={})

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@router.post(
    "/rag_query",
    response_model=RAGQueryResponse,
    tags=["Query Operations"],
)
async def rag_query(
    request: RAGQueryRequest,
    loader: DocumentIngestor = Depends(get_loader),
) -> RAGQueryResponse:

    try:
        result = run_graph(
            query=request.query,
            k=request.k,
            loader=loader
        )
        return RAGQueryResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


# ======================================================
# 🔥 Reset & Remove Document Endpoints
# ======================================================

@router.post(
    "/reset_index",
    response_model=ResetIndexResponse,
    tags=["Vector Store Operations"],
    summary="Reset FAISS vector index",
)
async def reset_index(
    loader: DocumentIngestor = Depends(get_loader),
) -> ResetIndexResponse:
    try:
        result = loader.index_store.reset_index()

        # Reset loader internal caches
        loader.index_store.vectorstore = None
        loader.index_store.document_hashes = set()
        loader.document_ingestor.document_hashes = set()

        return ResetIndexResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.post(
    "/remove_documents",
    response_model=RemoveDocumentsResponse,
    tags=["Vector Store Operations"],
)
async def remove_documents(
    request: RemoveDocumentsRequest,
    loader: DocumentIngestor = Depends(get_loader),
) -> RemoveDocumentsResponse:

    try:
        result = loader.index_store.remove_by_metadata(request.filter_metadata)
        return RemoveDocumentsResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )
