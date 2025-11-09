from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ============================================================
# Vector Store Schemas
# ============================================================

class QueryRequest(BaseModel):
    """Request body for querying the FAISS vector index."""
    query: str = Field(..., description="User query string for semantic similarity search.")
    k: int = Field(4, description="Number of top similar documents to return.", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain retrieval augmented generation.",
                "k": 3
            }
        }


class DocumentResponse(BaseModel):
    """A single document returned from vector search."""
    page_content: str = Field(..., description="The content of the retrieved document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the document.")

    class Config:
        json_schema_extra = {
            "example": {
                "page_content": "Retrieval-Augmented Generation (RAG) enhances generative AI by combining LLMs with vector search.",
                "metadata": {"source": "llms-overview.txt", "page": 12}
            }
        }


class QueryResponse(BaseModel):
    """Response containing multiple retrieved documents."""
    results: List[DocumentResponse] = Field(..., description="List of retrieved documents with metadata.")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "page_content": "Retrieval-Augmented Generation (RAG) combines external search with LLMs.",
                        "metadata": {"source": "llms-full.txt"}
                    },
                    {
                        "page_content": "RAG improves factual accuracy by grounding model responses in documents.",
                        "metadata": {"source": "whitepaper.pdf"}
                    }
                ]
            }
        }


class LoadVectorsResponse(BaseModel):
    """Response after loading files into the FAISS index."""
    status: str = Field(..., description="Status of the loading operation (e.g., 'ok' or 'error').")
    count: int = Field(..., description="Number of documents processed.")

    class Config:
        json_schema_extra = {
            "example": {"status": "ok", "count": 23}
        }


# ============================================================
# LLM Model Query Schemas
# ============================================================

class ModelQueryRequest(BaseModel):
    """Request body for direct model query (LLM prompt)."""
    prompt: str = Field(..., description="The natural language input or instruction for the model.")
    max_tokens: Optional[int] = Field(256, description="Maximum number of tokens to generate.")
    temperature: Optional[float] = Field(0.2, description="Sampling temperature for model creativity (0.0–1.0).")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain the concept of retrieval-augmented generation.",
                "max_tokens": 150,
                "temperature": 0.3
            }
        }


class ModelQueryResponse(BaseModel):
    """Response containing model output."""
    response: str = Field(..., description="Generated text response from the model.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata about generation (e.g., latency, tokens).")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Retrieval-Augmented Generation (RAG) combines LLMs with information retrieval to improve factual grounding.",
                "metadata": {"model": "gemini-pro", "tokens_used": 87}
            }
        }


# ============================================================
# RAG Pipeline Schemas
# ============================================================

class RAGQueryRequest(BaseModel):
    """Request body for the Adaptive RAG pipeline query."""
    query: str = Field(..., description="Natural language question for RAG pipeline.")
    k: int = Field(3, description="Number of top documents to retrieve from vectorstore.", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does retrieval-augmented generation reduce hallucinations?",
                "k": 3
            }
        }


class RAGQueryResponse(BaseModel):
    """Response from the Adaptive RAG pipeline."""
    query: str = Field(..., description="Original user query.")
    route: str = Field(..., description="Selected path for answering ('vector' or 'web').")
    route_reason: str = Field(..., description="Reason behind routing decision.")
    answer: str = Field(..., description="Final generated answer after retrieval and grading.")
    grounded: bool = Field(..., description="Whether the answer is grounded in retrieved context.")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unsupported claims detected, if any.")
    used_docs: List[int] = Field(default_factory=list, description="Indices of documents used for answer generation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metrics, reasoning, and intermediate state.")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain retrieval augmented generation.",
                "route": "vector",
                "route_reason": "Query appears factual/document-centered.",
                "answer": "Retrieval-Augmented Generation (RAG) enhances LLMs by grounding their answers in retrieved external data.",
                "grounded": True,
                "unsupported_claims": {},
                "used_docs": [1],
                "metadata": {
                    "avg_doc_score": 0.87,
                    "route_reason": "Vector route selected for factual query"
                }
            }
        }
