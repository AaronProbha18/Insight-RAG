from fastapi import FastAPI
from app.api.v1 import endpoints as api_v1_endpoints
from app.core.langsmith_setup import init_langsmith

app = FastAPI(
    title="🚀 Adaptive RAG API",
    description="""
**Adaptive RAG API**

This API powers the *Adaptive, Corrective, and Self-RAG System*, which integrates:
- Vectorstore-based document retrieval (FAISS)
- Intelligent routing (vector vs web search)
- Document grading & answer generation
- Hallucination detection and correction
- LangSmith-based logging & tracing

Use this interface to test RAG pipeline, vector loading, and LLM queries interactively.
""",
    version="1.0.0",
    contact={
        "name": "Aaron Probha",
        "url": "https://github.com/AaronProbha18",
        "email": "aaronprobha@gmail.com",
    }
)

app.include_router(api_v1_endpoints.router, prefix="/api/v1")

# Initialize LangSmith observability
init_langsmith()

@app.get("/", tags=["Health & Utility"])
def root():
    """API root endpoint — verify service availability."""
    return {"message": "🚀 Adaptive RAG API is running!"}

@app.get("/health", tags=["Health & Utility"])
def health():
    """Health check endpoint — returns OK if the API is live."""
    return {"status": "ok"}
