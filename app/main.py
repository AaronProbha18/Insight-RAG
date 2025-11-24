from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.vectorstore.loader import VectorStoreLoader
from app.api.v1 import endpoints as api_v1_endpoints
from app.core.langsmith_setup import init_langsmith


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🔄 Initializing VectorStoreLoader...")

    # Correct initialization
    app.state.vectorstore = VectorStoreLoader(
        model_name="all-MiniLM-L6-v2",
        index_path="faiss_index",
        index_type="flat",
        chunk_size=900,
        chunk_overlap=150,
        batch_size=100,
    )

    print("✅ VectorStoreLoader initialized.")
    yield

    print("🧹 Shutting down...")


app = FastAPI(lifespan=lifespan)

app.include_router(api_v1_endpoints.router, prefix="/api/v1")

# Observability
init_langsmith()


@app.get("/")
def root():
    return {"message": "🚀 Adaptive RAG API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}
