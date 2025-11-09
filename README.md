# InsightRAG — Adaptive, Self-Corrective Retrieval-Augmented Generation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-0.0.330-blue)](https://github.com/hwchase17/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.10-purple.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-yellow.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Tracing](https://img.shields.io/badge/tracing-LangSmith-orange.svg)](https://smith.langchain.com/)

A modular, production-ready **RAG pipeline** built with **FastAPI**, **LangGraph**, and **Streamlit**, featuring adaptive routing, self-grading, and hallucination correction for fact-grounded responses.

---

## 🚀 Overview
**InsightRAG** combines internal vector retrieval and external web search to deliver factual, explainable AI responses.  
It integrates **LangGraph** for adaptive orchestration, **LangSmith** for tracing and metrics, and **FAISS** for fast semantic search.

The system supports three RAG paradigms:
- **Adaptive RAG** → Chooses between internal (FAISS) or external (web) retrieval.  
- **Self RAG** → Grades document relevance and factual alignment.  
- **Corrective RAG** → Regenerates hallucinated answers for better accuracy.

---

## 🧰 Tech Stack

| Layer | Technologies |
|--------|---------------|
| **Backend** | Python, FastAPI, LangGraph, LangChain |
| **Models** | Gemini, GPT-4, Claude |
| **Vector Store** | FAISS, HuggingFace Embeddings (`all-MiniLM-L6-v2`) |
| **Web Search** | Tavily API |
| **Frontend** | Streamlit |
| **Observability** | LangSmith Tracing & Metrics |
| **Documentation** | FastAPI Swagger/OpenAPI |

For a detailed architectural diagram and agent interaction flow, check out the [LangGraph Flow Diagram](app/agents/README.md#langgraph-flow)

## 📡 API Endpoints (FastAPI)

| Endpoint | Method | Description | Request Schema |
|----------|--------|-------------|----------------|
| `/api/v1/load_vectors` | POST | Upload and embed text/JSON documents | `UploadFile` |
| `/api/v1/query_vectors` | POST | Query FAISS for similar docs | `QueryRequest`: `{query: str, k?: int}` |
| `/api/v1/model_query` | POST | Direct LLM query to model | `ModelQueryRequest`: `{query: str, provider?: str, model?: str}` |
| `/api/v1/rag_query` | POST | Run full Adaptive RAG pipeline | `RAGQueryRequest`: `{query: str, k?: int}` |
| `/health` | GET | Health check endpoint | - |
| `/` | GET | API root with version info | - |

## 💻 Frontend (Streamlit)

A minimal chatbot UI to demo all endpoints.
Launch it using:

```bash
streamlit run streamlit_app.py
```

## 🎯 Features

### Core Capabilities
- Adaptive routing between vector store and web search
- Document relevance grading and factual alignment
- Hallucination detection and correction
- Multi-model support (Gemini, GPT-4, Claude)
- Efficient FAISS-based vector similarity search

### Interface
- Interactive Streamlit chatbot interface
- Rich metadata and source attribution
- Support for both direct LLM and RAG queries
- Real-time tracing and monitoring via LangSmith
- Swagger/OpenAPI documentation

### Data Handling
- Text and JSON document ingestion
- HuggingFace embeddings integration
- Persistent FAISS index storage
- Web search capability via Tavily
- Structured response formatting

## 🧪 Quick Start

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/<yourname>/InsightRAG.git

cd InsightRAG

python -m venv venv

source venv/bin/activate   # or venv\Scripts\activate

pip install -r requirements.txt
```

### 2️⃣ Environment Variables (.env)
```bash
GOOGLE_API_KEY="<your_google_api_key>"
OPENAI_API_KEY="<your_openai_api_key>"
TAVILY_API_KEY="<your_tavily_api_key>"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="AdaptiveRAG"
LANGCHAIN_API_KEY="<your_langsmith_api_key>"
```

### 3️⃣ Run Services
```bash
# Start backend
uvicorn app.main:app --reload

# Start frontend
streamlit run streamlit_app.py
```

## 🧠 Example Query
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag_query" \
-H "Content-Type: application/json" \
-d '{"query": "Explain retrieval augmented generation", "k": 3}'
```

## 📈 Metrics & Tracing

### LangSmith Integration
- Comprehensive tracing of all agent nodes (Router, Retriever, Grader)
- Real-time monitoring and debugging via LangSmith dashboard
- Trace metadata includes: `component`, `agent`, `model`, `index`, `k`, `query_len`, `num_docs`, `avg_doc_score`

### Observability
- Environment Setup: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=AdaptiveRAG`
- LangSmith initialization at FastAPI startup
- Privacy-aware tracing: No PII or full document content in traces
- Performance metrics and latency tracking
- Error monitoring and alerting capabilities

## 🧭 Future Enhancements

- PDF and Docx Document Support: Add capability to process PDF documents and extract text for ingestion
- Conversation Memory: Implement chat history and context management
- Adding Data Visualisations Capablities