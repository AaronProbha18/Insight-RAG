# 🧠 Adaptive–Corrective–Self RAG Agents

A modular agentic architecture for reliable Retrieval-Augmented Generation (RAG) built with LangChain, LangGraph, FastAPI, and Gemini models. The system is explainable and self-correcting: it adapts routing, corrects low-quality sources, and grades hallucinations.

---

## Overview

This project implements three complementary behaviors:

- **Adaptive RAG** — dynamically chooses between internal (vectorstore) and external (web) retrieval.  
- **Corrective RAG** — detects irrelevant or low-confidence context and reroutes for better sources.  
- **Self-RAG** — grades and corrects hallucinations before finalizing an answer.

All prompts are centralized in `app/prompts/registry.py` for transparency and consistent tuning. Agents use `load_model()` for consistent LLM configuration and LCEL syntax (PromptTemplate | llm) for LangGraph composability.

---

## Agent Directory

```
app/
└── agents/
    ├── router_agent.py
    ├── retriever_agent.py
    ├── document_grader.py
    ├── answer_generator.py
    ├── hallucination_grader.py
    └── web_search_agent.py
```

Each file implements one autonomous agent in the LangGraph state machine.

---

## Agents

### 1. Router Agent (`router_agent.py`)
Goal: Decide whether to use internal vectorstore retrieval or web search.  
Prompt: `ROUTER_PROMPT`  
LLM: Gemini (temperature=0, deterministic)  
Fallback: Keyword-based heuristic

Output example:
```json
{
  "route": "vector" | "web",
  "reason": "Brief rationale for routing decision"
}
```

Key traits:
- Deterministic (temp=0)
- Handles JSON or plain-text responses
- Always returns a valid route

---

### 2. Retriever Agent (`retriever_agent.py`)
Goal: Retrieve top-matching context documents from the vectorstore (FAISS/Chroma).  
Input: Query string  
Output: List of relevant document snippets

Notes:
- Uses GoogleGenerativeAIEmbeddings or fallback embeddings
- Supports similarity search and scoring
- Integrates with Document Grader for relevance validation

---

### 3. Document Grader (`document_grader.py`)
Goal: Evaluate retrieved documents’ relevance to the user query.  
Prompt: `DOC_GRADER_PROMPT`  
LLM: Gemini (temperature=0, deterministic)  
Fallback: Token-overlap heuristic

Output example:
```json
{
  "relevant": true | false,
  "score": 0.85,
  "reason": "Token overlap or semantic match explanation"
}
```

Features:
- JSON parsing with regex extraction
- Heuristic fallback for reliability
- Explainable, deterministic grading

---

### 4. Answer Generator (`answer_generator.py`)
Goal: Produce a concise, cited answer using relevant context documents.  
Prompt: `ANSWER_PROMPT`  
LLM: Gemini (temperature=0.3, mildly creative)

Logic & output:
- Builds a numbered context block ([Doc 1], [Doc 2], …)
- Instructs the model to answer only from the given context
- If uncertain → responds "I don’t know."

Output example:
```json
{
  "answer": "A short, grounded explanation with citations [Doc 2]",
  "used_docs": [2, 4]
}
```

---

### 5. Hallucination Grader (`hallucination_grader.py`)
Goal: Detect and fix hallucinations or unsupported claims in the generated answer.  
Prompts: `HALLUCINATION_PROMPT`, `REGENERATE_PROMPT`  
LLM: Gemini (temperature=0, deterministic)  
Fallback: Heuristic overlap check

Output example:
```json
{
  "grounded": true | false,
  "unsupported_claims": ["Claim A", "Claim B"]
}
```

Highlights:
- JSON-based parsing with regex fallback
- Self-contained error correction
- Can automatically regenerate grounded answers

---

### 6. Web Search Agent (`web_search_agent.py`)
Goal: Retrieve and summarize live information for time-sensitive queries.  
Prompt: `WEBSEARCH_SUMMARY_PROMPT`  
Provider: Tavily API (fallback: empty list)  
LLM: Gemini (temperature=0.2)

Functions:
- `search_web()` — executes provider search
- `summarize_web_results()` — summarizes snippets into bullet points
- `aggregate_and_summarize()` — combined helper

Sample output bullets:
- Major update in AI frameworks in 2025  
- Gemini 2.5 introduces multi-modal agent orchestration  
- Open-source RAG benchmarks evolving quickly

Failsafe: Returns empty string if Tavily or Gemini unavailable.

---

## Integration Notes

- All agents use the same model-loading utility for consistent config.  
- Prompts are centralized in `app/prompts/registry.py`.  
- LCEL syntax (PromptTemplate | llm) is used over LLMChain for LangGraph composability.  
- Robust fallback logic ensures the pipeline degrades gracefully to heuristics rather than failing.

---

## Observability & Tracing (LangSmith)

This project includes LangSmith tracing integration across core components so you can observe LLM calls, vectorstore operations, and the LangGraph execution path in the LangSmith dashboard.

Key implementation details:

- Tracing API: `langsmith.traceable` (decorator) is used for function-level spans and `langsmith.trace` is used as a context manager for dynamic metadata.
- Top-level traceable spans implemented include:
  - `LoadGeminiModel` / `ModelInference` (model load and per-call inference)
  - `VectorStore.LoadDocuments`, `VectorQuery`, `LoadFAISSIndex`, `SaveFAISSIndex`
  - `RouterAgent.RouteQuery`, `RetrieverAgent.Retrieve`, `DocumentGrader.Grade`, `AnswerGenerator.Generate`, `HallucinationGrader.Grade`, `WebSearchAgent.Search` / `Summarize`
  - `GraphOrchestrator.BuildGraph`, `GraphOrchestrator.RunGraph` and per-node spans `Graph.Node.*`

Metadata and conventions:

- Standard metadata keys: `component`, `agent`, `model`, `index`, `k`, `query_len`, `num_docs`, `avg_doc_score`, `grounded`, `request_id`.
- Avoid storing full document bodies or PII in trace metadata. Use document ids, hashes, or short truncated snippets instead.

Where to find traces:

- Set these environment variables before running the app:
  - `LANGCHAIN_TRACING_V2=true`
  - `LANGCHAIN_PROJECT=AdaptiveRAG`
  - `LANGCHAIN_API_KEY=<your_langsmith_api_key>`

- The tracing initializer is `app/core/langsmith_setup.py` and runs at FastAPI startup.

---

## New `/rag_query` endpoint

A convenience endpoint was added to execute the LangGraph orchestrator end-to-end. The route is implemented in `app/api/v1/endpoints.py` as `/api/v1/rag_query`.

Models:

- `RAGQueryRequest` — `{ query: str, k?: int }`
- `RAGQueryResponse` — `{ query, route, route_reason, answer, grounded, unsupported_claims, used_docs, metadata }`

Example (PowerShell):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/rag_query" \
  -Method POST \
  -ContentType "application/json" \
  -Body '{"query": "Explain retrieval augmented generation briefly.", "k": 3}'
```

Expected minimal response structure:

```json
{
  "query": "...",
  "route": "vector",
  "route_reason": "...",
  "answer": "...",
  "grounded": true,
  "unsupported_claims": [],
  "used_docs": [1,2],
  "metadata": {"avg_doc_score": 0.9}
}
```

---

## Local quickstart

1. Create a virtualenv and install dependencies (adjust for your environment):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Set environment variables (PowerShell):

```powershell
$env:GOOGLE_API_KEY="<your_google_api_key>"
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_PROJECT="AdaptiveRAG"
$env:LANGCHAIN_API_KEY="<your_langsmith_api_key>"
```

3. Run the app:

```powershell
uvicorn app.main:app --reload
```

4. Try `/api/v1/rag_query` or the existing endpoints (`/api/v1/query_vectors`, `/api/v1/model_query`).

---

## Testing & CI suggestions

- Unit tests: monkeypatch `langsmith.traceable` and `langsmith.trace` to no-op functions for fast unit tests that don't send traces.
- Integration smoke: run `run_graph` with a mocked `VectorStoreLoader` and assert the graph path and final `grounded` value.

---

If you'd like, I can add the mocked unit tests now (pytest) and a smoke test, or create an `observability.md` describing trace names and recommended dashboards.

---

## LangGraph Flow

```
                         ┌────────────────────────────┐
                         │         START              │
                         └────────────┬───────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │        ROUTER AGENT        │
                         │ - Classifies query type    │
                         │ - Output: {route, reason}  │
                         └────────────┬───────────────┘
                        ┌─────────────┴───────────────┐
                        │                             │
                        ▼                             ▼
          ┌────────────────────────────┐    ┌────────────────────────────┐
          │      RETRIEVER AGENT       │    │     WEB SEARCH AGENT       │
          │ - Queries FAISS index      │    │ - Runs Tavily web search   │
          │ - Retrieves top-k docs     │    │ - Summarizes snippets      │
          └────────────┬───────────────┘    └──────────────────┬─────────┘
                       │                                       │
                       ▼                                       │
           ┌────────────────────────────┐                      │
           │     DOCUMENT GRADER        │                      │
           │ - Grades doc relevance     │                      │
           │ - Output: avg_score, flags │                      │
           └────────────┬───────────────┘                      │
              ┌─────────┴────────────┐                         │
              │                      │                         │
              ▼                      ▼                         │
┌────────────────────────────┐   ┌────────────────────┐        │
│     WEB SEARCH AGENT       │   │ ANSWER GENERATOR   │        │
│ (Fallback if low relevance)│   │ (Direct path)      │        │
└────────────┬───────────────┘   └────────────┬───────┘        │
             │                                │                │
             ▼                                ▼                │
         ┌──────────────────────────────────────────────┐      │
         │           ANSWER GENERATOR AGENT             │      │
         │ - Synthesizes final response                 │      │
         │ - Uses context + citations [Doc X]           │      │
         └────────────┬─────────────────────────────────┘      │
                      │                                        │
                      ▼                                        │
         ┌──────────────────────────────────────────────┐      │ 
         │         HALLUCINATION GRADER AGENT           │      │
         │ - Checks factual grounding                   │──────┘──────┐
         │ - Output: grounded / unsupported_claims      │             │
         └─────────────┬────────────────────────────────┘             │
              ┌────────┴──────────┐                                   │ 
              │                   │                                   │ 
              ▼                   ▼                                   │
┌────────────────────────────┐    ┌────────────────────────────┐      │
│      REGENERATE AGENT      │    │            END             │      │
│ (if hallucination detected)│    │ (if grounded=True)         │      │
└────────────┬───────────────┘    └────────────────────────────┘      │
             │                                                        │
             ▼                                                        │
 ┌────────────────────────────┐                                       │
 │   back to HALLUCINATION    │                                       │
 │   (loop until grounded)    │───────────────────────────────────────┘
 └────────────────────────────┘
```
---

## Tech Stack

- FastAPI — Backend API service  
- LangChain + LangGraph — Agent orchestration  
- Gemini (via langchain-google-genai) — Core LLM  
- Tavily API — Web retrieval  
- FAISS / Chroma — Vectorstore backend  
- dotenv — Environment configuration

---