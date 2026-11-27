import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from plumbline import models
from plumbline.config import check_legacy_env
from plumbline.connectors import FileConnector, ParseError
from plumbline.connectors.registry import load_connectors
from plumbline.db import database_url
from plumbline.feedback import FeedbackRequest, record_answer, save_feedback
from plumbline.graph import DEFAULT_THRESHOLD, build_graph
from plumbline.ingestion import UnsupportedFileType, delete_document, ingest
from plumbline.quality import load_quality
from plumbline.retrieval import retrieve

check_legacy_env()
app = FastAPI(title="Plumbline")
log = logging.getLogger(__name__)


def get_conn() -> Iterator[psycopg.Connection]:
    url = database_url()
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
        yield conn


@cache
def get_connectors() -> dict[str, FileConnector]:
    return load_connectors()


@cache
def _embeddings():
    return models.embeddings()


class _LazyEmbeddings:
    """Builds the model on first embed, so 415/422 work without a model configured."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return _embeddings().embed_query(text)


def get_embeddings() -> _LazyEmbeddings:
    return _LazyEmbeddings()


@cache
def _chat_model():
    return models.chat_model()


class _LazyLLM:
    """Builds the model on first use, so abstentions work without a model configured."""

    def invoke(self, prompt):
        return _chat_model().invoke(prompt)


def get_llm() -> _LazyLLM:
    return _LazyLLM()


Conn = Annotated[psycopg.Connection, Depends(get_conn)]


def get_retriever(conn: Conn, embeddings: Annotated[object, Depends(get_embeddings)]):
    return lambda question, top_k: retrieve(conn, embeddings, question, top_k)


def get_recorder(conn: Conn):
    """Records an answer; model ids and threshold come from the environment."""
    return lambda question, result, threshold: record_answer(
        conn,
        question,
        result,
        chat_model=os.environ.get("PLUMBLINE_LLM", ""),
        embedding_model=os.environ.get("PLUMBLINE_EMBEDDINGS", ""),
        threshold=threshold,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/quality")
def quality() -> dict:
    default = Path(__file__).parents[3].parent / "evals/reports/latest.summary.json"
    path = os.environ.get("PLUMBLINE_EVAL_SUMMARY", default)
    return load_quality(
        Path(path),
        {
            "llm": os.environ.get("PLUMBLINE_LLM", ""),
            "embeddings": os.environ.get("PLUMBLINE_EMBEDDINGS", ""),
            "threshold": os.environ.get(
                "PLUMBLINE_ABSTAIN_THRESHOLD", DEFAULT_THRESHOLD
            ),
        },
    )


@app.get("/connectors")
def connectors(
    connectors: Annotated[dict[str, FileConnector], Depends(get_connectors)],
) -> list[dict]:
    return [
        {
            "name": c.name,
            "extensions": list(c.extensions),
            "contract_version": c.contract_version,
            "builtin": type(c).__module__.startswith("plumbline.connectors.builtin."),
        }
        for c in sorted(connectors.values(), key=lambda c: c.name)
    ]


@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile,
    response: Response,
    conn: Conn,
    connectors: Annotated[dict[str, FileConnector], Depends(get_connectors)],
    embeddings: Annotated[object, Depends(get_embeddings)],
) -> dict:
    try:
        result = ingest(
            conn, file.file.read(), file.filename or "", connectors, embeddings
        )
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc
    except ParseError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not result.created:
        response.status_code = 200
    return asdict(result)


@app.post("/documents/samples")
def index_samples(
    conn: Conn,
    connectors: Annotated[dict[str, FileConnector], Depends(get_connectors)],
    embeddings: Annotated[object, Depends(get_embeddings)],
) -> dict:
    default = Path(__file__).parents[3].parent / "evals/connectors/pdf/corpus"
    directory = Path(os.environ.get("PLUMBLINE_SAMPLES_DIR", default))
    if not directory.is_dir():
        raise HTTPException(404, f"samples directory not found: {directory}")
    indexed, failed = [], []
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        try:
            result = ingest(conn, path.read_bytes(), path.name, connectors, embeddings)
        except (UnsupportedFileType, ParseError) as exc:
            failed.append({"filename": path.name, "reason": str(exc)})
        else:
            indexed.append({"filename": path.name, **asdict(result)})
    return {"indexed": indexed, "failed": failed}


@app.get("/documents")
def list_documents(conn: Conn) -> list[dict]:
    rows = conn.execute(
        "SELECT public_id, title, connector, created_at FROM documents"
        " ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return [
        {"document_id": r[0], "title": r[1], "connector": r[2], "created_at": r[3]}
        for r in rows
    ]


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: str, conn: Conn) -> Response:
    if not delete_document(conn, document_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def last_is_user_question(cls, messages: list[Message]) -> list[Message]:
        if (
            not messages
            or messages[-1].role != "user"
            or not messages[-1].content.strip()
        ):
            raise ValueError("last message must be a non-empty user message")
        return messages


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat")
def chat(
    request: ChatRequest,
    retriever: Annotated[object, Depends(get_retriever)],
    llm: Annotated[object, Depends(get_llm)],
    recorder: Annotated[object, Depends(get_recorder)],
) -> StreamingResponse:
    threshold = float(os.environ.get("PLUMBLINE_ABSTAIN_THRESHOLD", DEFAULT_THRESHOLD))
    # Option A (design D8): run to completion first, so only verified answers stream.
    try:
        question = request.messages[-1].content
        result = build_graph(retriever, llm, threshold).invoke({"question": question})
        # An answer that can't be recorded can't be rated, so it is not streamed.
        answer_id = recorder(question, result, threshold)
    except Exception:
        log.exception("answer generation failed")
        events = [_sse("error", {"message": "answer generation failed"})]
    else:
        if result["abstained"]:
            events = [_sse("abstain", {"message": result["answer"]})]
        else:
            events = [
                _sse("delta", {"text": t})
                for t in re.findall(r"\S+\s*", result["answer"])
            ]
            public = [
                {k: v for k, v in c.items() if k != "filename"}
                for c in result["citations"]
            ]
            events.append(_sse("citations", {"citations": public}))
        events.append(_sse("done", {"answer_id": answer_id}))
    return StreamingResponse(iter(events), media_type="text/event-stream")


@app.post("/feedback", status_code=204)
def post_feedback(request: FeedbackRequest, conn: Conn) -> Response:
    if not save_feedback(conn, request):
        raise HTTPException(404, "Answer not found")
    return Response(status_code=204)


# Production image: serve the built UI on the same origin. Mounted last so API routes win.
_ui_dir = os.environ.get("PLUMBLINE_UI_DIR")
if _ui_dir and os.path.isdir(_ui_dir):
    # Client-side pages live under /app/, so a shared link loads the SPA. Only this
    # prefix falls back to index.html; mistyped API paths still 404.
    @app.get("/app/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        return FileResponse(Path(_ui_dir) / "index.html")

    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
