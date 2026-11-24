"""Recorded answers, so a rating can point at exactly what was said."""

import json

import psycopg

from plumbline.ingestion import new_public_id


def record_answer(
    conn: psycopg.Connection,
    question: str,
    result: dict,
    *,
    chat_model: str,
    embedding_model: str,
    threshold: float,
) -> str:
    """Insert one answer (or abstention) and return its public `ans_` id."""
    answer_id = new_public_id("ans")
    # Copied, not joined: a cited document may be deleted later (#12).
    citations = [
        {k: c[k] for k in ("document_id", "title", "filename", "locator")}
        for c in result["citations"]
    ]
    conn.execute(
        "INSERT INTO answers (public_id, question, answer, abstained, citations,"
        " chat_model, embedding_model, abstain_threshold)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            answer_id,
            question,
            result["answer"],
            result["abstained"],
            json.dumps(citations),
            chat_model,
            embedding_model,
            threshold,
        ),
    )
    return answer_id
