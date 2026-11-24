"""Recorded answers, so a rating can point at exactly what was said."""

import json
from typing import Literal

import psycopg
from pydantic import BaseModel, Field, model_validator

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


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: Literal["up", "down"]
    reason: (
        Literal["wrong", "incomplete", "wrong_source", "should_have_answered", "other"]
        | None
    ) = None
    comment: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def reason_only_with_down(self):
        if self.reason is not None and self.rating != "down":
            raise ValueError("reason is only allowed with a thumbs down")
        return self


def save_feedback(conn: psycopg.Connection, req: FeedbackRequest) -> bool:
    """Store or replace the one feedback row for an answer; False if no such answer."""
    cur = conn.execute(
        "INSERT INTO feedback (answer_id, rating, reason, comment)"
        " SELECT id, %s, %s, %s FROM answers WHERE public_id = %s"
        " ON CONFLICT (answer_id) DO UPDATE SET rating = EXCLUDED.rating,"
        " reason = EXCLUDED.reason, comment = EXCLUDED.comment, updated_at = now()",
        (req.rating, req.reason, req.comment, req.answer_id),
    )
    return cur.rowcount == 1
