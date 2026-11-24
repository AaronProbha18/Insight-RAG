"""Recorded answers, so a rating can point at exactly what was said."""

import argparse
import json
from pathlib import Path
from typing import Literal

import psycopg
import yaml
from pydantic import BaseModel, Field, model_validator

from plumbline.db import database_url
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


DEFAULT_OUT = Path(__file__).parents[3] / "evals" / "feedback" / "candidates.yaml"


def export_candidates(conn: psycopg.Connection) -> list[dict]:
    """One golden-shaped candidate per feedback row; a human fills in the rest."""
    rows = conn.execute(
        "SELECT a.public_id, a.question, a.answer, a.abstained, a.citations,"
        " a.chat_model, a.embedding_model, a.abstain_threshold,"
        " f.rating, f.reason, f.comment"
        " FROM feedback f JOIN answers a ON a.id = f.answer_id ORDER BY f.id"
    ).fetchall()
    out = []
    for (
        pid,
        question,
        answer,
        abstained,
        cites,
        chat,
        emb,
        thr,
        rating,
        reason,
        comment,
    ) in rows:
        supporting = [
            {"document": Path(c["filename"]).stem, "locator": c["locator"]}
            for c in cites
        ]
        if rating == "up":
            answerable = not abstained
        elif reason == "should_have_answered":
            answerable, supporting = True, []
        else:
            answerable = None  # reviewer decides
        out.append(
            {
                "id": f"fb-{pid}",
                "question": question,
                "answerable": answerable,
                "reference_answer": "",
                "key_fact": "",
                "supporting": supporting,
                "feedback": {
                    "rating": rating,
                    "reason": reason,
                    "comment": comment,
                    "answer": answer,
                    "chat_model": chat,
                    "embedding_model": emb,
                    "abstain_threshold": thr,
                },
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plumbline-feedback-export",
        description="Write recorded feedback as golden-set candidates for review.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    with psycopg.connect(database_url(), connect_timeout=5) as conn:
        candidates = export_candidates(conn)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(candidates, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {len(candidates)} candidates to {args.out}")
    return 0
