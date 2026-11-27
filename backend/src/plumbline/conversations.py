"""Saved conversations: a conversation is a title plus the `answers` rows pointing at it."""

import psycopg

from plumbline.ingestion import new_public_id

TITLE_CUT = 80
TITLE_MAX = 200


class InvalidTitle(ValueError):
    """A title that is empty or longer than 200 characters after trimming."""


def create(conn: psycopg.Connection, first_question: str) -> tuple[int, str]:
    """Insert a conversation titled from its first question; return (id, public_id)."""
    title = " ".join(first_question.split())
    if len(title) > TITLE_CUT:
        title = title[:TITLE_CUT] + "…"
    public_id = new_public_id("conv")
    (conv_id,) = conn.execute(
        "INSERT INTO conversations (public_id, title) VALUES (%s, %s) RETURNING id",
        (public_id, title),
    ).fetchone()
    return conv_id, public_id


def resolve(conn: psycopg.Connection, public_id: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM conversations WHERE public_id = %s", (public_id,)
    ).fetchone()
    return row[0] if row else None


def touch(conn: psycopg.Connection, conv_id: int) -> None:
    conn.execute(
        "UPDATE conversations SET updated_at = now() WHERE id = %s", (conv_id,)
    )


def list_all(conn: psycopg.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT public_id, title, updated_at FROM conversations"
        " ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    return [
        {"conversation_id": p, "title": t, "updated_at": u.isoformat()}
        for p, t, u in rows
    ]


def get(conn: psycopg.Connection, public_id: str) -> dict | None:
    conv = conn.execute(
        "SELECT id, title FROM conversations WHERE public_id = %s", (public_id,)
    ).fetchone()
    if conv is None:
        return None
    conv_id, title = conv
    answers = conn.execute(
        "SELECT public_id, question, answer, abstained, citations FROM answers"
        " WHERE conversation_id = %s ORDER BY created_at, id",
        (conv_id,),
    ).fetchall()
    cited = {c["document_id"] for *_, citations in answers for c in citations}
    indexed = {
        d
        for (d,) in conn.execute(
            "SELECT public_id FROM documents WHERE public_id = ANY(%s)", (list(cited),)
        )
    }
    turns = [
        {
            "answer_id": a,
            "question": q,
            "answer": text,
            "abstained": abstained,
            "citations": [
                {
                    "document_id": c["document_id"],
                    "title": c["title"],
                    "locator": c["locator"],
                    "indexed": c["document_id"] in indexed,
                }
                for c in citations
            ],
        }
        for a, q, text, abstained, citations in answers
    ]
    return {"conversation_id": public_id, "title": title, "turns": turns}


def rename(conn: psycopg.Connection, public_id: str, title: str) -> dict | None:
    """Set a new title; None when unknown. Raises InvalidTitle outside 1-200 characters."""
    title = title.strip()
    if not 1 <= len(title) <= TITLE_MAX:
        raise InvalidTitle(f"title must be 1-{TITLE_MAX} characters")
    row = conn.execute(
        "UPDATE conversations SET title = %s, updated_at = now() WHERE public_id = %s"
        " RETURNING public_id, title, updated_at",
        (title, public_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "conversation_id": row[0],
        "title": row[1],
        "updated_at": row[2].isoformat(),
    }


def delete(conn: psycopg.Connection, public_id: str) -> bool:
    """Delete the conversation; its answers and their feedback stay (SET NULL)."""
    cur = conn.execute("DELETE FROM conversations WHERE public_id = %s", (public_id,))
    return cur.rowcount == 1
