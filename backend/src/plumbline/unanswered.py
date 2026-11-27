"""Questions the documents could not answer: refusals and thumbs-down answers."""

import psycopg

_KEY = r"lower(regexp_replace(btrim(a.question), '\s+', ' ', 'g'))"

_QUERY = f"""
WITH ev AS (
  SELECT a.id, {_KEY} AS key, a.question, a.created_at, a.abstained, f.reason
  FROM answers a LEFT JOIN feedback f ON f.answer_id = a.id
  WHERE (a.abstained OR f.rating = 'down')
    AND (%(days)s::int IS NULL
         OR a.created_at >= now() - make_interval(days => %(days)s::int))
), latest AS (
  SELECT DISTINCT ON (key) key, question, abstained, reason
  FROM ev ORDER BY key, created_at DESC, id DESC
), agg AS (
  SELECT key, count(*) AS n, max(created_at) AS last_at FROM ev GROUP BY key
)
SELECT l.question, agg.n, agg.last_at,
       CASE WHEN l.abstained THEN 'not_in_documents' ELSE 'marked_wrong' END,
       CASE WHEN l.abstained THEN NULL ELSE l.reason END
FROM latest l JOIN agg USING (key)
ORDER BY agg.last_at DESC, l.key
"""


def list_groups(conn: psycopg.Connection, days: int | None) -> list[dict]:
    """Groups, newest last_asked_at first; `days` None means all time."""
    keys = ("question", "count", "last_asked_at", "reason", "feedback_reason")
    rows = conn.execute(_QUERY, {"days": days}).fetchall()
    return [dict(zip(keys, row)) for row in rows]
