"""models/plumbline: valid, and the generated tables keep the shape recorded in the plan.
That it equals Alembic head is the drift gate's job (tools/drift.py), not a test here."""
import ddl
import validate
from model import load, model_dir

COLUMNS = {
    "documents": ["id", "public_id", "content_hash", "title", "filename", "connector", "created_at"],
    "chunks": ["id", "document_id", "ordinal", "text", "locator", "embedding"],
    "answers": ["id", "public_id", "question", "answer", "abstained", "citations", "chat_model",
                "embedding_model", "abstain_threshold", "created_at"],
    "feedback": ["id", "answer_id", "rating", "reason", "comment", "updated_at"],
}


def test_plumbline_model(monkeypatch):
    monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", "8")
    m = load(model_dir("models/plumbline"))
    assert validate.validate(m) == []
    tables = ddl.build_objects(m)["tables"]
    assert {n: list(t["columns"]) for n, t in tables.items()} == COLUMNS
    assert tables["documents"]["columns"]["id"]["def"] == "id bigserial NOT NULL"
    assert tables["chunks"]["columns"]["embedding"]["def"] == "embedding vector(8) NOT NULL"
