from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plumbline.api.app import app

FIX = Path(__file__).parent / "fixtures" / "quality"


@pytest.fixture
def get(monkeypatch):
    monkeypatch.setenv("PLUMBLINE_LLM", "fake:chat")
    monkeypatch.setenv("PLUMBLINE_EMBEDDINGS", "fake:embed")
    monkeypatch.setenv("PLUMBLINE_ABSTAIN_THRESHOLD", "0.3")

    def _get(name):
        monkeypatch.setenv("PLUMBLINE_EVAL_SUMMARY", str(FIX / f"{name}.json"))
        r = TestClient(app).get("/quality")
        assert r.status_code == 200
        return r.json(), r.text

    return _get


def test_missing_file_is_not_measured(get):
    body, _ = get("nope")
    assert body == {"state": "not_measured", "eval_command": "uv run plumbline-eval"}


def test_measured_has_summary_fields_only(get):
    body, text = get("both_pass")
    assert body["state"] == "measured" and body["config_matches"] is True
    assert set(body["connectors"]) == {"pdf", "docx"}
    for leak in (
        "host",
        "git_commit",
        "report",
        "/secret",
        "confidence",
        "score",
        "chunk_id",
    ):
        assert leak not in text


def test_docx_fails_faithfulness(get):
    body, _ = get("docx_fails_faithfulness")
    assert body["connectors"]["docx"]["failed"] == ["faithfulness"]


def test_different_chat_model_is_flagged(get, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_LLM", "other:chat")
    body, _ = get("both_pass")
    assert body["config_differences"] == ["chat"]


def test_question_counts_are_in_the_entry_and_nothing_else_new(get):
    body, text = get("with_counts")
    assert set(body["connectors"]["pdf"]) == {
        "result",
        "failed",
        "metrics",
        "answerable_questions",
        "unanswerable_questions",
    }
    for leak in ("latency_ms", "golden_version", '"questions"', "/secret"):
        assert leak not in text


def test_old_summary_has_no_counts(get):
    body, _ = get("both_pass")
    assert "answerable_questions" not in body["connectors"]["pdf"]
