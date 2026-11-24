import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from plumbline.api.app import app, get_llm, get_recorder, get_retriever
from plumbline.retrieval import RetrievedChunk

ANSWER = "Refunds take 14 days [c1]. Footwear gets 30 days [c2]."


def _chunk(i: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        100 + i, f"doc_{i}", f"Doc {i}", f"p. {i}", f"text {i}", score, f"f{i}.pdf"
    )


class RecordingLLM:
    def __init__(self, reply=ANSWER, fail=False):
        self.model, self.fail, self.calls = (
            FakeListChatModel(responses=[reply]),
            fail,
            0,
        )

    def invoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise ConnectionError("secret-host:11434 refused")
        return self.model.invoke(prompt)


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, question, result, threshold):
        self.calls.append((question, result, threshold))
        return "ans_FAKE"


@pytest.fixture
def recorder():
    return FakeRecorder()


@pytest.fixture
def chat(monkeypatch, recorder):
    monkeypatch.delenv("PLUMBLINE_ABSTAIN_THRESHOLD", raising=False)

    def run(chunks, llm=None, question="What is the refund window?", recorder=recorder):
        llm = llm or RecordingLLM()
        app.dependency_overrides[get_retriever] = lambda: lambda q, k: chunks[:k]
        app.dependency_overrides[get_llm] = lambda: llm
        app.dependency_overrides[get_recorder] = lambda: recorder
        r = TestClient(app).post(
            "/chat", json={"messages": [{"role": "user", "content": question}]}
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        for block in r.text.strip().split("\n\n"):
            name, data = block.split("\n")
            events.append(
                (name.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
            )
        return events, llm

    yield run
    app.dependency_overrides.clear()


def test_answer_event_order(chat):
    events, _ = chat([_chunk(1, 0.9), _chunk(2, 0.8)])
    names = [n for n, _ in events]

    assert names[-2:] == ["citations", "done"]
    assert set(names[:-2]) == {"delta"}
    assert "".join(d["text"] for n, d in events if n == "delta") == ANSWER
    assert events[-2][1]["citations"] == [
        {
            "id": "c1",
            "document_id": "doc_1",
            "title": "Doc 1",
            "locator": "p. 1",
            "text": "text 1",
        },
        {
            "id": "c2",
            "document_id": "doc_2",
            "title": "Doc 2",
            "locator": "p. 2",
            "text": "text 2",
        },
    ]


def test_no_confidence_field_anywhere(chat):
    events, _ = chat([_chunk(1, 0.9), _chunk(2, 0.8)])

    body = json.dumps([d for _, d in events])
    for key in ("confidence", "score", "chunk_id"):
        assert f'"{key}"' not in body


def test_abstain_events(chat):
    events, llm = chat([_chunk(1, 0.1)])

    assert [n for n, _ in events] == ["abstain", "done"]
    assert events[0][1]["message"]
    assert llm.calls == 0


def test_empty_index_abstains_without_llm(chat):
    events, llm = chat([])

    assert [n for n, _ in events] == ["abstain", "done"]
    assert llm.calls == 0


def test_threshold_from_env(chat, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_ABSTAIN_THRESHOLD", "0.6")
    assert chat([_chunk(1, 0.5)])[0][0][0] == "abstain"

    monkeypatch.setenv("PLUMBLINE_ABSTAIN_THRESHOLD", "0.4")
    assert chat([_chunk(1, 0.5)], RecordingLLM("Yes [c1]."))[0][0][0] == "delta"


def test_provider_error_sends_error_event(chat):
    events, _ = chat([_chunk(1, 0.9)], RecordingLLM(fail=True))

    assert events == [("error", {"message": "answer generation failed"})]


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "assistant", "content": "hi"}], [{"role": "user", "content": " "}]],
)
def test_invalid_request_422(messages):
    app.dependency_overrides[get_retriever] = lambda: lambda q, k: []
    app.dependency_overrides[get_llm] = RecordingLLM
    try:
        r = TestClient(app).post("/chat", json={"messages": messages})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 422


def test_done_carries_answer_id_for_answers_and_abstentions(chat, recorder):
    answered, _ = chat([_chunk(1, 0.9), _chunk(2, 0.8)])
    abstained, _ = chat([_chunk(1, 0.1)])

    assert answered[-1] == ("done", {"answer_id": "ans_FAKE"})
    assert abstained[-1] == ("done", {"answer_id": "ans_FAKE"})
    assert [r["abstained"] for _, r, _ in recorder.calls] == [False, True]


def test_recorder_gets_question_answer_and_filename(chat, recorder):
    chat([_chunk(1, 0.9), _chunk(2, 0.8)], question="Refund window?")

    question, result, threshold = recorder.calls[0]
    assert question == "Refund window?"
    assert result["answer"] == ANSWER
    assert [c["filename"] for c in result["citations"]] == ["f1.pdf", "f2.pdf"]
    assert threshold == 0.3


def test_citations_event_never_has_filename(chat):
    events, _ = chat([_chunk(1, 0.9), _chunk(2, 0.8)])

    assert '"filename"' not in json.dumps([d for n, d in events if n == "citations"])


def test_error_path_records_nothing_and_has_no_answer_id(chat, recorder):
    events, _ = chat([_chunk(1, 0.9)], RecordingLLM(fail=True))

    assert recorder.calls == []
    assert "answer_id" not in json.dumps(events)


def test_recording_failure_sends_error_not_the_answer(chat):
    def boom(*a):
        raise ConnectionError("db-host refused")

    events, _ = chat([_chunk(1, 0.9), _chunk(2, 0.8)], recorder=boom)

    assert events == [("error", {"message": "answer generation failed"})]
