from langchain_core.language_models.fake_chat_models import FakeListChatModel

from plumbline.graph import ABSTAIN_MESSAGE, build_graph
from plumbline.retrieval import RetrievedChunk


def _chunk(i: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + i,
        document_id=f"doc_{i}",
        title=f"Doc {i}",
        locator=f"p. {i}",
        text=f"text {i}",
        score=score,
    )


class RecordingLLM:
    def __init__(self, reply: str):
        self.model = FakeListChatModel(responses=[reply])
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.model.invoke(prompt)


def _run(chunks, reply="unused", threshold=0.3):
    llm = RecordingLLM(reply)
    graph = build_graph(lambda q, k: chunks[:k], llm, threshold=threshold)
    return graph.invoke({"question": "What is the refund window?"}), llm


def test_answer_with_citations():
    result, _ = _run([_chunk(1, 0.9)], "Refunds take 14 days [c1].")

    assert result["abstained"] is False
    assert result["answer"] == "Refunds take 14 days [c1]."
    assert result["citations"] == [
        {
            "id": "c1",
            "document_id": "doc_1",
            "title": "Doc 1",
            "filename": "",
            "locator": "p. 1",
            "text": "text 1",
        }
    ]


def test_only_cited_chunks_returned():
    chunks = [_chunk(1, 0.9), _chunk(2, 0.8), _chunk(3, 0.7)]

    result, _ = _run(chunks, "Shoes: 30 days [c2]. Worn indoors is fine [c2].")

    assert [c["id"] for c in result["citations"]] == ["c2"]
    assert result["citations"][0]["title"] == "Doc 2"


def test_below_threshold_abstains_without_llm():
    result, llm = _run([_chunk(1, 0.1)], threshold=0.3)

    assert result["abstained"] and result["citations"] == []
    assert result["answer"] == ABSTAIN_MESSAGE
    assert llm.prompts == []


def test_empty_index_abstains_without_llm():
    result, llm = _run([])

    assert result["abstained"] and result["citations"] == []
    assert llm.prompts == []


def test_model_says_not_in_sources():
    result, _ = _run([_chunk(1, 0.9)], "NOT_IN_SOURCES")

    assert result["abstained"] and result["citations"] == []


def test_uncited_answer_is_abstention():
    result, _ = _run([_chunk(1, 0.9)], "Probably 30 days.")

    assert result["abstained"] and result["citations"] == []


def test_prompt_contains_sources():
    _, llm = _run([_chunk(1, 0.9), _chunk(2, 0.8)], "x [c1]")

    (prompt,) = llm.prompts
    assert "[c1] (Doc 1, p. 1)\ntext 1" in prompt
    assert "[c2] (Doc 2, p. 2)\ntext 2" in prompt
    assert "NOT_IN_SOURCES" in prompt
    assert "What is the refund window?" in prompt
