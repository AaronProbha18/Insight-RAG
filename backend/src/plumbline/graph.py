"""Answer graph: retrieve -> generate | abstain (design D7)."""

import re
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from plumbline.retrieval import RetrievedChunk

ABSTAIN_MESSAGE = "I can't answer that from the indexed documents."
NOT_IN_SOURCES = "NOT_IN_SOURCES"
# ponytail: placeholder; calibrated once on the golden sets in the 7.4 baseline.
DEFAULT_THRESHOLD = 0.3

PROMPT = """Answer the question using ONLY the sources below.
Cite every claim with the source id in square brackets, e.g. [c1].
If the sources do not contain the answer, reply with exactly {sentinel} and nothing else.

Sources:
{sources}

Question: {question}"""

Retriever = Callable[[str, int], list[RetrievedChunk]]


class State(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]
    answer: str
    citations: list[dict]
    abstained: bool


def _abstain(_: State) -> State:
    return {"answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True}


def build_graph(
    retriever: Retriever, llm, threshold: float = DEFAULT_THRESHOLD, top_k: int = 5
):
    def retrieve(state: State) -> State:
        return {"chunks": retriever(state["question"], top_k)}

    def route(state: State) -> str:
        chunks = state["chunks"]
        return "generate" if chunks and chunks[0].score >= threshold else "abstain"

    def generate(state: State) -> State:
        chunks = state["chunks"]
        sources = "\n\n".join(
            f"[c{i}] ({c.title}, {c.locator})\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )
        prompt = PROMPT.format(
            sentinel=NOT_IN_SOURCES, sources=sources, question=state["question"]
        )
        answer = llm.invoke(prompt).content.strip()

        cited = []
        for n in re.findall(r"\[c(\d+)\]", answer):
            i = int(n)
            if 1 <= i <= len(chunks) and i not in cited:
                cited.append(i)
        # An answer that says it cannot answer, or cites nothing, cannot be trusted.
        if NOT_IN_SOURCES in answer or not cited:
            return _abstain(state)
        citations = [
            {
                "id": f"c{i}",
                "document_id": chunks[i - 1].document_id,
                "title": chunks[i - 1].title,
                "locator": chunks[i - 1].locator,
                "text": chunks[i - 1].text,
            }
            for i in cited
        ]
        return {"answer": answer, "citations": citations, "abstained": False}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("abstain", _abstain)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, ["generate", "abstain"])
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()
