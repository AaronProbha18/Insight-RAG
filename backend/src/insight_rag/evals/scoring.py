"""Score graph results against golden items; aggregate metrics; apply gates (design D9)."""

from dataclasses import dataclass, field

from insight_rag.evals.golden import GoldenItem
from insight_rag.evals.judge import judge_correctness, judge_faithfulness

GATE = 0.80
GATED = ("correctness", "faithfulness", "abstention")


@dataclass(frozen=True)
class QuestionScore:
    id: str
    answerable: bool
    abstained: bool
    correct: bool | None  # answerable only
    faithful: bool | None  # answered only
    hit: bool | None  # answerable only
    answer: str = ""
    citations: list[dict] = field(default_factory=list)


def score_question(
    item: GoldenItem, result: dict, judge_llm, doc_stems: dict[str, str]
):
    abstained = result["abstained"]
    correct = faithful = hit = None
    if not abstained:
        sources = [c["text"] for c in result["citations"]]
        faithful = judge_faithfulness(
            judge_llm, item.question, sources, result["answer"]
        )
    if item.answerable:
        correct = not abstained and judge_correctness(
            judge_llm, item.question, item.reference_answer, result["answer"]
        )
        wanted = {(s.document, s.locator) for s in item.supporting}
        hit = any(
            (doc_stems.get(c.document_id), c.locator) in wanted
            for c in result.get("chunks", [])
        )
    return QuestionScore(
        item.id,
        item.answerable,
        abstained,
        correct,
        faithful,
        hit,
        result["answer"],
        result["citations"],
    )


def _rate(values: list[bool]) -> float:
    # Fail closed: no applicable questions scores 0, never a free pass.
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[QuestionScore]) -> dict[str, float]:
    return {
        "correctness": _rate([s.correct for s in scores if s.correct is not None]),
        "faithfulness": _rate([s.faithful for s in scores if s.faithful is not None]),
        "abstention": _rate([s.abstained for s in scores if not s.answerable]),
        "hit_rate": _rate([s.hit for s in scores if s.hit is not None]),
    }


def gate(metrics: dict[str, float]) -> list[str]:
    """Gated metrics below the bar; empty means PASS."""
    return [m for m in GATED if metrics[m] < GATE]
