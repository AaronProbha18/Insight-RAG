from types import SimpleNamespace

from insight_rag.evals.golden import GoldenItem
from insight_rag.evals.scoring import QuestionScore, aggregate, gate, score_question
from insight_rag.retrieval import RetrievedChunk


class ScriptedJudge:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.replies.pop(0))


ANSWERABLE = GoldenItem(
    id="q1",
    question="What is the general return window?",
    answerable=True,
    reference_answer="14 days.",
    supporting=[{"document": "returns-policy", "locator": "p. 1"}],
)
UNANSWERABLE = GoldenItem(
    id="q2",
    question="Shipping time?",
    answerable=False,
    reference_answer="n/a",
    supporting=[],
)
STEMS = {"doc_R": "returns-policy"}


def _chunk(locator, text="t"):
    return RetrievedChunk(1, "doc_R", "Returns Policy", locator, text, 0.9)


def _answered(chunks, cited_text="CITED"):
    return {
        "answer": "14 days [c1].",
        "abstained": False,
        "citations": [{"id": "c1", "text": cited_text}],
        "chunks": chunks,
    }


ABSTAINED = {
    "answer": "I can't answer.",
    "abstained": True,
    "citations": [],
    "chunks": [],
}


def test_answered_correct_and_faithful():
    judge = ScriptedJudge("YES", "YES")
    chunks = [_chunk("p. 1", "CITED"), _chunk("p. 2", "UNCITED")]

    s = score_question(ANSWERABLE, _answered(chunks), judge, STEMS)

    assert (s.correct, s.faithful, s.hit) == (True, True, True)
    faithfulness_prompt = judge.prompts[0]
    assert "CITED" in faithfulness_prompt and "UNCITED" not in faithfulness_prompt


def test_abstained_on_answerable_is_wrong():
    judge = ScriptedJudge()

    s = score_question(ANSWERABLE, ABSTAINED, judge, STEMS)

    assert (s.correct, s.faithful) == (False, None)
    assert judge.prompts == []


def test_unanswerable_abstained():
    judge = ScriptedJudge()

    s = score_question(UNANSWERABLE, ABSTAINED, judge, STEMS)

    assert s.abstained and (s.correct, s.hit, s.faithful) == (None, None, None)
    assert judge.prompts == []


def test_unanswerable_answered():
    s = score_question(
        UNANSWERABLE, _answered([_chunk("p. 1")]), ScriptedJudge("NO"), STEMS
    )

    assert not s.abstained and s.faithful is False
    assert aggregate([s])["abstention"] == 0.0


def test_hit_rate_uses_stem_and_locator():
    s = score_question(
        ANSWERABLE, _answered([_chunk("p. 3")]), ScriptedJudge("YES", "YES"), STEMS
    )

    assert s.hit is False


def _score(answerable, abstained=False, correct=None, faithful=None, hit=None):
    return QuestionScore("q", answerable, abstained, correct, faithful, hit)


def test_aggregate_counts_each_metric_over_its_own_questions():
    scores = [
        _score(True, correct=True, faithful=True, hit=True),
        _score(True, correct=False, faithful=False, hit=False),
        _score(True, abstained=True, correct=False, hit=True),
        _score(False, abstained=True),
        _score(False, faithful=False),
    ]

    assert aggregate(scores) == {
        "correctness": 1 / 3,
        "faithfulness": 1 / 3,
        "abstention": 1 / 2,
        "hit_rate": 2 / 3,
    }


def test_connector_meets_the_bar():
    metrics = {
        "correctness": 0.86,
        "faithfulness": 0.91,
        "abstention": 0.83,
        "hit_rate": 0.5,
    }

    assert gate(metrics) == []


def test_connector_misses_the_bar():
    metrics = {
        "correctness": 0.86,
        "faithfulness": 0.74,
        "abstention": 0.83,
        "hit_rate": 0.9,
    }

    assert gate(metrics) == ["faithfulness"]


def test_empty_denominators_fail_closed():
    metrics = aggregate(
        [_score(True, abstained=True, correct=False), _score(False, abstained=True)]
    )

    assert metrics["faithfulness"] == 0.0
    assert "faithfulness" in gate(metrics)
