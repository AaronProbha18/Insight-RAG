from types import SimpleNamespace

import pytest

from insight_rag.evals.golden import Labelled, LabelledCase
from insight_rag.evals.judge import (
    JudgeNotValidated,
    judge_correctness,
    judge_faithfulness,
    validate_judge,
)


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.replies.pop(0))


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("YES", True),
        ("yes.", True),
        (" Yes, because it matches.", True),
        ("NO", False),
        ("MAYBE", False),
        ("", False),
    ],
)
def test_verdict_parsing(reply, verdict):
    assert judge_correctness(ScriptedLLM(reply), "q", "ref", "ans") is verdict


def test_prompts_carry_inputs():
    llm = ScriptedLLM("YES", "YES")

    judge_correctness(llm, "Q?", "REF", "ANS")
    judge_faithfulness(llm, "Q?", ["SRC1", "SRC2"], "ANS")

    correctness, faithfulness = llm.prompts
    assert all(s in correctness for s in ("Q?", "REF", "ANS"))
    assert all(s in faithfulness for s in ("Q?", "SRC1", "SRC2", "ANS"))


def _labelled(reviewed=True) -> Labelled:
    labels = [True] * 5 + [False] * 5
    return Labelled(
        reviewed=reviewed,
        correctness=[
            LabelledCase(question="q", reference_answer="r", answer="a", label=x)
            for x in labels
        ],
        faithfulness=[
            LabelledCase(question="q", sources=["s"], answer="a", label=x)
            for x in labels
        ],
    )


def _replies(wrong: int) -> list[str]:
    """Replies matching the 5 True + 5 False labels, with `wrong` of them flipped."""
    correct = ["YES"] * 5 + ["NO"] * 5
    return [
        ("NO" if r == "YES" else "YES") if i < wrong else r
        for i, r in enumerate(correct)
    ]


def test_validation_passes_at_threshold():
    llm = ScriptedLLM(*_replies(2), *_replies(2))

    result = validate_judge(llm, _labelled(), "ollama:judge")

    assert result.agreement == {"correctness": 0.8, "faithfulness": 0.8}
    assert result.model == "ollama:judge"


def test_unreliable_judge_stops():
    with pytest.raises(
        JudgeNotValidated, match=r"'ollama:judge'.*6/10 correctness.*0\.60"
    ):
        validate_judge(ScriptedLLM(*_replies(4)), _labelled(), "ollama:judge")


def test_unreviewed_labels_refuse_without_calling_judge():
    llm = ScriptedLLM()

    with pytest.raises(JudgeNotValidated, match="reviewed: true"):
        validate_judge(llm, _labelled(reviewed=False), "m")

    assert llm.prompts == []
