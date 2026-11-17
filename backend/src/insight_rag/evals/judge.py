"""LLM judge for correctness and faithfulness, validated against human labels."""

from dataclasses import dataclass

from insight_rag.evals.golden import Labelled

MIN_AGREEMENT = 0.80

CORRECTNESS = """You are grading an answer against a reference answer.
Does the ANSWER state the same facts as the REFERENCE for this QUESTION?
Extra wording is fine. A missing or different fact means NO.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Reply with exactly YES or NO."""

FAITHFULNESS = """You are checking an answer against its sources.
Is every factual claim in the ANSWER supported by the SOURCES?
Anything not stated in the sources means NO.

QUESTION: {question}
SOURCES:
{sources}
ANSWER: {answer}

Reply with exactly YES or NO."""


class JudgeNotValidated(Exception):
    """The judge may not be trusted, so no gate result may be reported."""


@dataclass(frozen=True)
class JudgeValidation:
    model: str
    agreement: dict[str, float]


def _verdict(llm, prompt: str) -> bool:
    words = llm.invoke(prompt).content.split()
    # Fail closed: anything but a clear YES counts as NO.
    return bool(words) and words[0].strip(".,:;!").upper() == "YES"


def judge_correctness(llm, question: str, reference_answer: str, answer: str) -> bool:
    return _verdict(
        llm,
        CORRECTNESS.format(
            question=question, reference=reference_answer, answer=answer
        ),
    )


def judge_faithfulness(llm, question: str, sources: list[str], answer: str) -> bool:
    joined = "\n".join(f"- {s}" for s in sources)
    return _verdict(
        llm, FAITHFULNESS.format(question=question, sources=joined, answer=answer)
    )


def validate_judge(llm, labelled: Labelled, model_name: str) -> JudgeValidation:
    if not labelled.reviewed:
        raise JudgeNotValidated(
            "judge labels are not human-reviewed; check every case and set "
            "`reviewed: true` in evals/judge/labelled.yaml"
        )
    agreement = {}
    for metric, cases in (
        ("correctness", labelled.correctness),
        ("faithfulness", labelled.faithfulness),
    ):
        hits = 0
        for c in cases:
            if metric == "correctness":
                verdict = judge_correctness(
                    llm, c.question, c.reference_answer, c.answer
                )
            else:
                verdict = judge_faithfulness(llm, c.question, c.sources, c.answer)
            hits += verdict == c.label
        agreement[metric] = hits / len(cases)
        if agreement[metric] < MIN_AGREEMENT:
            raise JudgeNotValidated(
                f"judge {model_name!r} agreed on {hits}/{len(cases)} {metric} cases "
                f"({agreement[metric]:.2f} < {MIN_AGREEMENT:.2f}); no gate result"
            )
    return JudgeValidation(model_name, agreement)
