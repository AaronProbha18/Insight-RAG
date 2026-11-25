"""Load and validate golden sets and judge-validation labels (eval-plan.md)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

MIN_ITEMS = 30
MIN_UNANSWERABLE = 0.20


class Support(BaseModel):
    document: str = Field(min_length=1)
    locator: str = Field(min_length=1)


class GoldenItem(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    reference_answer: str = Field(min_length=1)
    supporting: list[Support]
    key_fact: str | None = None  # must appear in the supporting text (checked by tests)

    @model_validator(mode="after")
    def support_matches_answerable(self):
        if self.answerable and not self.supporting:
            raise ValueError(
                f"{self.id}: answerable items need at least one supporting source"
            )
        if not self.answerable and self.supporting:
            raise ValueError(
                f"{self.id}: unanswerable items must have no supporting sources"
            )
        return self


class LabelledCase(BaseModel):
    question: str
    answer: str
    label: bool
    reference_answer: str | None = None  # correctness
    sources: list[str] | None = None  # faithfulness


class Labelled(BaseModel):
    reviewed: bool
    correctness: list[LabelledCase]
    faithfulness: list[LabelledCase]


def _read(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_golden(path: Path) -> list[GoldenItem]:
    try:
        items = [GoldenItem.model_validate(raw) for raw in _read(path) or []]
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    if len(items) < MIN_ITEMS:
        raise ValueError(f"{path}: {len(items)} items, need at least {MIN_ITEMS}")
    unanswerable = sum(not i.answerable for i in items) / len(items)
    if unanswerable < MIN_UNANSWERABLE:
        raise ValueError(
            f"{path}: {unanswerable:.0%} unanswerable, need at least {MIN_UNANSWERABLE:.0%}"
        )
    return items


def load_labelled(path: Path) -> Labelled:
    try:
        return Labelled.model_validate(_read(path))
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
