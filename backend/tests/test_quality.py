import json
from pathlib import Path

import pytest

from plumbline.quality import GATED, load_quality

FIX = Path(__file__).parent / "fixtures" / "quality"
RUNNING = {"llm": "fake:chat", "embeddings": "fake:embed", "threshold": 0.3}


def load(name, running=RUNNING):
    return load_quality(FIX / f"{name}.json", running)


def test_both_pass():
    q = load("both_pass")
    assert q["state"] == "measured" and q["result"] == "PASS"
    assert q["run_at"] == "2026-11-25T09:00:00+00:00"
    assert q["models"] == {
        "chat": "fake:chat",
        "embeddings": "fake:embed",
        "judge": "fake:judge",
    }
    assert set(q["connectors"]) == {"pdf", "docx"}
    for c in q["connectors"].values():
        assert c["result"] == "PASS" and c["failed"] == []
        assert tuple(c["metrics"]) == GATED


def test_docx_fails_faithfulness():
    q = load("docx_fails_faithfulness")
    assert q["result"] == "FAIL"
    assert q["connectors"]["docx"]["failed"] == ["faithfulness"]
    assert q["connectors"]["pdf"]["failed"] == []


@pytest.mark.parametrize(
    "name", ["exit2", "sampled", "missing_field", "malformed", "nope"]
)
def test_not_measured(name):
    q = load(name)
    assert q == {"state": "not_measured", "eval_command": "uv run plumbline-eval"}


def test_config_matches():
    q = load("both_pass")
    assert q["config_matches"] is True and q["config_differences"] == []


@pytest.mark.parametrize(
    "change,name",
    [
        ({"llm": "other"}, "chat"),
        ({"embeddings": "other"}, "embeddings"),
        ({"threshold": 0.5}, "threshold"),
    ],
)
def test_config_difference(change, name):
    q = load("both_pass", {**RUNNING, **change})
    assert q["config_matches"] is False and q["config_differences"] == [name]


def test_judge_not_compared():
    # the runner's judge is fake:judge; running config has no judge key at all
    assert load("both_pass")["config_matches"] is True


@pytest.mark.parametrize("name", ["both_pass", "docx_fails_faithfulness"])
def test_no_internal_fields_leak(name):
    out = json.dumps(load(name))
    for leak in (
        "host", "git_commit", "report", "/secret", "latency_ms", "golden_version",
        "judge_agreement", "reason", "top_k", "duration_s", "extra_metric", "abc1234",
    ):  # fmt: skip
        assert leak not in out
