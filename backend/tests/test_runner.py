import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
import yaml
from fake_embeddings import FailingEmbeddings, FakeEmbeddings

from plumbline.connectors.builtin.pdf import PdfConnector
from plumbline.evals import runner

pytestmark = pytest.mark.db

EVALS = Path(__file__).parents[2] / "evals"
IDS = {"llm": "fake:chat", "embeddings": "fake:embed", "judge": "fake:judge"}
UNANSWERABLE = [
    i["question"]
    for i in yaml.safe_load((EVALS / "connectors/pdf/golden.yaml").read_text())
    if not i["answerable"]
]


class FakeLLM:
    """Abstains on golden unanswerables (if `abstains`), otherwise answers citing c1."""

    def __init__(self, abstains=True):
        self.abstains = abstains

    def invoke(self, prompt):
        question = prompt.rsplit("Question: ", 1)[-1]
        if self.abstains and question in UNANSWERABLE:
            return SimpleNamespace(content="NOT_IN_SOURCES")
        return SimpleNamespace(content="Per the policy [c1].")


class YesJudge:
    def invoke(self, prompt):
        return SimpleNamespace(content="YES")


@pytest.fixture
def evals_dir(tmp_path, database_url, monkeypatch):
    """A copy of evals/ with reviewed labels, and DATABASE_URL on the test server."""
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("PLUMBLINE_EMBEDDING_DIM", "8")
    root = tmp_path / "evals"
    shutil.copytree(EVALS / "connectors" / "pdf", root / "connectors" / "pdf")
    labelled = yaml.safe_load((EVALS / "judge" / "labelled.yaml").read_text())
    labelled["reviewed"] = True
    # A YES-only judge must agree with every label for these runner tests.
    for metric in ("correctness", "faithfulness"):
        labelled[metric] = [{**c, "label": True} for c in labelled[metric]]
    (root / "judge").mkdir()
    (root / "judge" / "labelled.yaml").write_text(yaml.safe_dump(labelled))
    return root


def _run(evals_dir, tmp_path, llm=None, embeddings=None, sample=None, progress=None):
    report = tmp_path / "report.json"
    code = runner.run(
        connectors={"pdf": PdfConnector()},
        llm=llm or FakeLLM(),
        judge=YesJudge(),
        embeddings=embeddings or FakeEmbeddings(),
        model_ids=IDS,
        report_path=report,
        evals_dir=evals_dir,
        sample=sample,
        progress=progress,
    )
    return code, report


class RecordingProgress:
    """Stands in for rich's Progress: records each task's total and advances."""

    def __init__(self):
        self.tasks = {}

    def add_task(self, description, total):
        self.tasks[description] = {"total": total, "done": 0}
        return description

    def advance(self, task):
        self.tasks[task]["done"] += 1


def _summary(report):
    return json.loads(report.with_name("report.summary.json").read_text())


def _eval_databases(database_url):
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT datname FROM pg_database WHERE datname LIKE 'plumbline_eval_%'"
        ).fetchall()
    return rows


def test_passing_run_writes_report_and_exits_0(evals_dir, tmp_path):
    code, report = _run(evals_dir, tmp_path)

    data = json.loads(report.read_text())
    pdf = data["connectors"]["pdf"]
    assert code == 0
    assert data["models"] == IDS
    assert set(pdf["metrics"]) == {
        "correctness",
        "faithfulness",
        "abstention",
        "hit_rate",
    }
    assert pdf["result"] == "PASS" and len(pdf["questions"]) == 40
    assert len(pdf["golden_version"]) == 12


def test_failing_connector_exits_1(evals_dir, tmp_path):
    code, report = _run(evals_dir, tmp_path, llm=FakeLLM(abstains=False))

    pdf = json.loads(report.read_text())["connectors"]["pdf"]
    assert code == 1
    assert pdf["result"] == "FAIL" and pdf["failed"] == ["abstention"]


def test_unvalidated_judge_exits_2_without_db(evals_dir, tmp_path, database_url):
    path = evals_dir / "judge" / "labelled.yaml"
    path.write_text(path.read_text().replace("reviewed: true", "reviewed: false"))

    code, report = _run(evals_dir, tmp_path)

    assert code == 2 and not report.exists()
    assert _eval_databases(database_url) == []


def test_eval_databases_are_dropped(evals_dir, tmp_path, database_url):
    _run(evals_dir, tmp_path)
    with pytest.raises(ConnectionError):
        _run(evals_dir, tmp_path, embeddings=FailingEmbeddings())

    assert _eval_databases(database_url) == []


def test_sample_keeps_ratio(evals_dir, tmp_path):
    _, report = _run(evals_dir, tmp_path, sample=8)

    data = json.loads(report.read_text())
    questions = data["connectors"]["pdf"]["questions"]
    assert data["sample"] == 8
    assert [q["answerable"] for q in questions].count(False) == 2 and len(
        questions
    ) == 8


def test_cli_parses_arguments(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(runner, "run", lambda **kw: seen.update(kw) or 0)
    monkeypatch.setenv("PLUMBLINE_LLM", IDS["llm"])
    monkeypatch.setenv("PLUMBLINE_EMBEDDINGS", IDS["embeddings"])
    monkeypatch.setenv("PLUMBLINE_JUDGE_LLM", IDS["judge"])
    monkeypatch.setattr(runner.models, "chat_model", lambda: "llm")
    monkeypatch.setattr(runner.models, "judge_model", lambda: "judge")
    monkeypatch.setattr(runner.models, "embeddings", lambda: "emb")

    code = runner.main(
        ["--connectors", "pdf", "--report", str(tmp_path / "r.json"), "--sample", "4"]
    )

    assert code == 0
    assert list(seen["connectors"]) == ["pdf"] and seen["sample"] == 4
    assert seen["report_path"] == tmp_path / "r.json" and seen["model_ids"] == IDS


def test_summary_is_small_and_shareable(evals_dir, tmp_path):
    code, report = _run(evals_dir, tmp_path)

    summary = _summary(report)
    assert summary["exit_code"] == code == 0 and summary["result"] == "PASS"
    assert summary["models"] == IDS and summary["git_commit"]
    assert set(summary["host"]) == {"platform", "machine", "cpus", "python"}
    pdf = summary["connectors"]["pdf"]
    assert pdf["questions"] == 40 and pdf["result"] == "PASS"
    assert pdf["unanswerable_questions"] == len(UNANSWERABLE)
    assert pdf["answerable_questions"] + pdf["unanswerable_questions"] == 40
    assert set(pdf["metrics"]) == {
        "correctness",
        "faithfulness",
        "abstention",
        "hit_rate",
    }
    assert '"answer"' not in json.dumps(summary)  # no per-question text
    assert summary["duration_s"] >= 0


def test_summary_written_when_judge_not_validated(evals_dir, tmp_path):
    path = evals_dir / "judge" / "labelled.yaml"
    path.write_text(path.read_text().replace("reviewed: true", "reviewed: false"))

    code, report = _run(evals_dir, tmp_path)

    summary = _summary(report)
    assert code == 2 and not report.exists()
    assert summary["result"] == "NO_GATE_RESULT" and summary["exit_code"] == 2
    assert "reviewed" in summary["reason"]


def test_progress_tracks_every_step(evals_dir, tmp_path):
    progress = RecordingProgress()

    _run(evals_dir, tmp_path, progress=progress)

    assert progress.tasks == {
        "judge validation": {"total": 1, "done": 1},
        "pdf: ingest": {"total": 4, "done": 4},
        "pdf: questions": {"total": 40, "done": 40},
    }
