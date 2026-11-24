import importlib

import pytest

import plumbline.api.app as app_module
from plumbline.config import LegacyConfigError, check_legacy_env
from plumbline.evals import runner

SUFFIXES = [
    "LLM",
    "EMBEDDINGS",
    "EMBEDDING_DIM",
    "JUDGE_LLM",
    "MODEL_BASE_URL",
    "ABSTAIN_THRESHOLD",
    "UI_DIR",
]


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_each_old_variable_names_its_replacement(suffix):
    with pytest.raises(LegacyConfigError, match=f"-> PLUMBLINE_{suffix}"):
        check_legacy_env({f"INSIGHT_RAG_{suffix}": "x"})


def test_new_names_and_unrelated_variables_pass():
    check_legacy_env({"PLUMBLINE_LLM": "x", "DATABASE_URL": "y"})
    check_legacy_env({})


def test_all_offenders_are_listed_and_empty_values_count():
    environ = {"INSIGHT_RAG_LLM": "", "INSIGHT_RAG_UI_DIR": "x"}

    with pytest.raises(LegacyConfigError) as exc:
        check_legacy_env(environ)

    assert "PLUMBLINE_LLM" in str(exc.value) and "PLUMBLINE_UI_DIR" in str(exc.value)


def test_app_refuses_to_start(monkeypatch):
    monkeypatch.setenv("INSIGHT_RAG_LLM", "x")

    with pytest.raises(LegacyConfigError, match="PLUMBLINE_LLM"):
        importlib.reload(app_module)


def test_eval_cli_stops_without_a_gate_result(monkeypatch, capsys):
    monkeypatch.setenv("INSIGHT_RAG_LLM", "x")

    assert runner.main([]) == 2
    assert "PLUMBLINE_LLM" in capsys.readouterr().err
