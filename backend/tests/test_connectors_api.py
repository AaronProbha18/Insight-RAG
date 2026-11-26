from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_registry import _install

from plumbline.api.app import app, get_connectors

FIX = Path(__file__).parent / "fixtures" / "quality"


@pytest.fixture
def client():
    get_connectors.cache_clear()
    yield TestClient(app)
    get_connectors.cache_clear()


def test_built_ins_listed_sorted_with_allowlisted_fields(client):
    r = client.get("/connectors")

    assert r.status_code == 200
    assert r.json() == [
        {
            "name": "docx",
            "extensions": [".docx"],
            "contract_version": 1,
            "builtin": True,
        },
        {"name": "pdf", "extensions": [".pdf"], "contract_version": 1, "builtin": True},
    ]


def test_third_party_connector_is_not_builtin(client, monkeypatch):
    _install(monkeypatch, "GoodConnector")

    assert client.get("/connectors").json() == [
        {
            "name": "good",
            "extensions": [".txt"],
            "contract_version": 1,
            "builtin": False,
        }
    ]


def test_failed_connectors_not_listed(client, monkeypatch):
    _install(monkeypatch, "BrokenConnector", "FutureConnector", "GoodConnector")

    assert [c["name"] for c in client.get("/connectors").json()] == ["good"]


def test_names_match_quality_keys(client, monkeypatch):
    monkeypatch.setenv("PLUMBLINE_EVAL_SUMMARY", str(FIX / "both_pass.json"))

    names = {c["name"] for c in client.get("/connectors").json()}

    assert names == set(client.get("/quality").json()["connectors"])
