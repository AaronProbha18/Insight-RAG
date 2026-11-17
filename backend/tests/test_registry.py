import logging
from importlib.metadata import EntryPoint

from insight_rag.connectors import registry


def _install(monkeypatch, *classes):
    eps = [
        EntryPoint(name=c, value=f"fake_connectors:{c}", group=registry.GROUP)
        for c in classes
    ]
    monkeypatch.setattr(registry, "entry_points", lambda group: eps)


def test_registers_entry_point_connector(monkeypatch):
    _install(monkeypatch, "GoodConnector")

    assert list(registry.load_connectors()) == ["good"]


def test_skips_version_mismatch_with_warning(monkeypatch, caplog):
    _install(monkeypatch, "FutureConnector", "GoodConnector")

    with caplog.at_level(logging.WARNING):
        loaded = registry.load_connectors()

    assert list(loaded) == ["good"]
    assert "'future' declares contract_version 2, core supports 1" in caplog.text


def test_broken_connector_does_not_stop_loading(monkeypatch, caplog):
    _install(monkeypatch, "BrokenConnector", "GoodConnector")

    with caplog.at_level(logging.WARNING):
        loaded = registry.load_connectors()

    assert list(loaded) == ["good"]
    assert "'BrokenConnector' failed to load" in caplog.text


def test_no_connectors_installed(monkeypatch):
    _install(monkeypatch)

    assert registry.load_connectors() == {}
