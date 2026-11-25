import logging
from importlib.metadata import EntryPoint

from plumbline.connectors import registry


def _install(monkeypatch, *classes, legacy=()):
    def make(names, group):
        return [
            EntryPoint(name=c, value=f"fake_connectors:{c}", group=group) for c in names
        ]

    by_group = {
        registry.GROUP: make(classes, registry.GROUP),
        registry.LEGACY_GROUP: make(legacy, registry.LEGACY_GROUP),
    }
    monkeypatch.setattr(registry, "entry_points", lambda group: by_group[group])


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


def test_old_group_connector_is_skipped_with_warning(monkeypatch, caplog):
    _install(
        monkeypatch, "GoodConnector", legacy=["FutureConnector", "BrokenConnector"]
    )

    with caplog.at_level(logging.WARNING):
        loaded = registry.load_connectors()

    assert list(loaded) == ["good"]
    old_group_warnings = [r for r in caplog.records if "old group" in r.getMessage()]
    assert len(old_group_warnings) == 2
    assert all("plumbline.connectors" in r.getMessage() for r in old_group_warnings)


def test_valid_connector_on_old_group_alone_is_not_loaded(monkeypatch):
    _install(monkeypatch, legacy=["GoodConnector"])

    assert registry.load_connectors() == {}
