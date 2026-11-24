import fake_connectors as fakes
import pytest

from plumbline.connectors.testing import run_conformance


@pytest.fixture
def fixtures(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "ignored.md").write_text("not a .txt")
    (tmp_path / "corrupt").mkdir()
    (tmp_path / "corrupt" / "bad.txt").write_bytes(b"\xff\xfe")
    return tmp_path


def test_conforming_connector_passes(fixtures):
    run_conformance(fakes.GoodConnector(), fixtures)


@pytest.mark.parametrize(
    ("connector", "message"),
    [
        (fakes.FutureConnector, "contract_version 2"),
        (fakes.EmptyConnector, "no sections"),
        (fakes.NondeterministicConnector, "different output"),
        (fakes.CrashingConnector, "raised ValueError instead of ParseError"),
    ],
)
def test_broken_connector_fails(fixtures, connector, message):
    with pytest.raises(AssertionError, match=message):
        run_conformance(connector(), fixtures)


def test_empty_fixtures_dir_fails(tmp_path):
    with pytest.raises(AssertionError, match="no fixtures"):
        run_conformance(fakes.GoodConnector(), tmp_path)
