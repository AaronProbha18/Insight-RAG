"""Contract checks any connector package can run against its own fixtures.

Layout: valid files in `fixtures_dir/`, files that must raise `ParseError`
in `fixtures_dir/corrupt/`. Only files with the connector's extensions are used.
"""

from pathlib import Path

from insight_rag.connectors.contract import CONTRACT_VERSION, FileConnector, ParseError


def _files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )


def run_conformance(connector: FileConnector, fixtures_dir: Path) -> None:
    assert connector.contract_version == CONTRACT_VERSION, (
        f"contract_version {connector.contract_version}, core supports {CONTRACT_VERSION}"
    )
    valid = _files(fixtures_dir, connector.extensions)
    assert valid, (
        f"no fixtures with extensions {connector.extensions} in {fixtures_dir}"
    )

    for path in valid:
        data = path.read_bytes()
        doc = connector.parse(data, path.name)
        assert doc.sections, f"{path.name}: no sections"
        for i, section in enumerate(doc.sections):
            assert section.locator, f"{path.name}: section {i} has an empty locator"
        assert connector.parse(data, path.name) == doc, (
            f"{path.name}: parsing the same bytes twice gave different output"
        )

    for path in _files(fixtures_dir / "corrupt", connector.extensions):
        try:
            connector.parse(path.read_bytes(), path.name)
        except ParseError:
            continue
        except Exception as exc:
            raise AssertionError(
                f"corrupt/{path.name}: raised {type(exc).__name__} instead of ParseError"
            ) from exc
        raise AssertionError(
            f"corrupt/{path.name}: parsed instead of raising ParseError"
        )
