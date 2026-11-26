"""The toolkit does only what Plumbline needs: no migration generation."""
import re

from conftest import ROOT


def test_migration_tooling_absent():
    assert not (ROOT / "tools" / "migrate.py").exists()
    makefile = (ROOT / "Makefile").read_text()
    assert not re.search(r"^(migrate|check):", makefile, re.M)
    leftovers = [p for p in (ROOT / "models").rglob("*") if p.name in ("model.lock.json", "migrations")]
    assert leftovers == []
