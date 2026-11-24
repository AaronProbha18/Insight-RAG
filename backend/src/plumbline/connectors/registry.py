"""Discovers connectors registered under the `insight_rag.connectors` entry-point group."""

import logging
from importlib.metadata import entry_points

from plumbline.connectors.contract import CONTRACT_VERSION, FileConnector

GROUP = "insight_rag.connectors"
log = logging.getLogger(__name__)


def load_connectors() -> dict[str, FileConnector]:
    # ponytail: duplicate names - last loaded wins; reject duplicates if it bites.
    connectors: dict[str, FileConnector] = {}
    for ep in entry_points(group=GROUP):
        try:
            connector = ep.load()()
        except Exception:
            log.warning("connector %r failed to load; skipped", ep.name, exc_info=True)
            continue
        if connector.contract_version != CONTRACT_VERSION:
            log.warning(
                "connector %r declares contract_version %s, core supports %s; skipped",
                connector.name,
                connector.contract_version,
                CONTRACT_VERSION,
            )
            continue
        connectors[connector.name] = connector
    return connectors
