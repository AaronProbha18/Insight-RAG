# ADR 0001: Connector contract

## Status

Accepted. Scope for v1: ingestion connectors for PDF and Word only.

## Context

Connectors are the project's main extension point. Because the project is
open source, connectors will be written by people outside the core team,
so the contract must be small, stable, versioned, and testable without
access to the real source system.

## Proposed Decisions

- **Ingestion connectors** implement a small protocol, roughly:
  enumerate items, fetch content, report deletions, and resume from a
  sync cursor. They emit a normalized document (id, source, uri, title,
  content, metadata, updated_at). Exact shape is
  defined in the first connector change's design.md.
- **Discovery via Python entry points** (an `insight_rag.connectors`
  group). A third-party connector is a separate pip-installable package;
  core needs no change to load it. Built-in connectors register the same
  way, so there is one path.
- **Contract versioning**: the protocol carries a version; core states
  which versions it supports, and breaking changes need a superseding ADR.
- **Conformance test kit**: core ships a reusable test suite that any
  connector package runs against a fixture corpus. "Passes conformance"
  is the bar for listing a connector as supported.
- Each connector declares its config schema (credentials referenced by
  env var name, never stored in config files).

## Open Questions

1. RESOLVED: the goal is "bring any source" - files, databases,
   warehouses. That needs two connector kinds, delivered in phases:
   - **Ingestion connectors** (this protocol): content copied into the
     index. Phase 1: PDF and Word documents. Later: Excel/CSV, Confluence,
     SharePoint, etc.
   - **Query connectors** (separate protocol, own ADR when needed): live
     structured sources such as Postgres or Snowflake, queried at question
     time (e.g. text-to-SQL) and exposed to the graph as tools. Embedding
     rows of a warehouse table answers aggregate questions badly, so
     structured sources should not be forced through ingestion.
   Spreadsheets sit on the boundary: small ones ingest as text, large
   ones are better served as query connectors - decided per connector.
2. RESOLVED: per-document permissions (ACL-aware retrieval) are out of
   scope for v1. No placeholder field is reserved; adding permissions
   later is a contract version bump under its own ADR.

## Consequences

- The conformance kit and fixture corpus are part of the first connector
  change, not an afterthought.
