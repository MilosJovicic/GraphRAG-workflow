# Stage 6 Skeleton Loader Design

Date: 2026-05-01

## Goal

Load the existing Stage 1–3 outputs (`./entities/*.jsonl` plus `./entities/_entities.jsonl`) into Neo4j as a queryable graph, idempotently, without embeddings or `context_prefix`. This validates `schema.cypher` against real data and gives downstream work a concrete graph to query before Stages 4 and 5 are built.

The skeleton loader is the "load Stage 1–3 only" subset of Stage 6 from CLAUDE.md. It does not write `embedding`, `indexed_text`, or `context_prefix` properties — those are produced by Stages 4–5 and will be added by a follow-up loader pass.

## Approved Approach

Add a new `pipeline/stage6_load.py` module that consumes `./entities/` and writes nodes and relationships through the existing `Neo4jConnector`. Wire two new subcommands into `pipeline/stage6_load_neo4j.py`: `load` and `wipe`. Add `checks/check_load.py` to verify the load matches the JSONL input.

## Inputs

- `./entities/*.jsonl` — one file per page, mixed-kind records: `page`, `section`, `codeblock`, `tablerow`, `callout`, `link`, `relationship`, `unresolved_link`.
- `./entities/_entities.jsonl` — the closed-vocabulary entity catalog (Tool, SettingKey, Hook, PermissionMode, Provider, MessageType).

`link` and `unresolved_link` records are skipped: external links are already captured in `Section.external_links`; internal links are realised as `LINKS_TO`/`LINKS_TO_PAGE` relationships by Stage 2.

## Architecture

Three logical layers in `pipeline/stage6_load.py`:

- **Reader** — streams JSONL records from `./entities/`, yielding (page_path, records) groups.
- **Reconciler** — for each page, compares incoming `Page.content_hash` with the value stored in Neo4j and decides whether to skip, MERGE-only, or DETACH DELETE descendants then MERGE fresh.
- **Writer** — phase-ordered MERGE of nodes, then MERGE of relationships, batched via `UNWIND $batch` of ~500 records per transaction.

The CLI (`pipeline/stage6_load_neo4j.py`) gains `load` and `wipe` subcommands alongside the existing `check` and `schema`. Connection arguments and env-var precedence match the existing subcommands.

## Data Flow

The loader runs three phases in order. Relationships are *always* held until phase 3, regardless of whether their source page's content_hash changed; this guarantees both endpoints exist before any edge is MERGE'd.

**Phase 1 — Entity catalog.** Load `_entities.jsonl` and MERGE all entity nodes once. Cross-page `MENTIONS`/`DEFINES` edges later in phase 3 then always have valid entity targets.

**Phase 2 — Per-page node reconciliation.** For each page JSONL file:

1. Bucket records by `kind` (`page`, `section`, `codeblock`, `tablerow`, `callout`, `relationship`). Buffer the relationship records for phase 3.
2. Read the existing `Page.content_hash` from Neo4j (single query).
3. **Page absent** → MERGE the Page and all its child nodes.
4. **Page present, hash matches** → no node writes for this page; phase 3 will re-MERGE its relationships (idempotent if nothing changed; restores any edges that were collateral-deleted by another page's hash-differs branch in this same run).
5. **Page present, hash differs** → DETACH DELETE descendants (children-first then Sections, then the Page itself), then MERGE the Page and all child nodes fresh.

**Phase 3 — Global relationship MERGE.** Concatenate the buffered relationship records from all pages, dedupe in memory by `(source_id, type, target_id)` (the MERGE key — see Identity and Idempotency), and MERGE in `UNWIND` batches grouped by relationship type.

Phase ordering matters: relationship MERGE must run after node MERGE for both endpoints, otherwise Neo4j would create label-less placeholder nodes.

## Identity and Idempotency

Stable IDs are inherited from Stage 1–3:

- `Page.url` (uniqueness constraint already in `schema.cypher`).
- `Section.id = "{page_url}#{anchor}"`.
- `CodeBlock.id`, `TableRow.id`, `Callout.id` — sha256-derived in Stage 1.
- Entities: `Tool.name`, `SettingKey.key`, `Hook.name`, `PermissionMode.name`, `Provider.name`, `MessageType.name`.

Relationship `MERGE` matches on `(source_label, source_id, type, target_label, target_id)`. Stage 2 also writes a relationship `id` (sha256 of type + endpoints + href); we keep it as a property but do not match on it. Consequence: when a Section has multiple `[text](href)` links to the same target, they collapse into one edge in Neo4j with the most-recently-written `href`/`text` properties. Parallel edges with identical endpoints would be retrieval noise, so this collapse is intentional.

Entity nodes are never deleted by the loader. The catalog is closed-vocabulary; entity disappearance from Stage 3 output is a Stage 3 bug, not a corpus-driven deletion.

## CLI Surface

```
python -m pipeline.stage6_load_neo4j check          # existing — connectivity smoke test
python -m pipeline.stage6_load_neo4j schema         # existing — apply schema.cypher
python -m pipeline.stage6_load_neo4j load [flags]   # new
python -m pipeline.stage6_load_neo4j wipe [flags]   # new
```

`load` flags:

- `--in PATH` — input dir, default `./entities/`.
- `--batch-size N` — UNWIND batch size, default 500.
- `--dry-run` — parse and validate JSONL, log planned operations, execute no Cypher.

`wipe` flags:

- `--yes` — skip the confirmation prompt. Without it, the command refuses to run non-interactively.

`wipe` deletes only loader-owned labels: `Page`, `Section`, `Chunk`, `CodeBlock`, `TableRow`, `Callout`, `Tool`, `SettingKey`, `Provider`, `MessageType`, `Hook`, `PermissionMode`. Schema and indexes remain.

CLI output is concise:

```
loaded 123 pages: 7979 nodes (Page=123, Section=2794, ...), 13932 relationships (HAS_SECTION=1117, ..., MENTIONS=3442, DEFINES=169); 0 pages skipped (hash unchanged)
```

## Error Handling

- **Missing input dir** → fail with `error: ./entities/ not found; run pipeline.stage3_entities first` and exit non-zero.
- **Schema not applied** → check `SHOW CONSTRAINTS` for `page_url` once at the start of `load`; if absent, fail with `error: schema not applied; run 'python -m pipeline.stage6_load_neo4j schema' first`.
- **Malformed JSONL** (missing required field, unknown `kind`) → fail fast with the file path and line number; do not silently skip.
- **Batch transaction failure** → bubble up the original Neo4j error annotated with the kind and IDs in the failing batch.
- **Connection lost mid-load** → re-run is idempotent: pages whose `content_hash` matches will short-circuit; pages mid-write will re-MERGE.

Surfaced as warnings, not failures:

- **Unresolved-link records in the JSONL** → skipped silently (already counted in Stage 2's warning).
- **Relationship target not found** (for example a `MENTIONS` edge pointing at an entity not in the catalog) → warn with source/target IDs and continue. Should not happen in practice; worth catching loudly if it does.

## Validation: `checks/check_load.py`

Runs against the loaded database. Asserts:

1. Every `page` record in `./entities/` exists in Neo4j with matching `content_hash`.
2. Per-label node counts in Neo4j match per-`kind` counts in JSONL for `page`/`section`/`codeblock`/`tablerow`/`callout`, and per-label counts for the six entity labels.
3. Per-relationship-type counts in Neo4j match the count of *unique* `(source_id, type, target_id)` triples in the JSONL `relationship` records (not the raw record count, since Stage 2 may emit multiple `LINKS_TO` records for the same endpoint pair with different `href` values, which the loader collapses to a single edge per the rule in Identity and Idempotency).
4. Structural integrity spot checks:
   - No `Section` without an incoming `HAS_SECTION` edge from a `Page`.
   - No `CodeBlock`/`TableRow`/`Callout` without an incoming `CONTAINS_*`/`HAS_CALLOUT` edge.
   - Every `Page.content_hash` is non-null.
5. Canonical entity catalog spot checks:
   - `Tool {name: "Read"}` exists.
   - `Hook {name: "SessionStart"}` exists.
   - `Provider {name: "bedrock"}` exists.
   - `MessageType {name: "ResultMessage"}` exists.

Output: `check_load.py OK: 123 pages, <N> nodes, <M> relationships matched JSONL`. On mismatch, exit non-zero with a diff.

## Testing Strategy

- Unit-style checks for the loader's record-bucketing and content-hash gating live in `checks/check_load_logic.py` (no Neo4j required).
- End-to-end loader behaviour is verified by `checks/check_load.py` against the running Neo4j instance.
- Existing checks (`check_parse.py`, `check_links.py`, `check_entities.py`, `check_neo4j_config.py`, `check_neo4j.py`) must continue to pass.

## Out Of Scope

- Embeddings, `indexed_text`, `context_prefix` (Stages 4–5).
- `Chunk` nodes (depend on Stage 4/5 chunking that is not yet implemented).
- Vector and full-text indexes are already created by `schema.cypher`; the loader does not need to defer them. They will be populated by the Stage 5 follow-up loader pass.
- Cross-encoder reranking, query interface, evaluation harness.
