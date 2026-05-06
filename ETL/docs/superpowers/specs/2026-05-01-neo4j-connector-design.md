# Neo4j Connector Design

Date: 2026-05-01

## Goal

Add the foundation for Stage 6: a Neo4j connector that can verify connectivity to the local database and apply the canonical `schema.cypher` file. This should fit the current Python pipeline style and leave full bulk data loading for a later, explicit Stage 6 expansion.

## Approved Approach

Implement a small connector plus a Stage 6 CLI:

- `pipeline/neo4j_connector.py` owns configuration, driver creation, schema splitting, schema application, and a simple connection smoke check.
- `pipeline/stage6_load_neo4j.py` exposes CLI commands for `check` and `schema`.
- `checks/check_neo4j.py` verifies that the configured database responds to a simple query.
- `requirements.txt` gains the official Neo4j Python driver.

This is preferred over a connection helper alone because it gives the documented Stage 6 a real entry point. It is smaller than a full bulk loader, which would need more decisions around stale cleanup, labels, relationship matching, embeddings, and chunk support.

## Configuration

Connection values are read from CLI flags first, then environment variables:

- `NEO4J_URI`, default `neo4j://127.0.0.1:7687`
- `NEO4J_USER`, default `neo4j`
- `NEO4J_PASSWORD`, no source-code default
- `NEO4J_DATABASE`, optional

The local password supplied by the user will be used only during local verification through an environment variable or CLI flag, not embedded in project source files.

## Behavior

The connector will:

1. Build a config object from flags and environment.
2. Open a Neo4j driver with basic auth.
3. Run a smoke query such as `RETURN 1 AS ok`.
4. Read `schema.cypher`, split statements on semicolons outside comments and blank lines, and execute them in order.
5. Print concise CLI output and return non-zero exit codes on failure.

## Error Handling

Missing password fails with a clear message before attempting to connect.

Neo4j connection or authentication errors are caught at the CLI boundary and reported without stack traces during normal command use.

Schema execution stops on the first failing statement and reports which statement number failed.

## Testing

Tests/checks will cover:

- Config precedence and missing password validation.
- Cypher statement splitting for the existing `schema.cypher` file.
- Local connection smoke check against the running Neo4j instance.

The local smoke check depends on the user's running database and supplied password, so it will be a `checks/` script rather than a pure unit test.

## Out Of Scope

This change will not yet bulk-load JSONL records into Neo4j, delete stale nodes, create embeddings, or implement retrieval. It only creates the connector and schema-loading foundation for the later Stage 6 loader.
