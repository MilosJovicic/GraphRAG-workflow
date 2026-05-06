# Contextual Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the pilot review issues before running the full contextual embeddings corpus job.

**Architecture:** Keep the existing Temporal pipeline and Cypher-template layout. Harden the write path by matching the intended label and primary key only, harden entity fetches by scoping shared queries to the requested label, and make partial processing failures visible instead of silently dropping rows.

**Tech Stack:** Python 3.11, Temporal Python SDK, Neo4j Cypher templates, pytest.

---

## File Map

- Modify: `contextual_pipeline/src/cypher/write_context.cypher`
  - Scope writes by node label and the label's canonical key.
- Modify: `contextual_pipeline/src/cypher/fetch_nodes/entities.cypher`
  - Add `$label` filtering for shared entity fetches.
- Modify: `contextual_pipeline/src/activities/fetch.py`
  - Pass `$label` into node-fetch queries and fail malformed rows non-retryably.
- Modify: `contextual_pipeline/src/activities/write.py`
  - Validate Neo4j updated count equals submitted row count.
- Modify: `contextual_pipeline/src/workflows/per_type.py`
  - Validate embedding count equals requested count and write every successful row.
- Create: `contextual_pipeline/tests/test_write_contract.py`
  - Static and unit tests for label-scoped writes and write-count validation.
- Create: `contextual_pipeline/tests/test_fetch_contract.py`
  - Tests for label parameter passing and malformed-row failure.
- Create: `contextual_pipeline/tests/test_workflow_contract.py`
  - Static tests for embedding count validation and absence of filtering before writes.
- Create: `contextual_pipeline/README.md`
  - Document restart behavior, worker restart after code changes, and current index coverage.

---

## Task 1: Label-Scoped Writes

- [ ] Write failing tests asserting `write_context.cypher` does not use unscoped `OR n.id/url/name/key` matching and includes label-specific branches.
- [ ] Run the focused test and confirm it fails.
- [ ] Replace `write_context.cypher` with static label branches for all 10 processed labels.
- [ ] Add write-count validation in `write_results`.
- [ ] Run the focused tests and confirm they pass.

## Task 2: Label-Scoped Entity Fetches

- [ ] Write failing tests asserting `fetch_nodes_batch()` passes `$label` and malformed rows raise `ApplicationError(non_retryable=True)`.
- [ ] Run the focused test and confirm it fails.
- [ ] Change `fetch_nodes/entities.cypher` to filter the shared entity branch with `$label`.
- [ ] Change `fetch_nodes_batch()` to pass `{"label": label, "ids": ids}` and raise on malformed rows.
- [ ] Run the focused tests and confirm they pass.

## Task 3: Partial-Write Guards

- [ ] Write failing tests asserting the workflow validates embedding/result cardinality and does not filter successful rows before write.
- [ ] Run the focused test and confirm it fails.
- [ ] Add an embedding count check before assigning embeddings.
- [ ] Write all successful rows after embedding assignment.
- [ ] Run the focused tests and confirm they pass.

## Task 4: Operational Documentation

- [ ] Create `README.md` with commands for tests, worker restart, pilot run, full run, and known index coverage.
- [ ] Mention that worker PID 14952 is running old code and should be restarted before any new pipeline run.

## Task 5: Verification

- [ ] Run `..\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Run `..\.venv\Scripts\python.exe -m compileall -q src tests scripts`.
- [ ] Run the four Neo4j pilot verification queries again without mutating data.
- [ ] Report changed files, command results, and any remaining risks.
