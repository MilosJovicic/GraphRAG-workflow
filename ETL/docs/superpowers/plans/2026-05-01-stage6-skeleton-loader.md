# Stage 6 Skeleton Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load Stage 1–3 outputs from `./entities/` into Neo4j idempotently as a queryable graph (no embeddings yet).

**Architecture:** Three-phase loader (entities → per-page nodes with content-hash gating → global relationship MERGE) in `pipeline/stage6_load.py`. Two new CLI subcommands (`load`, `wipe`) on `pipeline/stage6_load_neo4j.py`. Two checks: pure-Python `check_load_logic.py` and end-to-end `check_load.py`.

**Tech Stack:** Python 3.13, official `neo4j==5.28.1` driver (already installed), standalone Python check scripts (no pytest — matches existing project convention).

**Repo conventions to honour:**
- Run Python via `./.venv/Scripts/python.exe` on Windows.
- Existing checks live in `./checks/`, follow the `def main() -> int` pattern (see [checks/check_neo4j_config.py](checks/check_neo4j_config.py)).
- Stage modules export their output roots as constants (see `PARSED_ROOT`, `RESOLVED_ROOT`, `ENTITIES_ROOT`) — mirror that for any new path constants.
- Project is not a git repo, so this plan does **not** include `git commit` steps. Don't add them.
- Read [docs/superpowers/specs/2026-05-01-stage6-skeleton-loader-design.md](docs/superpowers/specs/2026-05-01-stage6-skeleton-loader-design.md) before starting; the plan implements that spec verbatim.

---

## File Structure

- **Create** `pipeline/stage6_load.py` — loader logic (reader, reconciler, writer, plus top-level `load()` and `wipe()` functions).
- **Modify** `pipeline/stage6_load_neo4j.py` — add `load` and `wipe` subcommands.
- **Create** `checks/check_load_logic.py` — pure-Python tests for record bucketing, hash gating, dedup. No Neo4j required.
- **Create** `checks/check_load.py` — end-to-end check: runs `load()` against the live database, then verifies counts and structural integrity.

`pipeline/stage6_load.py` is split internally into clearly-named functions per phase so each is independently testable. If it grows past ~400 lines, split out the Cypher templates into a sibling `pipeline/stage6_cypher.py` — but don't pre-emptively split.

---

## Relationship → endpoint label mapping

The loader needs to know each relationship type's source and target labels because the JSONL records only carry `source_id` / `target_id` (no labels). This table is the single source of truth — copy it verbatim into `pipeline/stage6_load.py`:

| Type | Source | Source key | Target | Target key |
|---|---|---|---|---|
| `HAS_SECTION` | `Page` | `url` | `Section` | `id` |
| `HAS_SUBSECTION` | `Section` | `id` | `Section` | `id` |
| `CONTAINS_CODE` | `Section` | `id` | `CodeBlock` | `id` |
| `CONTAINS_TABLE_ROW` | `Section` | `id` | `TableRow` | `id` |
| `HAS_CALLOUT` | `Section` | `id` | `Callout` | `id` |
| `LINKS_TO` | `Section` | `id` | `Section` | `id` |
| `LINKS_TO_PAGE` | `Section` | `id` | `Page` | `url` |
| `NAVIGATES_TO` | `Section` | `id` | `Section` | `id` |
| `EQUIVALENT_TO` | `CodeBlock` | `id` | `CodeBlock` | `id` |
| `MENTIONS` | `Section` | `id` | *entity* (dynamic) | `id` (composite) |
| `DEFINES` | `Section` or `TableRow` | `id` | *entity* (dynamic) | `id` (composite) |

For `MENTIONS`/`DEFINES`: the **target label** is parsed from the `target_id` prefix (`"Tool:Read"` → label `Tool`). The **source label** for `DEFINES` is parsed from the `source_id` shape: a TableRow source_id is a 16-char hex string (no `#`), a Section source_id contains `#`. Use that distinction.

Entity nodes are stored with their natural key as the constraint property (`Tool.name`, `SettingKey.key`, `Hook.name`, `PermissionMode.name`, `Provider.name`, `MessageType.name`) **and** also with a redundant `id` property holding the composite `"Label:name"`. The loader writes `id` so MENTIONS/DEFINES MERGEs can match entity targets uniformly by `id` while uniqueness is still enforced by the existing label-specific constraints.

---

### Task 1: Pure-Python helpers (reader, hash gate, relationship dedup)

**Files:**
- Create: `pipeline/stage6_load.py`
- Create: `checks/check_load_logic.py`

- [ ] **Step 1: Create the empty module**

Create `pipeline/stage6_load.py` with only:

```python
"""Stage 6 - load Stage 1-3 JSONL into Neo4j.

See docs/superpowers/specs/2026-05-01-stage6-skeleton-loader-design.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .stage1_parse import REPO_ROOT
from .stage3_entities import ENTITIES_ROOT


SKIPPED_KINDS = {"link", "unresolved_link"}
NODE_KINDS = {"page", "section", "codeblock", "tablerow", "callout"}
ENTITY_KIND = "entity"
RELATIONSHIP_KIND = "relationship"
```

- [ ] **Step 2: Write the failing check for `read_entity_catalog`**

Create `checks/check_load_logic.py`:

```python
"""Pure-Python tests for Stage 6 loader helpers (no Neo4j required)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage6_load import (  # noqa: E402
    decide_load_action,
    dedupe_relationships,
    read_entity_catalog,
    read_page_records,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def test_read_entity_catalog() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_entities.jsonl"
        _write_jsonl(
            path,
            [
                {"kind": "entity", "id": "Tool:Read", "label": "Tool", "name": "Read"},
                {"kind": "entity", "id": "Hook:SessionStart", "label": "Hook", "name": "SessionStart"},
            ],
        )
        entities = read_entity_catalog(path)
        assert len(entities) == 2
        assert {e["id"] for e in entities} == {"Tool:Read", "Hook:SessionStart"}


def main() -> int:
    test_read_entity_catalog()
    print("check_load_logic.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run check to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `ImportError: cannot import name 'read_entity_catalog' from 'pipeline.stage6_load'`.

- [ ] **Step 4: Implement `read_entity_catalog`**

Add to `pipeline/stage6_load.py`:

```python
def read_entity_catalog(path: Path) -> list[dict]:
    """Read _entities.jsonl. Returns entity records only; raises if any record
    has kind != 'entity'."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if record.get("kind") != ENTITY_KIND:
            raise ValueError(f"{path}: expected entity record, got {record.get('kind')}")
    return records
```

- [ ] **Step 5: Run check to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `check_load_logic.py OK`.

- [ ] **Step 6: Add the failing check for `read_page_records`**

Append to `checks/check_load_logic.py` before `main()`:

```python
def test_read_page_records_buckets_and_skips() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.jsonl"
        _write_jsonl(
            path,
            [
                {"kind": "page", "url": "u", "path": "p", "title": "T", "content_hash": "h"},
                {"kind": "section", "id": "u#a", "page_url": "u", "anchor": "a", "level": 2, "position": 0},
                {"kind": "codeblock", "id": "cb1", "section_id": "u#a", "position": 0, "language": "py", "text": "x"},
                {"kind": "tablerow", "id": "tr1", "section_id": "u#a", "position": 0, "table_position": 0, "headers": ["h"], "cells": ["c"]},
                {"kind": "callout", "id": "co1", "section_id": "u#a", "position": 0, "callout_kind": "note", "text": "t"},
                {"kind": "link", "section_id": "u#a", "href": "x", "text": "y", "is_internal": False, "is_anchor_only": False},
                {"kind": "unresolved_link", "section_id": "u#a", "href": "/en/missing"},
                {"kind": "relationship", "id": "r1", "type": "HAS_SECTION", "source_id": "u", "target_id": "u#a"},
            ],
        )
        page, by_kind = read_page_records(path)
        assert page["url"] == "u"
        assert [r["id"] for r in by_kind["section"]] == ["u#a"]
        assert [r["id"] for r in by_kind["codeblock"]] == ["cb1"]
        assert [r["id"] for r in by_kind["tablerow"]] == ["tr1"]
        assert [r["id"] for r in by_kind["callout"]] == ["co1"]
        assert [r["id"] for r in by_kind["relationship"]] == ["r1"]
        assert "link" not in by_kind
        assert "unresolved_link" not in by_kind


def test_read_page_records_rejects_unknown_kind() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.jsonl"
        _write_jsonl(path, [{"kind": "wat", "id": "x"}])
        try:
            read_page_records(path)
        except ValueError as exc:
            assert "wat" in str(exc)
            assert str(path) in str(exc)
        else:
            raise AssertionError("unknown kind must raise")
```

Wire them into `main()`:

```python
def main() -> int:
    test_read_entity_catalog()
    test_read_page_records_buckets_and_skips()
    test_read_page_records_rejects_unknown_kind()
    print("check_load_logic.py OK")
    return 0
```

- [ ] **Step 7: Run check to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `ImportError: cannot import name 'read_page_records'`.

- [ ] **Step 8: Implement `read_page_records`**

Add to `pipeline/stage6_load.py`:

```python
def read_page_records(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Read one page JSONL. Returns (page_record, records_by_kind).

    Skips link and unresolved_link records (they are realised as Stage 2
    relationships or Section.external_links). Raises ValueError for unknown
    kinds.
    """
    by_kind: dict[str, list[dict]] = defaultdict(list)
    page: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind in SKIPPED_KINDS:
                continue
            if kind == "page":
                if page is not None:
                    raise ValueError(f"{path}:{line_no}: more than one page record in file")
                page = record
                continue
            if kind in NODE_KINDS or kind == RELATIONSHIP_KIND:
                by_kind[kind].append(record)
                continue
            raise ValueError(f"{path}:{line_no}: unknown record kind {kind!r}")
    if page is None:
        raise ValueError(f"{path}: no page record found")
    return page, dict(by_kind)
```

- [ ] **Step 9: Run check to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `check_load_logic.py OK`.

- [ ] **Step 10: Add the failing check for `decide_load_action`**

Append before `main()`:

```python
def test_decide_load_action() -> None:
    assert decide_load_action("h1", None) == "merge_all"
    assert decide_load_action("h1", "h1") == "skip_nodes"
    assert decide_load_action("h2", "h1") == "replace"
```

Add to `main()`:

```python
    test_decide_load_action()
```

- [ ] **Step 11: Run check to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `ImportError: cannot import name 'decide_load_action'`.

- [ ] **Step 12: Implement `decide_load_action`**

Add to `pipeline/stage6_load.py`:

```python
def decide_load_action(incoming_hash: str, stored_hash: str | None) -> str:
    """Return one of: 'merge_all' (page absent), 'skip_nodes' (hash matches),
    'replace' (hash differs)."""
    if stored_hash is None:
        return "merge_all"
    if stored_hash == incoming_hash:
        return "skip_nodes"
    return "replace"
```

- [ ] **Step 13: Run check to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `check_load_logic.py OK`.

- [ ] **Step 14: Add the failing check for `dedupe_relationships`**

Append before `main()`:

```python
def test_dedupe_relationships() -> None:
    rels = [
        {"kind": "relationship", "id": "a1", "type": "LINKS_TO", "source_id": "s", "target_id": "t", "href": "/en/a"},
        {"kind": "relationship", "id": "a2", "type": "LINKS_TO", "source_id": "s", "target_id": "t", "href": "/en/b"},
        {"kind": "relationship", "id": "b1", "type": "HAS_SECTION", "source_id": "u", "target_id": "u#a"},
        {"kind": "relationship", "id": "b1", "type": "HAS_SECTION", "source_id": "u", "target_id": "u#a"},
    ]
    deduped = dedupe_relationships(rels)
    keys = [(r["type"], r["source_id"], r["target_id"]) for r in deduped]
    assert keys == [
        ("LINKS_TO", "s", "t"),
        ("HAS_SECTION", "u", "u#a"),
    ]
    # The retained LINKS_TO record is the last-seen one (so writers see the
    # most recent href/text).
    assert deduped[0]["href"] == "/en/b"
```

Add to `main()`:

```python
    test_dedupe_relationships()
```

- [ ] **Step 15: Run check to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `ImportError: cannot import name 'dedupe_relationships'`.

- [ ] **Step 16: Implement `dedupe_relationships`**

Add to `pipeline/stage6_load.py`:

```python
def dedupe_relationships(records: Iterable[dict]) -> list[dict]:
    """Dedupe by (source_id, type, target_id), keeping the last record seen.
    Order of first occurrence is preserved.
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for record in records:
        key = (record["source_id"], record["type"], record["target_id"])
        if key not in by_key:
            order.append(key)
        by_key[key] = record
    return [by_key[key] for key in order]
```

- [ ] **Step 17: Run check to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load_logic.py`

Expected: `check_load_logic.py OK`.

---

### Task 2: Schema check, page lookup, wipe (live Neo4j)

**Files:**
- Modify: `pipeline/stage6_load.py`
- Create: `checks/check_load.py`

This task introduces the live-database check script. It needs `NEO4J_PASSWORD` set in the environment.

- [ ] **Step 1: Create `checks/check_load.py` with a fixture-cleaning helper**

Create `checks/check_load.py`:

```python
"""End-to-end check: run Stage 6 loader against live Neo4j and verify."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.neo4j_connector import Neo4jConnector, config_from_env  # noqa: E402
from pipeline.stage6_load import (  # noqa: E402
    LOADER_LABELS,
    check_schema_applied,
    get_stored_hash,
    wipe,
)


def test_schema_applied(connector: Neo4jConnector) -> None:
    check_schema_applied(connector)


def test_wipe_clears_loader_labels(connector: Neo4jConnector) -> None:
    # Seed one node per loader-owned label.
    for label in LOADER_LABELS:
        connector.run_statement(f"CREATE (n:{label} {{__test_seed: true}})")
    wipe(connector)
    for label in LOADER_LABELS:
        records, _, _ = connector._driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS c",
            database_=connector.config.database,
        )
        assert records[0]["c"] == 0, f"wipe left {label} nodes"


def test_get_stored_hash_missing(connector: Neo4jConnector) -> None:
    assert get_stored_hash(connector, "https://nope.example/x") is None


def main() -> int:
    config = config_from_env()
    with Neo4jConnector(config) as connector:
        test_schema_applied(connector)
        test_wipe_clears_loader_labels(connector)
        test_get_stored_hash_missing(connector)
    print("check_load.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run check to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'LOADER_LABELS' from 'pipeline.stage6_load'`.

- [ ] **Step 3: Implement `LOADER_LABELS`, `check_schema_applied`, `get_stored_hash`, `wipe`**

Add to `pipeline/stage6_load.py`:

```python
LOADER_LABELS = (
    "Page",
    "Section",
    "Chunk",
    "CodeBlock",
    "TableRow",
    "Callout",
    "Tool",
    "SettingKey",
    "Provider",
    "MessageType",
    "Hook",
    "PermissionMode",
)


class SchemaNotAppliedError(RuntimeError):
    pass


def check_schema_applied(connector) -> None:
    """Raise SchemaNotAppliedError if the page_url constraint is missing."""
    records, _, _ = connector._driver.execute_query(
        "SHOW CONSTRAINTS YIELD name WHERE name = 'page_url' RETURN count(*) AS c",
        database_=connector.config.database,
    )
    if records[0]["c"] == 0:
        raise SchemaNotAppliedError(
            "schema not applied; run 'python -m pipeline.stage6_load_neo4j schema' first"
        )


def get_stored_hash(connector, page_url: str) -> str | None:
    records, _, _ = connector._driver.execute_query(
        "MATCH (p:Page {url: $url}) RETURN p.content_hash AS hash",
        {"url": page_url},
        database_=connector.config.database,
    )
    if not records:
        return None
    return records[0]["hash"]


def wipe(connector) -> None:
    """DETACH DELETE all loader-owned label nodes. Schema and indexes remain."""
    for label in LOADER_LABELS:
        connector.run_statement(f"MATCH (n:{label}) DETACH DELETE n")
```

- [ ] **Step 4: Run check to verify it passes**

Make sure `NEO4J_PASSWORD` is set in env, then run:

```bash
./.venv/Scripts/python.exe checks/check_load.py
```

Expected: `check_load.py OK`.

If this fails because the schema isn't applied yet, run `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j schema` first.

---

### Task 3: Phase 1 — entity catalog MERGE

**Files:**
- Modify: `pipeline/stage6_load.py`
- Modify: `checks/check_load.py`

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def test_merge_entities_loads_catalog(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_entities

    wipe(connector)
    entities = [
        {"kind": "entity", "id": "Tool:Read", "label": "Tool", "name": "Read"},
        {"kind": "entity", "id": "Tool:Edit", "label": "Tool", "name": "Edit", "description": "Edit files"},
        {"kind": "entity", "id": "Hook:SessionStart", "label": "Hook", "name": "SessionStart"},
        {"kind": "entity", "id": "SettingKey:permissions.allow", "label": "SettingKey", "key": "permissions.allow"},
    ]
    merge_entities(connector, entities, batch_size=10)

    records, _, _ = connector._driver.execute_query(
        "MATCH (t:Tool {name: 'Edit'}) RETURN t.id AS id, t.description AS d",
        database_=connector.config.database,
    )
    assert records[0]["id"] == "Tool:Edit"
    assert records[0]["d"] == "Edit files"

    records, _, _ = connector._driver.execute_query(
        "MATCH (s:SettingKey {key: 'permissions.allow'}) RETURN s.id AS id",
        database_=connector.config.database,
    )
    assert records[0]["id"] == "SettingKey:permissions.allow"

    # Idempotent re-run.
    merge_entities(connector, entities, batch_size=10)
    records, _, _ = connector._driver.execute_query(
        "MATCH (t:Tool) RETURN count(t) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == 2
```

Add to `main()`:

```python
        test_merge_entities_loads_catalog(connector)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'merge_entities'`.

- [ ] **Step 3: Implement `merge_entities`**

Add to `pipeline/stage6_load.py`:

```python
ENTITY_KEY_PROP = {
    "Tool": "name",
    "SettingKey": "key",
    "Hook": "name",
    "PermissionMode": "name",
    "Provider": "name",
    "MessageType": "name",
}


def _batched(items: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def merge_entities(connector, entities: list[dict], batch_size: int) -> None:
    """Phase 1: MERGE entity nodes by their natural key. Writes the composite
    `id` property too so MENTIONS/DEFINES targets can be matched by id later.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        label = entity["label"]
        if label not in ENTITY_KEY_PROP:
            raise ValueError(f"unknown entity label {label!r}")
        by_label[label].append(entity)

    for label, label_entities in by_label.items():
        key_prop = ENTITY_KEY_PROP[label]
        cypher = (
            f"UNWIND $batch AS row "
            f"MERGE (n:{label} {{{key_prop}: row.{key_prop}}}) "
            f"SET n.id = row.id, n += row.props"
        )
        for batch in _batched(label_entities, batch_size):
            payload = []
            for entity in batch:
                props = {k: v for k, v in entity.items() if k not in {"kind", "id", "label", key_prop}}
                payload.append({key_prop: entity[key_prop], "id": entity["id"], "props": props})
            connector._driver.execute_query(
                cypher,
                {"batch": payload},
                database_=connector.config.database,
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

---

### Task 4: Phase 2 — per-page node MERGE (page-absent branch)

**Files:**
- Modify: `pipeline/stage6_load.py`
- Modify: `checks/check_load.py`

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def test_merge_page_subtree_fresh(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_page_subtree

    wipe(connector)
    page = {"kind": "page", "url": "https://example/p", "path": "en/x.md", "title": "X",
            "description": "d", "description_source": "llms.txt", "content_hash": "h1",
            "volatile": False}
    by_kind = {
        "section": [
            {"kind": "section", "id": "https://example/p#s1", "page_url": "https://example/p",
             "anchor": "s1", "breadcrumb": "X > S1", "level": 2, "position": 0,
             "text": "hello", "synthesized": False, "parent_section_id": None,
             "external_links": ["https://e/x"]},
        ],
        "codeblock": [
            {"kind": "codeblock", "id": "cb1", "section_id": "https://example/p#s1",
             "position": 0, "language": "python", "text": "x = 1",
             "preceding_paragraph": "before", "in_codegroup": False, "codegroup_id": None},
        ],
        "tablerow": [
            {"kind": "tablerow", "id": "tr1", "section_id": "https://example/p#s1",
             "position": 0, "table_position": 0, "headers": ["k", "v"], "cells": ["a", "b"]},
        ],
        "callout": [
            {"kind": "callout", "id": "co1", "section_id": "https://example/p#s1",
             "position": 0, "callout_kind": "note", "text": "note text"},
        ],
    }
    merge_page_subtree(connector, page, by_kind, batch_size=10)

    counts = {}
    for label in ("Page", "Section", "CodeBlock", "TableRow", "Callout"):
        records, _, _ = connector._driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS c",
            database_=connector.config.database,
        )
        counts[label] = records[0]["c"]
    assert counts == {"Page": 1, "Section": 1, "CodeBlock": 1, "TableRow": 1, "Callout": 1}

    records, _, _ = connector._driver.execute_query(
        "MATCH (s:Section {id: 'https://example/p#s1'}) "
        "RETURN s.breadcrumb AS bc, s.external_links AS ext",
        database_=connector.config.database,
    )
    assert records[0]["bc"] == "X > S1"
    assert records[0]["ext"] == ["https://e/x"]
```

Add to `main()`:

```python
        test_merge_page_subtree_fresh(connector)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'merge_page_subtree'`.

- [ ] **Step 3: Implement `merge_page_subtree`**

Add to `pipeline/stage6_load.py`:

```python
_PAGE_PROPS = ("url", "path", "title", "description", "description_source", "content_hash", "volatile")
_SECTION_PROPS = ("id", "page_url", "anchor", "breadcrumb", "level", "position", "text",
                  "synthesized", "parent_section_id", "external_links")
_CODEBLOCK_PROPS = ("id", "section_id", "position", "language", "text",
                    "preceding_paragraph", "in_codegroup", "codegroup_id")
_TABLEROW_PROPS = ("id", "section_id", "position", "table_position", "headers", "cells")
_CALLOUT_PROPS = ("id", "section_id", "position", "callout_kind", "text")


def _project(record: dict, props: tuple[str, ...]) -> dict:
    return {p: record.get(p) for p in props}


def _merge_batch(connector, label: str, key_prop: str, props: tuple[str, ...],
                 records: list[dict], batch_size: int) -> None:
    cypher = (
        f"UNWIND $batch AS row "
        f"MERGE (n:{label} {{{key_prop}: row.{key_prop}}}) "
        f"SET n += row"
    )
    for batch in _batched(records, batch_size):
        payload = [_project(record, props) for record in batch]
        connector._driver.execute_query(
            cypher,
            {"batch": payload},
            database_=connector.config.database,
        )


def merge_page_subtree(connector, page: dict, by_kind: dict[str, list[dict]],
                       batch_size: int) -> None:
    """Phase 2 happy path: MERGE the Page and all its child nodes."""
    _merge_batch(connector, "Page", "url", _PAGE_PROPS, [page], batch_size)
    if by_kind.get("section"):
        _merge_batch(connector, "Section", "id", _SECTION_PROPS, by_kind["section"], batch_size)
    if by_kind.get("codeblock"):
        _merge_batch(connector, "CodeBlock", "id", _CODEBLOCK_PROPS, by_kind["codeblock"], batch_size)
    if by_kind.get("tablerow"):
        _merge_batch(connector, "TableRow", "id", _TABLEROW_PROPS, by_kind["tablerow"], batch_size)
    if by_kind.get("callout"):
        _merge_batch(connector, "Callout", "id", _CALLOUT_PROPS, by_kind["callout"], batch_size)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

---

### Task 5: Reconciler — DETACH DELETE for changed-hash branch

**Files:**
- Modify: `pipeline/stage6_load.py`
- Modify: `checks/check_load.py`

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def test_detach_delete_page_descendants(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import detach_delete_page, merge_page_subtree

    wipe(connector)
    page = {"kind": "page", "url": "https://example/p", "path": "en/x.md", "title": "X",
            "description": "d", "description_source": "llms.txt", "content_hash": "h1",
            "volatile": False}
    by_kind = {
        "section": [
            {"kind": "section", "id": "https://example/p#s1", "page_url": "https://example/p",
             "anchor": "s1", "breadcrumb": "X > S1", "level": 2, "position": 0, "text": "x",
             "synthesized": False, "parent_section_id": None, "external_links": []},
        ],
        "codeblock": [
            {"kind": "codeblock", "id": "cb1", "section_id": "https://example/p#s1",
             "position": 0, "language": "py", "text": "x", "preceding_paragraph": "",
             "in_codegroup": False, "codegroup_id": None},
        ],
    }
    merge_page_subtree(connector, page, by_kind, batch_size=10)

    # Manually link Section->CodeBlock so detach_delete_page has something to walk.
    connector.run_statement(
        "MATCH (s:Section {id: 'https://example/p#s1'}), (b:CodeBlock {id: 'cb1'}) "
        "MERGE (s)-[:CONTAINS_CODE]->(b)"
    )
    connector.run_statement(
        "MATCH (p:Page {url: 'https://example/p'}), (s:Section {id: 'https://example/p#s1'}) "
        "MERGE (p)-[:HAS_SECTION]->(s)"
    )

    detach_delete_page(connector, "https://example/p")

    for label in ("Page", "Section", "CodeBlock"):
        records, _, _ = connector._driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS c",
            database_=connector.config.database,
        )
        assert records[0]["c"] == 0, f"detach_delete left {label} nodes"


def test_detach_delete_page_missing_is_noop(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import detach_delete_page

    wipe(connector)
    detach_delete_page(connector, "https://example/missing")
```

Add to `main()`:

```python
        test_detach_delete_page_descendants(connector)
        test_detach_delete_page_missing_is_noop(connector)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'detach_delete_page'`.

- [ ] **Step 3: Implement `detach_delete_page`**

Add to `pipeline/stage6_load.py`:

```python
def detach_delete_page(connector, page_url: str) -> None:
    """Delete a Page and all descendant nodes (Sections + their children).

    Children are deleted first so DETACH DELETE on the Section doesn't orphan
    them (the only edge from CodeBlock/TableRow/Callout to the rest of the
    graph is the CONTAINS_*/HAS_CALLOUT edge from its parent Section).
    Entity nodes are never touched.
    """
    queries = (
        # Children of any Section under this Page.
        "MATCH (:Page {url: $url})-[:HAS_SECTION]->(s:Section) "
        "OPTIONAL MATCH (s)-[:CONTAINS_CODE|CONTAINS_TABLE_ROW|HAS_CALLOUT]->(child) "
        "DETACH DELETE child",
        # Sections (including nested HAS_SUBSECTION descendants).
        "MATCH (:Page {url: $url})-[:HAS_SECTION]->(s:Section) "
        "OPTIONAL MATCH (s)-[:HAS_SUBSECTION*0..]->(d:Section) "
        "DETACH DELETE d",
        # The Page itself.
        "MATCH (p:Page {url: $url}) DETACH DELETE p",
    )
    for cypher in queries:
        connector._driver.execute_query(cypher, {"url": page_url}, database_=connector.config.database)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

---

### Task 6: Phase 3 — global relationship MERGE

**Files:**
- Modify: `pipeline/stage6_load.py`
- Modify: `checks/check_load.py`

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def test_merge_relationships_all_types(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_entities, merge_page_subtree, merge_relationships

    wipe(connector)
    # Two pages, two sections each, one codeblock under one section.
    pages = [
        {"kind": "page", "url": "https://e/a", "path": "en/a.md", "title": "A",
         "description": "", "description_source": "missing", "content_hash": "ha", "volatile": False},
        {"kind": "page", "url": "https://e/b", "path": "en/b.md", "title": "B",
         "description": "", "description_source": "missing", "content_hash": "hb", "volatile": False},
    ]
    sections = [
        {"kind": "section", "id": "https://e/a#x", "page_url": "https://e/a",
         "anchor": "x", "breadcrumb": "A > x", "level": 2, "position": 0, "text": "",
         "synthesized": False, "parent_section_id": None, "external_links": []},
        {"kind": "section", "id": "https://e/a#y", "page_url": "https://e/a",
         "anchor": "y", "breadcrumb": "A > y", "level": 3, "position": 1, "text": "",
         "synthesized": False, "parent_section_id": "https://e/a#x", "external_links": []},
        {"kind": "section", "id": "https://e/b#z", "page_url": "https://e/b",
         "anchor": "z", "breadcrumb": "B > z", "level": 2, "position": 0, "text": "",
         "synthesized": False, "parent_section_id": None, "external_links": []},
    ]
    codeblocks = [
        {"kind": "codeblock", "id": "cb1", "section_id": "https://e/a#x", "position": 0,
         "language": "py", "text": "x", "preceding_paragraph": "", "in_codegroup": True, "codegroup_id": "g1"},
        {"kind": "codeblock", "id": "cb2", "section_id": "https://e/a#x", "position": 1,
         "language": "ts", "text": "y", "preceding_paragraph": "", "in_codegroup": True, "codegroup_id": "g1"},
    ]
    tablerows = [
        {"kind": "tablerow", "id": "tr1", "section_id": "https://e/a#x", "position": 0,
         "table_position": 0, "headers": ["tool"], "cells": ["Read"]},
    ]
    callouts: list[dict] = []
    merge_page_subtree(connector, pages[0], {"section": [sections[0], sections[1]],
                                              "codeblock": codeblocks, "tablerow": tablerows,
                                              "callout": callouts}, batch_size=10)
    merge_page_subtree(connector, pages[1], {"section": [sections[2]], "codeblock": [],
                                              "tablerow": [], "callout": []}, batch_size=10)
    merge_entities(connector, [
        {"kind": "entity", "id": "Tool:Read", "label": "Tool", "name": "Read"},
    ], batch_size=10)

    relationships = [
        {"kind": "relationship", "id": "r1", "type": "HAS_SECTION", "source_id": "https://e/a", "target_id": "https://e/a#x"},
        {"kind": "relationship", "id": "r2", "type": "HAS_SECTION", "source_id": "https://e/a", "target_id": "https://e/a#y"},
        {"kind": "relationship", "id": "r3", "type": "HAS_SUBSECTION", "source_id": "https://e/a#x", "target_id": "https://e/a#y"},
        {"kind": "relationship", "id": "r4", "type": "HAS_SECTION", "source_id": "https://e/b", "target_id": "https://e/b#z"},
        {"kind": "relationship", "id": "r5", "type": "CONTAINS_CODE", "source_id": "https://e/a#x", "target_id": "cb1"},
        {"kind": "relationship", "id": "r6", "type": "CONTAINS_CODE", "source_id": "https://e/a#x", "target_id": "cb2"},
        {"kind": "relationship", "id": "r7", "type": "CONTAINS_TABLE_ROW", "source_id": "https://e/a#x", "target_id": "tr1"},
        {"kind": "relationship", "id": "r8", "type": "EQUIVALENT_TO", "source_id": "cb1", "target_id": "cb2"},
        {"kind": "relationship", "id": "r9", "type": "LINKS_TO", "source_id": "https://e/a#x", "target_id": "https://e/b#z", "href": "/en/b#z", "text": "see B"},
        {"kind": "relationship", "id": "r10", "type": "LINKS_TO_PAGE", "source_id": "https://e/a#x", "target_id": "https://e/b", "href": "/en/b"},
        {"kind": "relationship", "id": "r11", "type": "NAVIGATES_TO", "source_id": "https://e/b#z", "target_id": "https://e/a#x"},
        {"kind": "relationship", "id": "r12", "type": "MENTIONS", "source_id": "https://e/a#x", "target_id": "Tool:Read", "evidence": "Read"},
        {"kind": "relationship", "id": "r13", "type": "DEFINES", "source_id": "tr1", "target_id": "Tool:Read"},
    ]
    merge_relationships(connector, relationships, batch_size=10)

    expected = {
        "HAS_SECTION": 3, "HAS_SUBSECTION": 1, "CONTAINS_CODE": 2, "CONTAINS_TABLE_ROW": 1,
        "EQUIVALENT_TO": 1, "LINKS_TO": 1, "LINKS_TO_PAGE": 1, "NAVIGATES_TO": 1,
        "MENTIONS": 1, "DEFINES": 1,
    }
    for rel_type, count in expected.items():
        records, _, _ = connector._driver.execute_query(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c",
            database_=connector.config.database,
        )
        assert records[0]["c"] == count, f"{rel_type}: expected {count}, got {records[0]['c']}"

    # Idempotent re-run.
    merge_relationships(connector, relationships, batch_size=10)
    records, _, _ = connector._driver.execute_query(
        "MATCH ()-[r]->() RETURN count(r) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == sum(expected.values())
```

Add to `main()`:

```python
        test_merge_relationships_all_types(connector)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'merge_relationships'`.

- [ ] **Step 3: Implement `merge_relationships`**

Add to `pipeline/stage6_load.py`:

```python
# (source_label, source_key_prop, target_label, target_key_prop)
# None means dispatch dynamically (entity targets, or DEFINES sources).
_REL_DISPATCH: dict[str, tuple[str, str, str | None, str | None]] = {
    "HAS_SECTION":         ("Page",      "url", "Section",   "id"),
    "HAS_SUBSECTION":      ("Section",   "id",  "Section",   "id"),
    "CONTAINS_CODE":       ("Section",   "id",  "CodeBlock", "id"),
    "CONTAINS_TABLE_ROW":  ("Section",   "id",  "TableRow",  "id"),
    "HAS_CALLOUT":         ("Section",   "id",  "Callout",   "id"),
    "LINKS_TO":            ("Section",   "id",  "Section",   "id"),
    "LINKS_TO_PAGE":       ("Section",   "id",  "Page",      "url"),
    "NAVIGATES_TO":        ("Section",   "id",  "Section",   "id"),
    "EQUIVALENT_TO":       ("CodeBlock", "id",  "CodeBlock", "id"),
    # MENTIONS and DEFINES use entity ids; targets are matched by id across labels.
    # DEFINES sources may be Section or TableRow; we match by id across both.
    "MENTIONS":            ("Section",   "id",  None,        None),
    "DEFINES":             (None,        "id",  None,        None),
}

_REL_PROP_KEYS = ("id", "href", "text", "evidence")


def _entity_label_from_id(entity_id: str) -> str:
    """'Tool:Read' -> 'Tool'."""
    return entity_id.split(":", 1)[0]


def _rel_props(record: dict) -> dict:
    return {k: record[k] for k in _REL_PROP_KEYS if k in record}


def merge_relationships(connector, records: list[dict], batch_size: int) -> None:
    """Phase 3: MERGE all relationships globally, dispatched by type and
    (for entity edges) by parsed target label.
    """
    deduped = dedupe_relationships(records)

    # Group by (rel_type, source_label, target_label) so each group can run with one Cypher.
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in deduped:
        rel_type = record["type"]
        if rel_type not in _REL_DISPATCH:
            raise ValueError(f"unknown relationship type {rel_type!r}")
        src_label, src_key, tgt_label, tgt_key = _REL_DISPATCH[rel_type]

        if src_label is None:
            # DEFINES: source can be Section or TableRow. Distinguish by source_id shape:
            # Section ids contain '#', TableRow/CodeBlock/Callout ids do not.
            src_label = "Section" if "#" in record["source_id"] else "TableRow"
        if tgt_label is None:
            tgt_label = _entity_label_from_id(record["target_id"])
            if tgt_label not in ENTITY_KEY_PROP:
                raise ValueError(f"unknown entity target label {tgt_label!r} in {record}")

        groups[(rel_type, src_label, tgt_label)].append(record)

    for (rel_type, src_label, tgt_label), group_records in groups.items():
        src_key = _REL_DISPATCH[rel_type][1]
        # Target key is `id` for entities (we wrote it in merge_entities) and the
        # static key from _REL_DISPATCH for everything else.
        tgt_key = _REL_DISPATCH[rel_type][3] or "id"

        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (a:{src_label} {{{src_key}: row.source_id}}) "
            f"MATCH (b:{tgt_label} {{{tgt_key}: row.target_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r += row.props"
        )
        for batch in _batched(group_records, batch_size):
            payload = [
                {"source_id": record["source_id"], "target_id": record["target_id"], "props": _rel_props(record)}
                for record in batch
            ]
            connector._driver.execute_query(
                cypher,
                {"batch": payload},
                database_=connector.config.database,
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

---

### Task 7: Top-level `load()` function

**Files:**
- Modify: `pipeline/stage6_load.py`
- Modify: `checks/check_load.py`

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def test_load_full_corpus(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import ENTITIES_ROOT, load

    wipe(connector)
    counts = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=False)
    assert counts["pages_loaded"] == 123, counts
    assert counts["pages_skipped"] == 0
    # Sanity-check a few label totals (exact values can drift a bit if the
    # corpus changes; the tight check_load.py validation is a separate task).
    assert counts["nodes"]["Page"] == 123
    assert counts["nodes"]["Section"] >= 2700
    assert counts["nodes"]["Tool"] == 35
    assert counts["nodes"]["Hook"] == 29
    # Idempotent re-run: second load should skip every page.
    counts2 = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=False)
    assert counts2["pages_skipped"] == 123
    assert counts2["pages_loaded"] == 0


def test_load_dry_run(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import ENTITIES_ROOT, load

    wipe(connector)
    counts = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=True)
    # dry_run reports planned counts but writes nothing.
    assert counts["pages_loaded"] == 123
    records, _, _ = connector._driver.execute_query(
        "MATCH (p:Page) RETURN count(p) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == 0
```

Add to `main()`:

```python
        test_load_full_corpus(connector)
        test_load_dry_run(connector)
```

- [ ] **Step 2: Run to verify it fails**

Make sure `./entities/` exists: `./.venv/Scripts/python.exe -m pipeline.stage1_parse && ./.venv/Scripts/python.exe -m pipeline.stage2_resolve && ./.venv/Scripts/python.exe -m pipeline.stage3_entities`.

Then: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `ImportError: cannot import name 'load'`.

- [ ] **Step 3: Implement `load`**

Add to `pipeline/stage6_load.py`:

```python
def load(connector, in_dir: Path, batch_size: int, dry_run: bool) -> dict:
    """Top-level Stage 6 loader. Returns a counts dict for CLI output.

    On dry_run, the planned operation counts are reported but no Cypher executes.
    Idempotent: pages whose content_hash matches the stored value are skipped.
    """
    if not in_dir.exists():
        raise FileNotFoundError(
            f"{in_dir} not found; run 'python -m pipeline.stage3_entities' first"
        )

    if not dry_run:
        check_schema_applied(connector)

    entities_path = in_dir / "_entities.jsonl"
    page_files = sorted(p for p in in_dir.glob("*.jsonl") if p.name != "_entities.jsonl")
    if not page_files:
        raise FileNotFoundError(f"no page JSONL files in {in_dir}")

    entities = read_entity_catalog(entities_path) if entities_path.exists() else []

    counts = {
        "pages_loaded": 0,
        "pages_skipped": 0,
        "pages_replaced": 0,
        "nodes": defaultdict(int),
        "relationships": defaultdict(int),
    }

    all_relationships: list[dict] = []
    page_buckets: list[tuple[dict, dict[str, list[dict]]]] = []

    for page_file in page_files:
        page, by_kind = read_page_records(page_file)
        page_buckets.append((page, by_kind))
        for rel in by_kind.get("relationship", []):
            all_relationships.append(rel)

    if dry_run:
        for page, by_kind in page_buckets:
            counts["pages_loaded"] += 1
            counts["nodes"]["Page"] += 1
            for kind in NODE_KINDS - {"page"}:
                label = {"section": "Section", "codeblock": "CodeBlock",
                         "tablerow": "TableRow", "callout": "Callout"}[kind]
                counts["nodes"][label] += len(by_kind.get(kind, []))
        for entity in entities:
            counts["nodes"][entity["label"]] += 1
        for rel in dedupe_relationships(all_relationships):
            counts["relationships"][rel["type"]] += 1
        return _finalize_counts(counts)

    # Phase 1: entities.
    if entities:
        merge_entities(connector, entities, batch_size)
    for entity in entities:
        counts["nodes"][entity["label"]] += 1

    # Phase 2: per-page nodes with hash gating.
    for page, by_kind in page_buckets:
        action = decide_load_action(page["content_hash"], get_stored_hash(connector, page["url"]))
        if action == "skip_nodes":
            counts["pages_skipped"] += 1
        elif action == "replace":
            detach_delete_page(connector, page["url"])
            merge_page_subtree(connector, page, by_kind, batch_size)
            counts["pages_replaced"] += 1
            counts["pages_loaded"] += 1
        else:  # merge_all
            merge_page_subtree(connector, page, by_kind, batch_size)
            counts["pages_loaded"] += 1

        counts["nodes"]["Page"] += 1
        for kind in ("section", "codeblock", "tablerow", "callout"):
            label = {"section": "Section", "codeblock": "CodeBlock",
                     "tablerow": "TableRow", "callout": "Callout"}[kind]
            counts["nodes"][label] += len(by_kind.get(kind, []))

    # Phase 3: relationships.
    if all_relationships:
        merge_relationships(connector, all_relationships, batch_size)
        for rel in dedupe_relationships(all_relationships):
            counts["relationships"][rel["type"]] += 1

    return _finalize_counts(counts)


def _finalize_counts(counts: dict) -> dict:
    counts["nodes"] = dict(counts["nodes"])
    counts["relationships"] = dict(counts["relationships"])
    return counts
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

---

### Task 8: CLI subcommands `load` and `wipe`

**Files:**
- Modify: `pipeline/stage6_load_neo4j.py`

- [ ] **Step 1: Read the existing CLI to understand its argparse structure**

Open [pipeline/stage6_load_neo4j.py](pipeline/stage6_load_neo4j.py) and locate the `main()` function and `_add_connection_args` helper.

- [ ] **Step 2: Extend `main()` with `load` and `wipe` subparsers**

Replace the `main()` function in `pipeline/stage6_load_neo4j.py` with the version below. Keep `_add_connection_args` unchanged.

```python
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 - Neo4j connector and loader.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify Neo4j connectivity.")
    _add_connection_args(check_parser)

    schema_parser = subparsers.add_parser("schema", help="Apply schema.cypher.")
    _add_connection_args(schema_parser)
    schema_parser.add_argument(
        "--schema",
        default=str(REPO_ROOT / "schema.cypher"),
        help="Path to a Cypher schema file.",
    )

    load_parser = subparsers.add_parser("load", help="Load Stage 1-3 outputs into Neo4j.")
    _add_connection_args(load_parser)
    load_parser.add_argument("--in", dest="in_dir", default=None,
                             help="Input dir, defaults to ./entities/.")
    load_parser.add_argument("--batch-size", type=int, default=500)
    load_parser.add_argument("--dry-run", action="store_true",
                             help="Plan and validate without executing Cypher.")

    wipe_parser = subparsers.add_parser("wipe", help="Delete all loader-owned label nodes.")
    _add_connection_args(wipe_parser)
    wipe_parser.add_argument("--yes", action="store_true",
                             help="Skip confirmation. Required for non-interactive use.")

    args = parser.parse_args(argv)
    try:
        config = config_from_env(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        with Neo4jConnector(config) as connector:
            if args.command == "check":
                connector.verify_connectivity()
                ok = connector.smoke_check()
                print(f"Neo4j connection OK: RETURN 1 -> {ok}")
                return 0
            if args.command == "schema":
                count = connector.apply_schema(Path(args.schema))
                print(f"Applied {count} schema statements")
                return 0
            if args.command == "load":
                from pipeline.stage6_load import ENTITIES_ROOT, load
                in_dir = Path(args.in_dir) if args.in_dir else ENTITIES_ROOT
                counts = load(connector, in_dir, batch_size=args.batch_size, dry_run=args.dry_run)
                _print_load_summary(counts, dry_run=args.dry_run)
                return 0
            if args.command == "wipe":
                if not args.yes:
                    print("error: refusing to wipe without --yes", file=sys.stderr)
                    return 1
                from pipeline.stage6_load import wipe
                wipe(connector)
                print("wiped all loader-owned label nodes")
                return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _print_load_summary(counts: dict, *, dry_run: bool) -> None:
    prefix = "would load" if dry_run else "loaded"
    nodes = counts["nodes"]
    rels = counts["relationships"]
    node_total = sum(nodes.values())
    rel_total = sum(rels.values())
    node_breakdown = ", ".join(f"{k}={v}" for k, v in sorted(nodes.items()) if v)
    rel_breakdown = ", ".join(f"{k}={v}" for k, v in sorted(rels.items()) if v)
    print(
        f"{prefix} {counts['pages_loaded']} pages "
        f"({counts['pages_skipped']} skipped, {counts['pages_replaced']} replaced): "
        f"{node_total} nodes ({node_breakdown}), {rel_total} relationships ({rel_breakdown})"
    )
```

- [ ] **Step 3: Run the CLI in dry-run mode**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load --dry-run`

Expected: a single-line summary like `would load 123 pages (0 skipped, 0 replaced): 7979 nodes (Callout=365, ...), 13932 relationships (...)`.

- [ ] **Step 4: Run the CLI for real**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load`

Expected: `loaded 123 pages (0 skipped, 0 replaced): ... nodes (...), ... relationships (...)`.

- [ ] **Step 5: Run again to confirm idempotency**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load`

Expected: `loaded 0 pages (123 skipped, 0 replaced): ...`.

- [ ] **Step 6: Test the wipe guard**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j wipe`

Expected: exit code 1 with `error: refusing to wipe without --yes`.

- [ ] **Step 7: Test wipe with `--yes`**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j wipe --yes`

Expected: `wiped all loader-owned label nodes`. Verify with:

`./.venv/Scripts/python.exe -c "from pipeline.neo4j_connector import *; c = Neo4jConnector(config_from_env()); r,_,_ = c._driver.execute_query('MATCH (n) RETURN count(n) AS c'); print(r[0]['c']); c.close()"`

Expected: `0`.

- [ ] **Step 8: Reload for downstream tasks**

Run: `./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load`

Expected: full corpus reloaded.

---

### Task 9: Validation check `check_load.py` — full assertions

**Files:**
- Modify: `checks/check_load.py`

The previous tasks wrote per-function tests. Now add the spec's "Validation" section: verify counts and structural integrity *after a real load* against the corpus.

- [ ] **Step 1: Write the failing check**

Append to `checks/check_load.py` before `main()`:

```python
def _count_input_records(in_dir: Path) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Return (node_counts_by_label, relationship_counts_by_type, page_hashes)."""
    import json
    from pipeline.stage6_load import ENTITY_KIND, NODE_KINDS, RELATIONSHIP_KIND, dedupe_relationships

    label_for_node_kind = {"page": "Page", "section": "Section",
                           "codeblock": "CodeBlock", "tablerow": "TableRow",
                           "callout": "Callout"}
    nodes: dict[str, int] = {label: 0 for label in label_for_node_kind.values()}
    rels: list[dict] = []
    hashes: dict[str, str] = {}

    for path in sorted(in_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind in label_for_node_kind:
                nodes[label_for_node_kind[kind]] += 1
                if kind == "page":
                    hashes[record["url"]] = record["content_hash"]
            elif kind == ENTITY_KIND:
                nodes[record["label"]] = nodes.get(record["label"], 0) + 1
            elif kind == RELATIONSHIP_KIND:
                rels.append(record)

    rel_counts: dict[str, int] = {}
    for rel in dedupe_relationships(rels):
        rel_counts[rel["type"]] = rel_counts.get(rel["type"], 0) + 1
    return nodes, rel_counts, hashes


def test_corpus_load_counts_match(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import ENTITIES_ROOT, load

    wipe(connector)
    load(connector, ENTITIES_ROOT, batch_size=500, dry_run=False)

    expected_nodes, expected_rels, expected_hashes = _count_input_records(ENTITIES_ROOT)

    for label, expected in expected_nodes.items():
        records, _, _ = connector._driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS c",
            database_=connector.config.database,
        )
        actual = records[0]["c"]
        assert actual == expected, f"{label}: expected {expected}, got {actual}"

    for rel_type, expected in expected_rels.items():
        records, _, _ = connector._driver.execute_query(
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c",
            database_=connector.config.database,
        )
        actual = records[0]["c"]
        assert actual == expected, f"{rel_type}: expected {expected}, got {actual}"

    # Spot-check page hashes match.
    for url, expected_hash in list(expected_hashes.items())[:10]:
        records, _, _ = connector._driver.execute_query(
            "MATCH (p:Page {url: $url}) RETURN p.content_hash AS h",
            {"url": url},
            database_=connector.config.database,
        )
        assert records[0]["h"] == expected_hash, f"{url}: hash mismatch"


def test_no_orphan_sections(connector: Neo4jConnector) -> None:
    records, _, _ = connector._driver.execute_query(
        "MATCH (s:Section) WHERE NOT ( (:Page)-[:HAS_SECTION]->(s) ) RETURN count(s) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == 0, f"{records[0]['c']} orphan Sections"


def test_no_orphan_atoms(connector: Neo4jConnector) -> None:
    queries = (
        "MATCH (b:CodeBlock) WHERE NOT ( (:Section)-[:CONTAINS_CODE]->(b) ) RETURN count(b) AS c",
        "MATCH (r:TableRow) WHERE NOT ( (:Section)-[:CONTAINS_TABLE_ROW]->(r) ) RETURN count(r) AS c",
        "MATCH (c:Callout) WHERE NOT ( (:Section)-[:HAS_CALLOUT]->(c) ) RETURN count(c) AS c",
    )
    for query in queries:
        records, _, _ = connector._driver.execute_query(query, database_=connector.config.database)
        assert records[0]["c"] == 0, f"orphan atoms: {query}"


def test_canonical_entities_present(connector: Neo4jConnector) -> None:
    canonical = (
        "MATCH (t:Tool {name: 'Read'}) RETURN count(t) AS c",
        "MATCH (h:Hook {name: 'SessionStart'}) RETURN count(h) AS c",
        "MATCH (p:Provider {name: 'bedrock'}) RETURN count(p) AS c",
        "MATCH (m:MessageType {name: 'ResultMessage'}) RETURN count(m) AS c",
    )
    for query in canonical:
        records, _, _ = connector._driver.execute_query(query, database_=connector.config.database)
        assert records[0]["c"] == 1, f"missing canonical entity: {query}"
```

Add to `main()` (after the previous tests):

```python
        test_corpus_load_counts_match(connector)
        test_no_orphan_sections(connector)
        test_no_orphan_atoms(connector)
        test_canonical_entities_present(connector)
```

- [ ] **Step 2: Run to verify it passes**

Run: `./.venv/Scripts/python.exe checks/check_load.py`

Expected: `check_load.py OK`.

If a count mismatch fails: the assertion message names the label/type and counts. Investigate by re-running Stage 1–3, checking `./entities/` totals against the existing `check_links.py`/`check_entities.py` reports, and (only if those agree) inspecting Cypher MERGE behaviour for the failing label/type.

---

### Task 10: Regression sweep

**Files:**
- None (verification only).

- [ ] **Step 1: Run all existing checks**

Run, in order:

```
./.venv/Scripts/python.exe checks/check_parse.py
./.venv/Scripts/python.exe checks/check_links.py
./.venv/Scripts/python.exe checks/check_entities.py
./.venv/Scripts/python.exe checks/check_neo4j_config.py
./.venv/Scripts/python.exe checks/check_neo4j.py
./.venv/Scripts/python.exe checks/check_load_logic.py
./.venv/Scripts/python.exe checks/check_load.py
```

Expected: each prints `<name>.py OK`. None should fail or warn beyond the 86 unresolved-link warnings already documented in [CLAUDE.md](CLAUDE.md).

- [ ] **Step 2: Confirm CLI surface end-to-end**

Run:

```
./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j check
./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load
./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load
./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j wipe --yes
./.venv/Scripts/python.exe -m pipeline.stage6_load_neo4j load
```

Expected: connectivity OK; first `load` reports `loaded 123 pages (0 skipped, 0 replaced)`; second reports `loaded 0 pages (123 skipped, 0 replaced)`; `wipe --yes` succeeds; final `load` reports `loaded 123 pages (0 skipped, 0 replaced)` again.

- [ ] **Step 3: Spot-check Neo4j directly**

Run:

```
./.venv/Scripts/python.exe -c "from pipeline.neo4j_connector import *; c = Neo4jConnector(config_from_env()); r,_,_ = c._driver.execute_query('MATCH (n) RETURN count(n) AS c'); print('total nodes:', r[0]['c']); r,_,_ = c._driver.execute_query('MATCH ()-[r]->() RETURN count(r) AS c'); print('total rels:', r[0]['c']); c.close()"
```

Expected: `total nodes: 7979` (Stage 1–3 graph nodes + 159 entities) and `total rels: 13932`. Exact numbers may drift by a few records as the corpus changes; failing only matters if `check_load.py` also fails.
