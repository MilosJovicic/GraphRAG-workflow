# Neo4j Connector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Neo4j connector foundation that can smoke-test the local database and apply `schema.cypher`.

**Architecture:** Keep database access in `pipeline/neo4j_connector.py`, with a thin CLI wrapper in `pipeline/stage6_load_neo4j.py`. Use existing `checks/` style scripts for unit-like connector checks and local Neo4j verification.

**Tech Stack:** Python standard library, official `neo4j` Python driver, existing plain Python check scripts.

---

## File Structure

- Create `pipeline/neo4j_connector.py`: config object, env/CLI precedence helper, schema statement parser, driver wrapper, smoke check, schema application.
- Create `pipeline/stage6_load_neo4j.py`: CLI entry point with `check` and `schema` subcommands.
- Create `checks/check_neo4j_config.py`: unit-like checks for config and schema splitting.
- Create `checks/check_neo4j.py`: local Neo4j smoke check.
- Modify `requirements.txt`: add official Neo4j driver dependency.

---

### Task 1: Connector Unit Checks

**Files:**
- Create: `checks/check_neo4j_config.py`
- Later create: `pipeline/neo4j_connector.py`

- [x] **Step 1: Write the failing check**

Create `checks/check_neo4j_config.py` with tests for:

```python
from pipeline.neo4j_connector import (
    Neo4jConfig,
    config_from_env,
    split_cypher_statements,
)


def test_config_precedence() -> None:
    env = {
        "NEO4J_URI": "neo4j://env:7687",
        "NEO4J_USER": "env-user",
        "NEO4J_PASSWORD": "env-pass",
        "NEO4J_DATABASE": "env-db",
    }
    config = config_from_env(
        env,
        uri="neo4j://flag:7687",
        user=None,
        password="flag-pass",
        database=None,
    )
    assert config == Neo4jConfig(
        uri="neo4j://flag:7687",
        user="env-user",
        password="flag-pass",
        database="env-db",
    )


def test_missing_password_fails() -> None:
    try:
        config_from_env({}, uri=None, user=None, password=None, database=None)
    except ValueError as exc:
        assert "NEO4J_PASSWORD" in str(exc)
    else:
        raise AssertionError("missing password should fail")


def test_split_schema_ignores_comments() -> None:
    source = """
    // comment with ; semicolon
    CREATE CONSTRAINT example IF NOT EXISTS
      FOR (p:Page) REQUIRE p.url IS UNIQUE;

    // another comment
    CREATE INDEX page_path IF NOT EXISTS
      FOR (p:Page) ON (p.path);
    """
    assert split_cypher_statements(source) == [
        "CREATE CONSTRAINT example IF NOT EXISTS\n      FOR (p:Page) REQUIRE p.url IS UNIQUE",
        "CREATE INDEX page_path IF NOT EXISTS\n      FOR (p:Page) ON (p.path)",
    ]


def main() -> int:
    test_config_precedence()
    test_missing_password_fails()
    test_split_schema_ignores_comments()
    print("check_neo4j_config.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run check to verify it fails**

Run: `python checks/check_neo4j_config.py`

Expected: failure because `pipeline.neo4j_connector` does not exist.

- [x] **Step 3: Write minimal connector helpers**

Create `pipeline/neo4j_connector.py` with:

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_URI = "neo4j://127.0.0.1:7687"
DEFAULT_USER = "neo4j"


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str | None = None


def config_from_env(
    env: Mapping[str, str] | None = None,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Neo4jConfig:
    values = os.environ if env is None else env
    resolved_password = password or values.get("NEO4J_PASSWORD")
    if not resolved_password:
        raise ValueError("Neo4j password is required; set NEO4J_PASSWORD or pass --password.")
    return Neo4jConfig(
        uri=uri or values.get("NEO4J_URI", DEFAULT_URI),
        user=user or values.get("NEO4J_USER", DEFAULT_USER),
        password=resolved_password,
        database=database or values.get("NEO4J_DATABASE") or None,
    )


def split_cypher_statements(source: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line.rstrip())
        if stripped.endswith(";"):
            statement = "\n".join(current).rstrip()
            statements.append(statement[:-1].rstrip())
            current = []
    if current:
        statements.append("\n".join(current).rstrip())
    return statements
```

- [x] **Step 4: Run check to verify it passes**

Run: `python checks/check_neo4j_config.py`

Expected: `check_neo4j_config.py OK`.

---

### Task 2: Neo4j Driver Wrapper And CLI

**Files:**
- Modify: `pipeline/neo4j_connector.py`
- Create: `pipeline/stage6_load_neo4j.py`
- Modify: `requirements.txt`

- [x] **Step 1: Add failing check expectations**

Extend `checks/check_neo4j_config.py` to import `Neo4jConnector`, verify it exposes `verify_connectivity`, `smoke_check`, and `apply_schema`, and verify `schema.cypher` splits into more than ten statements.

- [x] **Step 2: Run check to verify it fails**

Run: `python checks/check_neo4j_config.py`

Expected: failure because `Neo4jConnector` does not exist yet.

- [x] **Step 3: Add driver wrapper**

Extend `pipeline/neo4j_connector.py` with:

```python
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


class Neo4jConnector:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jConnector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def smoke_check(self) -> int:
        records, _, _ = self._driver.execute_query(
            "RETURN 1 AS ok",
            database_=self.config.database,
        )
        return int(records[0]["ok"])

    def run_statement(self, statement: str) -> None:
        self._driver.execute_query(statement, database_=self.config.database)

    def apply_schema(self, schema_path: Path) -> int:
        statements = split_cypher_statements(schema_path.read_text(encoding="utf-8"))
        for statement in statements:
            self.run_statement(statement)
        return len(statements)
```

Add `neo4j==5.28.1` to `requirements.txt`.

- [x] **Step 4: Add CLI**

Create `pipeline/stage6_load_neo4j.py` with argparse subcommands:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .neo4j_connector import Neo4jConnector, config_from_env
from .stage1_parse import REPO_ROOT


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uri", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 - Neo4j connector utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify Neo4j connectivity.")
    _add_connection_args(check_parser)

    schema_parser = subparsers.add_parser("schema", help="Apply schema.cypher.")
    _add_connection_args(schema_parser)
    schema_parser.add_argument("--schema", default=str(REPO_ROOT / "schema.cypher"))

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
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [x] **Step 5: Run check to verify it passes**

Run: `python checks/check_neo4j_config.py`

Expected: `check_neo4j_config.py OK`.

---

### Task 3: Local Neo4j Smoke Check

**Files:**
- Create: `checks/check_neo4j.py`

- [x] **Step 1: Write local smoke check**

Create `checks/check_neo4j.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.neo4j_connector import Neo4jConnector, config_from_env  # noqa: E402


def main() -> int:
    try:
        config = config_from_env()
        with Neo4jConnector(config) as connector:
            connector.verify_connectivity()
            ok = connector.smoke_check()
    except Exception as exc:
        print(f"check_neo4j.py FAILED: {exc}")
        return 1
    if ok != 1:
        print(f"check_neo4j.py FAILED: expected 1, got {ok}")
        return 1
    print("check_neo4j.py OK: Neo4j returned 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Install dependencies**

Run: `python -m pip install -r requirements.txt`

Expected: `neo4j` package installed or already satisfied.

- [x] **Step 3: Run local smoke check**

Run with the local password supplied through the environment:

```powershell
$env:NEO4J_URI='neo4j://127.0.0.1:7687'
$env:NEO4J_USER='neo4j'
$env:NEO4J_PASSWORD='<local password>'
python checks/check_neo4j.py
```

Expected: `check_neo4j.py OK: Neo4j returned 1`.

- [x] **Step 4: Run CLI smoke check**

Run: `python -m pipeline.stage6_load_neo4j check`

Expected: `Neo4j connection OK: RETURN 1 -> 1`.

- [x] **Step 5: Run schema application**

Run: `python -m pipeline.stage6_load_neo4j schema`

Expected: `Applied <n> schema statements`, where `<n>` is the number of statements parsed from `schema.cypher`.

---

### Task 4: Regression Verification

**Files:**
- No new files.

- [x] **Step 1: Run existing checks**

Run:

```powershell
python checks/check_parse.py
python checks/check_links.py
python checks/check_entities.py
```

Expected: all existing checks print `OK`.

- [x] **Step 2: Run new checks again**

Run:

```powershell
python checks/check_neo4j_config.py
python checks/check_neo4j.py
```

Expected: both new checks print `OK`.

- [x] **Step 3: Confirm source does not contain local password**

Run: `rg "<local password value>"`

Expected: no source files contain the password. Command may find shell history outside the project only if run beyond the repo root; run from project root.
