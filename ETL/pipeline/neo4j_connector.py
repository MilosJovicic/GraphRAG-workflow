"""Neo4j connection helpers for Stage 6."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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


class SchemaApplyError(RuntimeError):
    """Raised when a schema statement fails during application."""

    def __init__(self, statement_number: int, original: Exception) -> None:
        super().__init__(f"schema statement {statement_number} failed: {original}")
        self.statement_number = statement_number
        self.original = original


class Neo4jConnector:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver = self._make_driver(config)

    @staticmethod
    def _make_driver(config: Neo4jConfig) -> Any:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError(
                "Neo4j driver is not installed; run `python -m pip install -r requirements.txt`."
            ) from exc
        return GraphDatabase.driver(
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
        for idx, statement in enumerate(statements, start=1):
            try:
                self.run_statement(statement)
            except Exception as exc:
                raise SchemaApplyError(idx, exc) from exc
        return len(statements)
