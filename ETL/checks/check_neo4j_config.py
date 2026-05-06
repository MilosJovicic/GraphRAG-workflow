"""Validate Neo4j connector configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.neo4j_connector import (  # noqa: E402
    Neo4jConnector,
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
        "    CREATE CONSTRAINT example IF NOT EXISTS\n      FOR (p:Page) REQUIRE p.url IS UNIQUE",
        "    CREATE INDEX page_path IF NOT EXISTS\n      FOR (p:Page) ON (p.path)",
    ]


def test_connector_api_exists() -> None:
    for method_name in ("verify_connectivity", "smoke_check", "apply_schema"):
        assert callable(getattr(Neo4jConnector, method_name))


def test_project_schema_splits_into_statements() -> None:
    schema_path = REPO_ROOT / "schema.cypher"
    statements = split_cypher_statements(schema_path.read_text(encoding="utf-8"))
    assert len(statements) > 10
    assert statements[0].startswith("CREATE CONSTRAINT page_url")


def main() -> int:
    test_config_precedence()
    test_missing_password_fails()
    test_split_schema_ignores_comments()
    test_connector_api_exists()
    test_project_schema_splits_into_statements()
    print("check_neo4j_config.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
