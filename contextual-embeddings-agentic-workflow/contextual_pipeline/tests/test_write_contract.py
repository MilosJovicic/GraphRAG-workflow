import asyncio
from pathlib import Path

import pytest
from temporalio.exceptions import ApplicationError

from src.activities import write as write_activity


SRC_DIR = Path(__file__).parent.parent / "src"


def test_write_cypher_scopes_matches_by_label_and_primary_key():
    cypher = (SRC_DIR / "cypher" / "write_context.cypher").read_text(encoding="utf-8")

    assert "OR n.url = row.node_id" not in cypher
    assert "OR n.name = row.node_id" not in cypher
    assert "OR n.key = row.node_id" not in cypher

    expected_branches = {
        "Section": "n.id = row.node_id",
        "CodeBlock": "n.id = row.node_id",
        "TableRow": "n.id = row.node_id",
        "Callout": "n.id = row.node_id",
        "Tool": "n.name = row.node_id",
        "Hook": "n.name = row.node_id",
        "PermissionMode": "n.name = row.node_id",
        "MessageType": "n.name = row.node_id",
        "Provider": "n.name = row.node_id",
        "SettingKey": "n.key = row.node_id",
    }
    for label, key_predicate in expected_branches.items():
        assert f"row.label = '{label}'" in cypher
        assert f"MATCH (n:{label})" in cypher
        assert key_predicate in cypher


def test_write_cypher_uses_scoped_subquery_syntax():
    cypher = (SRC_DIR / "cypher" / "write_context.cypher").read_text(encoding="utf-8")

    assert "CALL (row) {" in cypher


def test_per_type_rows_include_label_for_label_scoped_write():
    workflow_source = (SRC_DIR / "workflows" / "per_type.py").read_text(
        encoding="utf-8"
    )

    assert '"label": payload.label' in workflow_source


def test_write_results_raises_when_neo4j_updates_too_few_rows(monkeypatch):
    async def fake_run_query_async(cypher, parameters):
        return [{"updated": len(parameters["rows"]) - 1}]

    monkeypatch.setattr(write_activity, "run_query_async", fake_run_query_async)

    rows = [
        {
            "label": "CodeBlock",
            "node_id": "c1",
            "context": "Context string.",
            "indexed_text": "Context string.\n\n[python]\nprint(1)",
            "embedding": [0.1, 0.2],
            "context_source": "llm",
            "context_version": 1,
        }
    ]

    with pytest.raises(ApplicationError, match="updated 0 of 1 rows"):
        asyncio.run(write_activity.write_results(rows))
