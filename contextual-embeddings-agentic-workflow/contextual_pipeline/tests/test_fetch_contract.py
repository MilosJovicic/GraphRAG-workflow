import asyncio
from pathlib import Path

import pytest
from temporalio.exceptions import ApplicationError

from src.activities import fetch as fetch_activity


SRC_DIR = Path(__file__).parent.parent / "src"


def test_entities_fetch_cypher_filters_by_requested_label():
    cypher = (
        SRC_DIR / "cypher" / "fetch_nodes" / "entities.cypher"
    ).read_text(encoding="utf-8")

    assert "$label" in cypher
    assert "$label IN labels(n)" in cypher
    assert "$label = 'SettingKey'" in cypher


def test_tablerow_fetch_uses_current_graph_relationships():
    cypher = (
        SRC_DIR / "cypher" / "fetch_nodes" / "tablerows.cypher"
    ).read_text(encoding="utf-8")

    assert "CONTAINS_TABLE_ROW" in cypher
    assert "HAS_ROW" not in cypher
    assert "[:CONTAINS_TABLE]" not in cypher
    assert "s.text        AS parent_text" in cypher
    assert "p.title       AS page_title" in cypher


def test_callout_fetch_uses_current_graph_relationships_and_kind_property():
    cypher = (
        SRC_DIR / "cypher" / "fetch_nodes" / "callouts.cypher"
    ).read_text(encoding="utf-8")

    assert "HAS_CALLOUT" in cypher
    assert "CONTAINS_CALLOUT" not in cypher
    assert "c.callout_kind" in cypher
    assert "c.kind" not in cypher
    assert "s.text        AS parent_text" in cypher


def test_codeblock_fetch_returns_equivalent_hints():
    cypher = (
        SRC_DIR / "cypher" / "fetch_nodes" / "codeblocks.cypher"
    ).read_text(encoding="utf-8")

    assert "EQUIVALENT_TO" in cypher
    assert "related_hints" in cypher


def test_entities_fetch_returns_definition_context():
    cypher = (
        SRC_DIR / "cypher" / "fetch_nodes" / "entities.cypher"
    ).read_text(encoding="utf-8")

    assert "DEFINES" in cypher
    assert "defined_in" in cypher
    assert "definition_text" in cypher


def test_fetch_nodes_batch_passes_label_parameter(monkeypatch):
    captured = {}

    async def fake_run_query_async(cypher, parameters):
        captured.update(parameters)
        return [
            {
                "node_id": "Bash",
                "raw_text": "Runs shell commands.",
                "name": "Bash",
                "label": "Tool",
            }
        ]

    monkeypatch.setattr(fetch_activity, "run_query_async", fake_run_query_async)

    payloads = asyncio.run(fetch_activity.fetch_nodes_batch("Tool", ["Bash"]))

    assert captured == {"label": "Tool", "ids": ["Bash"]}
    assert payloads[0].label == "Tool"


def test_fetch_nodes_batch_maps_entity_definition_context(monkeypatch):
    async def fake_run_query_async(cypher, parameters):
        return [
            {
                "node_id": "default",
                "raw_text": "Reads only",
                "name": "default",
                "label": "PermissionMode",
                "related_hints": ["plan", "acceptEdits"],
                "defined_in": "Choose a permission mode > Available modes",
                "definition_text": "Modes set the baseline. The table below shows each mode.",
            }
        ]

    monkeypatch.setattr(fetch_activity, "run_query_async", fake_run_query_async)

    payloads = asyncio.run(fetch_activity.fetch_nodes_batch("PermissionMode", ["default"]))

    assert payloads[0].related_hints == ["plan", "acceptEdits"]
    assert payloads[0].extras["defined_in"] == "Choose a permission mode > Available modes"
    assert payloads[0].extras["definition_text"].startswith("Modes set the baseline")


def test_fetch_nodes_batch_maps_tablerow_and_callout_parent_context(monkeypatch):
    async def fake_run_query_async(cypher, parameters):
        if parameters["label"] == "TableRow":
            return [
                {
                    "node_id": "row-1",
                    "raw_text": "",
                    "headers": ["error", "fix"],
                    "cells": ["No tab available", "Create a new tab"],
                    "breadcrumb": "Chrome > Troubleshooting",
                    "parent_text": "Common Chrome extension errors.",
                    "page_title": "Use Claude Code with Chrome",
                    "page_description": "Browser integration",
                }
            ]
        return [
            {
                "node_id": "callout-1",
                "raw_text": "Use jq to parse the response.",
                "callout_kind": "tip",
                "breadcrumb": "CLI > JSON output",
                "parent_text": "Responses can be JSON or stream-json.",
                "page_title": "Run Claude Code programmatically",
                "page_description": "CLI automation",
            }
        ]

    monkeypatch.setattr(fetch_activity, "run_query_async", fake_run_query_async)

    table = asyncio.run(fetch_activity.fetch_nodes_batch("TableRow", ["row-1"]))[0]
    callout = asyncio.run(fetch_activity.fetch_nodes_batch("Callout", ["callout-1"]))[0]

    assert table.breadcrumb == "Chrome > Troubleshooting"
    assert table.parent_text == "Common Chrome extension errors."
    assert table.extras["headers"] == "error|fix"
    assert callout.parent_text == "Responses can be JSON or stream-json."
    assert callout.extras["callout_kind"] == "tip"


def test_fetch_nodes_batch_maps_codeblock_equivalent_hints(monkeypatch):
    async def fake_run_query_async(cypher, parameters):
        return [
            {
                "node_id": "cb1",
                "raw_text": "print('hi')",
                "language": "python",
                "preceding": "",
                "related_hints": ["typescript", "javascript"],
            }
        ]

    monkeypatch.setattr(fetch_activity, "run_query_async", fake_run_query_async)

    payloads = asyncio.run(fetch_activity.fetch_nodes_batch("CodeBlock", ["cb1"]))

    assert payloads[0].related_hints == ["typescript", "javascript"]


def test_fetch_nodes_batch_raises_for_malformed_rows(monkeypatch):
    async def fake_run_query_async(cypher, parameters):
        return [
            {
                "node_id": "page-1",
                "raw_text": "Pages are not processed by this activity.",
                "label": "Page",
            }
        ]

    monkeypatch.setattr(fetch_activity, "run_query_async", fake_run_query_async)

    with pytest.raises(ApplicationError, match="Malformed Neo4j row"):
        asyncio.run(fetch_activity.fetch_nodes_batch("Tool", ["page-1"]))
