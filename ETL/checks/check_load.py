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
    wipe(connector)
    seed_props = {
        "Page": {"url": "__seed_Page", "path": "__seed.md", "content_hash": "__seed"},
        "Section": {"id": "__seed_Section", "anchor": "__seed", "level": 2},
        "Chunk": {"id": "__seed_Chunk"},
        "CodeBlock": {"id": "__seed_CodeBlock"},
        "TableRow": {"id": "__seed_TableRow"},
        "Callout": {"id": "__seed_Callout"},
        "Tool": {"name": "__seed_Tool"},
        "SettingKey": {"key": "__seed_SettingKey"},
        "Provider": {"name": "__seed_Provider"},
        "MessageType": {"name": "__seed_MessageType"},
        "Hook": {"name": "__seed_Hook"},
        "PermissionMode": {"name": "__seed_PermissionMode"},
    }
    # Seed one node per loader-owned label.
    for label in LOADER_LABELS:
        connector._driver.execute_query(
            f"CREATE (n:{label}) SET n = $props",
            {"props": {**seed_props[label], "__test_seed": True}},
            database_=connector.config.database,
        )
    wipe(connector)
    for label in LOADER_LABELS:
        records, _, _ = connector._driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS c",
            database_=connector.config.database,
        )
        assert records[0]["c"] == 0, f"wipe left {label} nodes"


def test_get_stored_hash_missing(connector: Neo4jConnector) -> None:
    assert get_stored_hash(connector, "https://nope.example/x") is None


def test_merge_entities_loads_catalog(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_entities

    wipe(connector)
    entities = [
        {"kind": "entity", "id": "Tool:Read", "label": "Tool", "name": "Read"},
        {"kind": "entity", "id": "Tool:Edit", "label": "Tool", "name": "Edit", "description": "Edit files"},
        {"kind": "entity", "id": "Hook:SessionStart", "label": "Hook", "name": "SessionStart"},
        {
            "kind": "entity",
            "id": "SettingKey:permissions.allow",
            "label": "SettingKey",
            "key": "permissions.allow",
        },
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


def test_merge_page_subtree_fresh(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_page_subtree

    wipe(connector)
    page = {
        "kind": "page",
        "url": "https://example/p",
        "path": "en/x.md",
        "title": "X",
        "description": "d",
        "description_source": "llms.txt",
        "content_hash": "h1",
        "volatile": False,
    }
    by_kind = {
        "section": [
            {
                "kind": "section",
                "id": "https://example/p#s1",
                "page_url": "https://example/p",
                "anchor": "s1",
                "breadcrumb": "X > S1",
                "level": 2,
                "position": 0,
                "text": "hello",
                "synthesized": False,
                "parent_section_id": None,
                "external_links": ["https://e/x"],
            },
        ],
        "codeblock": [
            {
                "kind": "codeblock",
                "id": "cb1",
                "section_id": "https://example/p#s1",
                "position": 0,
                "language": "python",
                "text": "x = 1",
                "preceding_paragraph": "before",
                "in_codegroup": False,
                "codegroup_id": None,
            },
        ],
        "tablerow": [
            {
                "kind": "tablerow",
                "id": "tr1",
                "section_id": "https://example/p#s1",
                "position": 0,
                "table_position": 0,
                "headers": ["k", "v"],
                "cells": ["a", "b"],
            },
        ],
        "callout": [
            {
                "kind": "callout",
                "id": "co1",
                "section_id": "https://example/p#s1",
                "position": 0,
                "callout_kind": "note",
                "text": "note text",
            },
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


def test_detach_delete_page_descendants(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import detach_delete_page, merge_page_subtree

    wipe(connector)
    page = {
        "kind": "page",
        "url": "https://example/p",
        "path": "en/x.md",
        "title": "X",
        "description": "d",
        "description_source": "llms.txt",
        "content_hash": "h1",
        "volatile": False,
    }
    by_kind = {
        "section": [
            {
                "kind": "section",
                "id": "https://example/p#s1",
                "page_url": "https://example/p",
                "anchor": "s1",
                "breadcrumb": "X > S1",
                "level": 2,
                "position": 0,
                "text": "x",
                "synthesized": False,
                "parent_section_id": None,
                "external_links": [],
            },
        ],
        "codeblock": [
            {
                "kind": "codeblock",
                "id": "cb1",
                "section_id": "https://example/p#s1",
                "position": 0,
                "language": "py",
                "text": "x",
                "preceding_paragraph": "",
                "in_codegroup": False,
                "codegroup_id": None,
            },
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


def test_merge_relationships_all_types(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import merge_entities, merge_page_subtree, merge_relationships

    wipe(connector)
    pages = [
        {
            "kind": "page",
            "url": "https://e/a",
            "path": "en/a.md",
            "title": "A",
            "description": "",
            "description_source": "missing",
            "content_hash": "ha",
            "volatile": False,
        },
        {
            "kind": "page",
            "url": "https://e/b",
            "path": "en/b.md",
            "title": "B",
            "description": "",
            "description_source": "missing",
            "content_hash": "hb",
            "volatile": False,
        },
    ]
    sections = [
        {
            "kind": "section",
            "id": "https://e/a#x",
            "page_url": "https://e/a",
            "anchor": "x",
            "breadcrumb": "A > x",
            "level": 2,
            "position": 0,
            "text": "",
            "synthesized": False,
            "parent_section_id": None,
            "external_links": [],
        },
        {
            "kind": "section",
            "id": "https://e/a#y",
            "page_url": "https://e/a",
            "anchor": "y",
            "breadcrumb": "A > y",
            "level": 3,
            "position": 1,
            "text": "",
            "synthesized": False,
            "parent_section_id": "https://e/a#x",
            "external_links": [],
        },
        {
            "kind": "section",
            "id": "https://e/b#z",
            "page_url": "https://e/b",
            "anchor": "z",
            "breadcrumb": "B > z",
            "level": 2,
            "position": 0,
            "text": "",
            "synthesized": False,
            "parent_section_id": None,
            "external_links": [],
        },
    ]
    codeblocks = [
        {
            "kind": "codeblock",
            "id": "cb1",
            "section_id": "https://e/a#x",
            "position": 0,
            "language": "py",
            "text": "x",
            "preceding_paragraph": "",
            "in_codegroup": True,
            "codegroup_id": "g1",
        },
        {
            "kind": "codeblock",
            "id": "cb2",
            "section_id": "https://e/a#x",
            "position": 1,
            "language": "ts",
            "text": "y",
            "preceding_paragraph": "",
            "in_codegroup": True,
            "codegroup_id": "g1",
        },
    ]
    tablerows = [
        {
            "kind": "tablerow",
            "id": "tr1",
            "section_id": "https://e/a#x",
            "position": 0,
            "table_position": 0,
            "headers": ["tool"],
            "cells": ["Read"],
        },
    ]
    callouts: list[dict] = []
    merge_page_subtree(
        connector,
        pages[0],
        {"section": [sections[0], sections[1]], "codeblock": codeblocks, "tablerow": tablerows, "callout": callouts},
        batch_size=10,
    )
    merge_page_subtree(
        connector,
        pages[1],
        {"section": [sections[2]], "codeblock": [], "tablerow": [], "callout": []},
        batch_size=10,
    )
    merge_entities(
        connector,
        [{"kind": "entity", "id": "Tool:Read", "label": "Tool", "name": "Read"}],
        batch_size=10,
    )

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
        "HAS_SECTION": 3,
        "HAS_SUBSECTION": 1,
        "CONTAINS_CODE": 2,
        "CONTAINS_TABLE_ROW": 1,
        "EQUIVALENT_TO": 1,
        "LINKS_TO": 1,
        "LINKS_TO_PAGE": 1,
        "NAVIGATES_TO": 1,
        "MENTIONS": 1,
        "DEFINES": 1,
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


def test_load_full_corpus(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import ENTITIES_ROOT, load

    wipe(connector)
    counts = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=False)
    assert counts["pages_loaded"] == 123, counts
    assert counts["pages_skipped"] == 0
    assert counts["nodes"]["Page"] == 123
    assert counts["nodes"]["Section"] >= 2700
    assert counts["nodes"]["Tool"] == 35
    assert counts["nodes"]["Hook"] == 29

    counts2 = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=False)
    assert counts2["pages_skipped"] == 123
    assert counts2["pages_loaded"] == 0


def test_load_dry_run(connector: Neo4jConnector) -> None:
    from pipeline.stage6_load import ENTITIES_ROOT, load

    wipe(connector)
    counts = load(connector, ENTITIES_ROOT, batch_size=500, dry_run=True)
    assert counts["pages_loaded"] == 123
    records, _, _ = connector._driver.execute_query(
        "MATCH (p:Page) RETURN count(p) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == 0


def _count_input_records(in_dir: Path) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Return (node_counts_by_label, relationship_counts_by_type, page_hashes)."""
    import json

    from pipeline.stage6_load import ENTITY_KIND, RELATIONSHIP_KIND, dedupe_relationships

    label_for_node_kind = {
        "page": "Page",
        "section": "Section",
        "codeblock": "CodeBlock",
        "tablerow": "TableRow",
        "callout": "Callout",
    }
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

    records, _, _ = connector._driver.execute_query(
        "MATCH (p:Page) RETURN p.url AS url, p.content_hash AS hash",
        database_=connector.config.database,
    )
    actual_hashes = {record["url"]: record["hash"] for record in records}
    assert actual_hashes == expected_hashes


def test_no_orphan_sections(connector: Neo4jConnector) -> None:
    records, _, _ = connector._driver.execute_query(
        "MATCH (s:Section) "
        "WHERE NOT ((:Page)-[:HAS_SECTION]->(s)) "
        "AND NOT ((:Section)-[:HAS_SUBSECTION]->(s)) "
        "RETURN count(s) AS c",
        database_=connector.config.database,
    )
    assert records[0]["c"] == 0, f"{records[0]['c']} orphan Sections"


def test_no_orphan_atoms(connector: Neo4jConnector) -> None:
    queries = (
        "MATCH (b:CodeBlock) WHERE NOT ((:Section)-[:CONTAINS_CODE]->(b)) RETURN count(b) AS c",
        "MATCH (r:TableRow) WHERE NOT ((:Section)-[:CONTAINS_TABLE_ROW]->(r)) RETURN count(r) AS c",
        "MATCH (c:Callout) WHERE NOT ((:Section)-[:HAS_CALLOUT]->(c)) RETURN count(c) AS c",
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


def main() -> int:
    config = config_from_env()
    with Neo4jConnector(config) as connector:
        test_schema_applied(connector)
        test_wipe_clears_loader_labels(connector)
        test_get_stored_hash_missing(connector)
        test_merge_entities_loads_catalog(connector)
        test_merge_page_subtree_fresh(connector)
        test_detach_delete_page_descendants(connector)
        test_detach_delete_page_missing_is_noop(connector)
        test_merge_relationships_all_types(connector)
        test_load_full_corpus(connector)
        test_load_dry_run(connector)
        test_corpus_load_counts_match(connector)
        test_no_orphan_sections(connector)
        test_no_orphan_atoms(connector)
        test_canonical_entities_present(connector)
    print("check_load.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
