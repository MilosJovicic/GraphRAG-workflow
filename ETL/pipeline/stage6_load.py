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

ENTITY_KEY_PROP = {
    "Tool": "name",
    "SettingKey": "key",
    "Hook": "name",
    "PermissionMode": "name",
    "Provider": "name",
    "MessageType": "name",
}

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


def read_entity_catalog(path: Path) -> list[dict]:
    """Read _entities.jsonl. Returns entity records only.

    Raises ValueError (with file path and line number) if any record has
    kind != 'entity'.
    """
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") != ENTITY_KIND:
                raise ValueError(
                    f"{path}:{line_no}: expected entity record, got {record.get('kind')!r}"
                )
            records.append(record)
    return records


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


def decide_load_action(incoming_hash: str, stored_hash: str | None) -> str:
    """Return one of: 'merge_all' (page absent), 'skip_nodes' (hash matches),
    'replace' (hash differs)."""
    if stored_hash is None:
        return "merge_all"
    if stored_hash == incoming_hash:
        return "skip_nodes"
    return "replace"


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


def _batched(items: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def merge_entities(connector, entities: list[dict], batch_size: int) -> None:
    """Phase 1: MERGE entity nodes by their natural key.

    The composite `id` property is also written so MENTIONS/DEFINES targets can
    be matched uniformly across entity labels later.
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
                props = {
                    key: value
                    for key, value in entity.items()
                    if key not in {"kind", "id", "label", key_prop}
                }
                payload.append({key_prop: entity[key_prop], "id": entity["id"], "props": props})
            connector._driver.execute_query(
                cypher,
                {"batch": payload},
                database_=connector.config.database,
            )


_PAGE_PROPS = ("url", "path", "title", "description", "description_source", "content_hash", "volatile")
_SECTION_PROPS = (
    "id",
    "page_url",
    "anchor",
    "breadcrumb",
    "level",
    "position",
    "text",
    "synthesized",
    "parent_section_id",
    "external_links",
)
_CODEBLOCK_PROPS = (
    "id",
    "section_id",
    "position",
    "language",
    "text",
    "preceding_paragraph",
    "in_codegroup",
    "codegroup_id",
)
_TABLEROW_PROPS = ("id", "section_id", "position", "table_position", "headers", "cells")
_CALLOUT_PROPS = ("id", "section_id", "position", "callout_kind", "text")


def _project(record: dict, props: tuple[str, ...]) -> dict:
    return {prop: record.get(prop) for prop in props}


def _merge_batch(
    connector,
    label: str,
    key_prop: str,
    props: tuple[str, ...],
    records: list[dict],
    batch_size: int,
) -> None:
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


def merge_page_subtree(
    connector,
    page: dict,
    by_kind: dict[str, list[dict]],
    batch_size: int,
) -> None:
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


def detach_delete_page(connector, page_url: str) -> None:
    """Delete a Page and all descendant nodes.

    Entity nodes are never touched. Children of nested Sections are removed
    before Sections so a changed-page reload cannot leave atoms orphaned.
    """
    queries = (
        (
            "MATCH (:Page {url: $url})-[:HAS_SECTION]->(s:Section) "
            "OPTIONAL MATCH (s)-[:HAS_SUBSECTION*0..]->(d:Section) "
            "WITH collect(DISTINCT d) AS sections "
            "UNWIND sections AS section "
            "WITH DISTINCT section "
            "WHERE section IS NOT NULL "
            "OPTIONAL MATCH (section)-[:CONTAINS_CODE|CONTAINS_TABLE_ROW|HAS_CALLOUT]->(child) "
            "WITH collect(DISTINCT child) AS children "
            "UNWIND children AS child "
            "WITH DISTINCT child "
            "WHERE child IS NOT NULL "
            "DETACH DELETE child"
        ),
        (
            "MATCH (:Page {url: $url})-[:HAS_SECTION]->(s:Section) "
            "OPTIONAL MATCH (s)-[:HAS_SUBSECTION*0..]->(d:Section) "
            "WITH collect(DISTINCT d) AS sections "
            "UNWIND sections AS section "
            "WITH DISTINCT section "
            "WHERE section IS NOT NULL "
            "DETACH DELETE section"
        ),
        "MATCH (p:Page {url: $url}) DETACH DELETE p",
    )
    for cypher in queries:
        connector._driver.execute_query(
            cypher,
            {"url": page_url},
            database_=connector.config.database,
        )


# (source_label, source_key_prop, target_label, target_key_prop)
# None means dispatch dynamically (entity targets, or DEFINES sources).
_REL_DISPATCH: dict[str, tuple[str | None, str, str | None, str | None]] = {
    "HAS_SECTION": ("Page", "url", "Section", "id"),
    "HAS_SUBSECTION": ("Section", "id", "Section", "id"),
    "CONTAINS_CODE": ("Section", "id", "CodeBlock", "id"),
    "CONTAINS_TABLE_ROW": ("Section", "id", "TableRow", "id"),
    "HAS_CALLOUT": ("Section", "id", "Callout", "id"),
    "LINKS_TO": ("Section", "id", "Section", "id"),
    "LINKS_TO_PAGE": ("Section", "id", "Page", "url"),
    "NAVIGATES_TO": ("Section", "id", "Section", "id"),
    "EQUIVALENT_TO": ("CodeBlock", "id", "CodeBlock", "id"),
    "MENTIONS": ("Section", "id", None, None),
    "DEFINES": (None, "id", None, None),
}

_REL_PROP_KEYS = ("id", "href", "text", "evidence")


def _entity_label_from_id(entity_id: str) -> str:
    return entity_id.split(":", 1)[0]


def _rel_props(record: dict) -> dict:
    return {key: record[key] for key in _REL_PROP_KEYS if key in record}


def merge_relationships(connector, records: list[dict], batch_size: int) -> None:
    """Phase 3: MERGE all relationships globally."""
    deduped = dedupe_relationships(records)

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in deduped:
        rel_type = record["type"]
        if rel_type not in _REL_DISPATCH:
            raise ValueError(f"unknown relationship type {rel_type!r}")
        src_label, _src_key, tgt_label, _tgt_key = _REL_DISPATCH[rel_type]

        if src_label is None:
            src_label = "Section" if "#" in record["source_id"] else "TableRow"
        if tgt_label is None:
            tgt_label = _entity_label_from_id(record["target_id"])
            if tgt_label not in ENTITY_KEY_PROP:
                raise ValueError(f"unknown entity target label {tgt_label!r} in {record}")

        groups[(rel_type, src_label, tgt_label)].append(record)

    for (rel_type, src_label, tgt_label), group_records in groups.items():
        src_key = _REL_DISPATCH[rel_type][1]
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
                {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "props": _rel_props(record),
                }
                for record in batch
            ]
            connector._driver.execute_query(
                cypher,
                {"batch": payload},
                database_=connector.config.database,
            )


def load(connector, in_dir: Path, batch_size: int, dry_run: bool) -> dict:
    """Top-level Stage 6 loader.

    On dry-run, planned operation counts are reported but no Cypher executes.
    Re-runs are idempotent: pages whose content_hash matches are skipped for
    node writes while relationships are still re-MERGE'd globally.
    """
    if not in_dir.exists():
        raise FileNotFoundError(
            f"{in_dir} not found; run 'python -m pipeline.stage3_entities' first"
        )

    if not dry_run:
        check_schema_applied(connector)

    entities_path = in_dir / "_entities.jsonl"
    page_files = sorted(path for path in in_dir.glob("*.jsonl") if path.name != "_entities.jsonl")
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
        all_relationships.extend(by_kind.get("relationship", []))

    if dry_run:
        for _page, by_kind in page_buckets:
            counts["pages_loaded"] += 1
            counts["nodes"]["Page"] += 1
            for kind, label in (
                ("section", "Section"),
                ("codeblock", "CodeBlock"),
                ("tablerow", "TableRow"),
                ("callout", "Callout"),
            ):
                counts["nodes"][label] += len(by_kind.get(kind, []))
        for entity in entities:
            counts["nodes"][entity["label"]] += 1
        for rel in dedupe_relationships(all_relationships):
            counts["relationships"][rel["type"]] += 1
        return _finalize_counts(counts)

    if entities:
        merge_entities(connector, entities, batch_size)
    for entity in entities:
        counts["nodes"][entity["label"]] += 1

    for page, by_kind in page_buckets:
        action = decide_load_action(page["content_hash"], get_stored_hash(connector, page["url"]))
        if action == "skip_nodes":
            counts["pages_skipped"] += 1
        elif action == "replace":
            detach_delete_page(connector, page["url"])
            merge_page_subtree(connector, page, by_kind, batch_size)
            counts["pages_replaced"] += 1
            counts["pages_loaded"] += 1
        else:
            merge_page_subtree(connector, page, by_kind, batch_size)
            counts["pages_loaded"] += 1

        counts["nodes"]["Page"] += 1
        for kind, label in (
            ("section", "Section"),
            ("codeblock", "CodeBlock"),
            ("tablerow", "TableRow"),
            ("callout", "Callout"),
        ):
            counts["nodes"][label] += len(by_kind.get(kind, []))

    if all_relationships:
        merge_relationships(connector, all_relationships, batch_size)
        for rel in dedupe_relationships(all_relationships):
            counts["relationships"][rel["type"]] += 1

    return _finalize_counts(counts)


def _finalize_counts(counts: dict) -> dict:
    counts["nodes"] = dict(counts["nodes"])
    counts["relationships"] = dict(counts["relationships"])
    return counts
