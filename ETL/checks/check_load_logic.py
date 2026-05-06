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


def test_decide_load_action() -> None:
    assert decide_load_action("h1", None) == "merge_all"
    assert decide_load_action("h1", "h1") == "skip_nodes"
    assert decide_load_action("h2", "h1") == "replace"


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


def test_read_entity_catalog_rejects_non_entity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_entities.jsonl"
        _write_jsonl(path, [{"kind": "section", "id": "x"}])
        try:
            read_entity_catalog(path)
        except ValueError as exc:
            assert "section" in str(exc)
            assert str(path) in str(exc)
        else:
            raise AssertionError("non-entity record must raise")


def test_read_page_records_no_page_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.jsonl"
        _write_jsonl(path, [{"kind": "section", "id": "u#a", "page_url": "u",
                             "anchor": "a", "level": 2, "position": 0}])
        try:
            read_page_records(path)
        except ValueError as exc:
            assert "no page record" in str(exc)
            assert str(path) in str(exc)
        else:
            raise AssertionError("missing page record must raise")


def test_read_page_records_multiple_page_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.jsonl"
        _write_jsonl(
            path,
            [
                {"kind": "page", "url": "u1", "path": "p1", "title": "T1", "content_hash": "h1"},
                {"kind": "page", "url": "u2", "path": "p2", "title": "T2", "content_hash": "h2"},
            ],
        )
        try:
            read_page_records(path)
        except ValueError as exc:
            assert "more than one page record" in str(exc)
            assert str(path) in str(exc)
        else:
            raise AssertionError("multiple page records must raise")


def main() -> int:
    test_read_entity_catalog()
    test_read_entity_catalog_rejects_non_entity()
    test_read_page_records_buckets_and_skips()
    test_read_page_records_rejects_unknown_kind()
    test_read_page_records_no_page_record()
    test_read_page_records_multiple_page_records()
    test_decide_load_action()
    test_dedupe_relationships()
    print("check_load_logic.py OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
