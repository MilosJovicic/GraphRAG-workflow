"""Stage 3 - closed-vocabulary entity extraction.

Consumes Stage 2 resolved JSONL and appends closed-catalog entity nodes plus
DEFINES/MENTIONS relationships. This deliberately avoids generic NER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .stage1_parse import REPO_ROOT
from .stage2_resolve import RESOLVED_ROOT


ENTITIES_ROOT = REPO_ROOT / "entities"

ENTITY_LABELS = {
    "Tool",
    "SettingKey",
    "Provider",
    "MessageType",
    "Hook",
    "PermissionMode",
}

MESSAGE_TYPES = [
    "SystemMessage",
    "AssistantMessage",
    "UserMessage",
    "StreamEvent",
    "ResultMessage",
]

PROVIDERS = {
    "bedrock": "en/amazon-bedrock.md",
    "vertex-ai": "en/google-vertex-ai.md",
    "foundry": "en/microsoft-foundry.md",
}

CODE_SPAN_RE = re.compile(r"`([^`]+)`")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(records: Iterable[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def _entity(label: str, name: str, **props: object) -> dict:
    if label not in ENTITY_LABELS:
        raise ValueError(f"unexpected entity label {label}")
    key_prop = "key" if label == "SettingKey" else "name"
    return {
        "kind": "entity",
        "id": f"{label}:{name}",
        "label": label,
        key_prop: name,
        **{k: v for k, v in props.items() if v not in (None, "", [])},
    }


def _relationship(
    rel_type: str,
    source_id: str,
    target_id: str,
    **props: object,
) -> dict:
    h = hashlib.sha256(
        f"{rel_type}|{source_id}|{target_id}|{props.get('evidence', '')}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "kind": "relationship",
        "id": h,
        "type": rel_type,
        "source_id": source_id,
        "target_id": target_id,
        **{k: v for k, v in props.items() if v not in (None, "", [])},
    }


def _code_identifiers(text: str) -> list[str]:
    values = []
    for raw in CODE_SPAN_RE.findall(text):
        cleaned = raw.strip().strip('"').strip("'")
        if cleaned:
            values.append(cleaned)
    return values


def _first_code_or_text(text: str) -> str:
    codes = _code_identifiers(text)
    if codes:
        return codes[0]
    return re.sub(r"\s+", " ", text).strip()


def _mention_pattern(identifier: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    )


class Catalog:
    def __init__(self, records_by_file: dict[Path, list[dict]]) -> None:
        self.records_by_file = records_by_file
        self.records = [record for records in records_by_file.values() for record in records]
        self.file_by_page_path: dict[str, Path] = {}
        self.file_by_record_id: dict[str, Path] = {}
        self.pages_by_path: dict[str, dict] = {}
        self.sections_by_page_anchor: dict[tuple[str, str], dict] = {}
        self.sections_by_page: dict[str, list[dict]] = defaultdict(list)

        for path, records in records_by_file.items():
            page = next((record for record in records if record["kind"] == "page"), None)
            if page:
                self.file_by_page_path[page["path"]] = path
                self.pages_by_path[page["path"]] = page
            for record in records:
                if "id" in record:
                    self.file_by_record_id[record["id"]] = path
                if record["kind"] == "section":
                    self.sections_by_page_anchor[
                        (record["page_url"], record["anchor"])
                    ] = record
                    self.sections_by_page[record["page_url"]].append(record)

        for sections in self.sections_by_page.values():
            sections.sort(key=lambda section: section["position"])

    def page_records(self, page_path: str) -> list[dict]:
        path = self.file_by_page_path.get(page_path)
        return self.records_by_file.get(path, []) if path else []

    def first_section_for_page_path(self, page_path: str) -> dict | None:
        page = self.pages_by_path.get(page_path)
        if not page:
            return None
        sections = self.sections_by_page.get(page["url"], [])
        return sections[0] if sections else None

    def section(self, page_path: str, anchor: str) -> dict | None:
        page = self.pages_by_path.get(page_path)
        if not page:
            return None
        return self.sections_by_page_anchor.get((page["url"], anchor))

    def output_file_for_source(self, source_id: str) -> Path:
        return self.file_by_record_id[source_id]


def _add_entity(
    entities: dict[str, dict],
    label: str,
    name: str,
    **props: object,
) -> dict:
    entity = _entity(label, name, **props)
    existing = entities.get(entity["id"])
    if existing:
        return existing
    entities[entity["id"]] = entity
    return entity


def _extract_tools(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    relationships: list[dict] = []
    for record in catalog.page_records("en/tools-reference.md"):
        if record["kind"] != "tablerow" or record.get("headers", [None])[0] != "tool":
            continue
        name = _first_code_or_text(record["cells"][0])
        if not name:
            continue
        entity = _add_entity(
            entities,
            "Tool",
            name,
            description=record["cells"][1] if len(record["cells"]) > 1 else "",
            permission_required=record["cells"][2] if len(record["cells"]) > 2 else "",
        )
        relationships.append(_relationship("DEFINES", record["id"], entity["id"]))
    return relationships


def _extract_settings(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    relationships: list[dict] = []
    for page_path in ("en/settings.md", "en/permissions.md"):
        for record in catalog.page_records(page_path):
            if record["kind"] != "tablerow" or not record.get("headers"):
                continue
            first_header = record["headers"][0]
            if first_header not in {"key", "setting"}:
                continue
            key = _first_code_or_text(record["cells"][0])
            if not key:
                continue
            entity = _add_entity(
                entities,
                "SettingKey",
                key,
                description=record["cells"][1] if len(record["cells"]) > 1 else "",
            )
            relationships.append(_relationship("DEFINES", record["section_id"], entity["id"]))
    return relationships


def _extract_hooks(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    relationships: list[dict] = []
    for record in catalog.page_records("en/hooks.md"):
        if (
            record["kind"] != "tablerow"
            or record.get("section_id", "").split("#")[-1] != "hook-lifecycle"
            or record.get("headers", [None])[0] != "event"
        ):
            continue
        name = _first_code_or_text(record["cells"][0])
        if not name:
            continue
        entity = _add_entity(
            entities,
            "Hook",
            name,
            description=record["cells"][1] if len(record["cells"]) > 1 else "",
        )
        relationships.append(_relationship("DEFINES", record["section_id"], entity["id"]))
    return relationships


def _extract_permission_modes(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    relationships: list[dict] = []
    for record in catalog.page_records("en/permission-modes.md"):
        if (
            record["kind"] != "tablerow"
            or record.get("section_id", "").split("#")[-1] != "available-modes"
            or record.get("headers", [None])[0] != "mode"
        ):
            continue
        name = _first_code_or_text(record["cells"][0])
        if not name:
            continue
        entity = _add_entity(
            entities,
            "PermissionMode",
            name,
            description=record["cells"][1] if len(record["cells"]) > 1 else "",
        )
        relationships.append(_relationship("DEFINES", record["section_id"], entity["id"]))
    return relationships


def _extract_message_types(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    section = catalog.section("en/agent-sdk/agent-loop.md", "message-types")
    if not section:
        return []
    relationships: list[dict] = []
    for name in MESSAGE_TYPES:
        entity = _add_entity(entities, "MessageType", name)
        relationships.append(_relationship("DEFINES", section["id"], entity["id"]))
    return relationships


def _extract_providers(catalog: Catalog, entities: dict[str, dict]) -> list[dict]:
    relationships: list[dict] = []
    for name, page_path in PROVIDERS.items():
        entity = _add_entity(entities, "Provider", name)
        source = catalog.first_section_for_page_path(page_path)
        if source:
            relationships.append(_relationship("DEFINES", source["id"], entity["id"]))
    return relationships


def build_entity_catalog(catalog: Catalog) -> tuple[dict[str, dict], list[dict]]:
    entities: dict[str, dict] = {}
    define_relationships: list[dict] = []
    define_relationships.extend(_extract_tools(catalog, entities))
    define_relationships.extend(_extract_settings(catalog, entities))
    define_relationships.extend(_extract_hooks(catalog, entities))
    define_relationships.extend(_extract_permission_modes(catalog, entities))
    define_relationships.extend(_extract_message_types(catalog, entities))
    define_relationships.extend(_extract_providers(catalog, entities))
    return entities, define_relationships


def _section_mentions(section: dict, entities: dict[str, dict]) -> list[dict]:
    text = "\n\n".join(
        part for part in (section.get("breadcrumb", ""), section.get("text", "")) if part
    )
    if not text:
        return []

    mentions: list[dict] = []
    for entity in entities.values():
        identifier = entity.get("key") if entity["label"] == "SettingKey" else entity["name"]
        if not identifier:
            continue
        if _mention_pattern(str(identifier)).search(text):
            mentions.append(
                _relationship(
                    "MENTIONS",
                    section["id"],
                    entity["id"],
                    evidence=str(identifier),
                )
            )
    return mentions


def enrich_records(records_by_file: dict[Path, list[dict]]) -> dict[Path, list[dict]]:
    catalog = Catalog(records_by_file)
    entities, define_relationships = build_entity_catalog(catalog)

    output: dict[Path, list[dict]] = {
        path: [dict(record) for record in records] for path, records in records_by_file.items()
    }
    output[Path("_entities.jsonl")] = list(entities.values())

    emitted_relationship_ids = {
        record["id"]
        for records in output.values()
        for record in records
        if record.get("kind") == "relationship" and "id" in record
    }

    for relationship in define_relationships:
        if relationship["id"] in emitted_relationship_ids:
            continue
        emitted_relationship_ids.add(relationship["id"])
        source_file = catalog.output_file_for_source(relationship["source_id"])
        output[source_file].append(relationship)

    for path, records in records_by_file.items():
        for section in (record for record in records if record["kind"] == "section"):
            for relationship in _section_mentions(section, entities):
                if relationship["id"] in emitted_relationship_ids:
                    continue
                emitted_relationship_ids.add(relationship["id"])
                output[path].append(relationship)

    return output


def load_records(root: Path) -> dict[Path, list[dict]]:
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL files found in {root}")
    return {path: _read_jsonl(path) for path in files}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 - extract closed entity catalog.")
    parser.add_argument("--in", dest="in_dir", default=str(RESOLVED_ROOT), help="Stage 2 JSONL dir.")
    parser.add_argument("--out", default=str(ENTITIES_ROOT), help="Entity-enriched JSONL output dir.")
    args = parser.parse_args(argv)

    in_root = Path(args.in_dir)
    out_root = Path(args.out)
    try:
        records_by_file = load_records(in_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if out_root.exists():
        shutil.rmtree(out_root)
    output = enrich_records(records_by_file)

    entity_count = 0
    define_count = 0
    mention_count = 0
    for relative_path, records in output.items():
        entity_count += sum(1 for record in records if record["kind"] == "entity")
        define_count += sum(
            1 for record in records if record["kind"] == "relationship" and record["type"] == "DEFINES"
        )
        mention_count += sum(
            1 for record in records if record["kind"] == "relationship" and record["type"] == "MENTIONS"
        )
        _write_jsonl(records, out_root / relative_path.name)

    print(
        f"extracted {entity_count} entities, {define_count} DEFINES, "
        f"{mention_count} MENTIONS -> {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
