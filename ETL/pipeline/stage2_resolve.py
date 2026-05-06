"""Stage 2 - resolve links and emit relationship records.

Input is Stage 1 JSONL under ./parsed/. Output is one JSONL per input page
under ./resolved/ containing copied node records plus relationship records.
Broken internal links are emitted as `unresolved_link` diagnostics and warned,
but they do not fail the stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlsplit

from .mdx_preprocess import slugify
from .stage1_parse import CANONICAL_URL_PREFIX, REPO_ROOT


PARSED_ROOT = REPO_ROOT / "parsed"
RESOLVED_ROOT = REPO_ROOT / "resolved"

PAGE_ALIASES = {
    "en/azure-ai-foundry.md": "en/microsoft-foundry.md",
}


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


def _relationship(
    rel_type: str,
    source_id: str,
    target_id: str,
    **props: object,
) -> dict:
    h = hashlib.sha256(
        f"{rel_type}|{source_id}|{target_id}|{props.get('href', '')}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "kind": "relationship",
        "id": h,
        "type": rel_type,
        "source_id": source_id,
        "target_id": target_id,
        **{k: v for k, v in props.items() if v not in (None, "", [])},
    }


class Catalog:
    def __init__(self, records_by_file: dict[Path, list[dict]]) -> None:
        records = [record for page_records in records_by_file.values() for record in page_records]
        self.pages_by_url = {
            record["url"]: record for record in records if record["kind"] == "page"
        }
        self.page_url_by_path = {
            record["path"]: record["url"] for record in records if record["kind"] == "page"
        }
        self.sections_by_id = {
            record["id"]: record for record in records if record["kind"] == "section"
        }
        self.sections_by_page: dict[str, list[dict]] = defaultdict(list)
        self.section_by_page_anchor: dict[tuple[str, str], dict] = {}
        self.sections_by_page_loose_anchor: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in records:
            if record["kind"] != "section":
                continue
            self.sections_by_page[record["page_url"]].append(record)
            self.section_by_page_anchor[(record["page_url"], record["anchor"])] = record
            self.sections_by_page_loose_anchor[
                (record["page_url"], _loose_anchor_key(record["anchor"]))
            ].append(record)
        for sections in self.sections_by_page.values():
            sections.sort(key=lambda section: section["position"])

    def first_section_for_page(self, page_url: str) -> dict | None:
        sections = self.sections_by_page.get(page_url, [])
        return sections[0] if sections else None

    def section_for_anchor(self, page_url: str, anchor: str) -> dict | None:
        candidates = [anchor, slugify(anchor)]
        if "-" in anchor:
            candidates.append(anchor.replace("-", ""))
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            section = self.section_by_page_anchor.get((page_url, candidate))
            if section:
                return section
        loose_matches = self.sections_by_page_loose_anchor.get(
            (page_url, _loose_anchor_key(anchor)),
            [],
        )
        if len(loose_matches) == 1:
            return loose_matches[0]
        return None


def _candidate_paths(raw_path: str, source_page_path: str) -> list[str]:
    path = unquote(raw_path).strip()
    if path.startswith("/docs/"):
        path = path.removeprefix("/docs/")
    elif path.startswith("/"):
        path = path[1:]
    elif path:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source_page_path), path))
    else:
        path = source_page_path

    if path.endswith(".md"):
        candidates = [path]
    else:
        candidates = [path + ".md", posixpath.join(path, "index.md")]
    out: list[str] = []
    for candidate in candidates:
        out.append(candidate)
        alias = PAGE_ALIASES.get(candidate)
        if alias:
            out.append(alias)
    return out


def _normalize_anchor(raw_anchor: str) -> str:
    return slugify(unquote(raw_anchor).strip().lstrip("#"))


def _clean_href_for_resolution(href: str) -> str:
    return unquote(href.strip()).strip("`").strip()


def _loose_anchor_key(anchor: str) -> str:
    return re.sub(r"[^a-z0-9]", "", anchor.lower())


def resolve_href(
    href: str,
    source_page_path: str,
    catalog: Catalog,
) -> dict:
    """Resolve one href into page/section targets or a diagnostic."""
    href = _clean_href_for_resolution(href)
    if not href:
        return {"status": "external"}

    parsed = urlsplit(href)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc != "code.claude.com" or not parsed.path.startswith("/docs/"):
            return {"status": "external", "url": href}
        raw_path = parsed.path
    elif parsed.scheme or href.startswith(("mailto:", "tel:", "javascript:")):
        return {"status": "external", "url": href}
    else:
        raw_path = parsed.path

    anchor = _normalize_anchor(parsed.fragment) if parsed.fragment else ""
    if anchor.startswith("ce-"):
        return {"status": "dom_anchor"}
    for candidate in _candidate_paths(raw_path, source_page_path):
        page_url = catalog.page_url_by_path.get(candidate)
        if not page_url:
            continue
        if anchor:
            section = catalog.section_for_anchor(page_url, anchor)
            if section:
                return {
                    "status": "section",
                    "page_url": page_url,
                    "section_id": section["id"],
                    "anchor": section["anchor"],
                }
            return {
                "status": "unresolved",
                "reason": "anchor",
                "page_url": page_url,
                "anchor": anchor,
            }
        return {"status": "page", "page_url": page_url}

    return {
        "status": "unresolved",
        "reason": "page",
        "path_candidates": _candidate_paths(raw_path, source_page_path),
        "anchor": anchor,
    }


def _copy_nodes_with_external_links(
    records: list[dict],
    external_links_by_section: dict[str, set[str]],
) -> list[dict]:
    out: list[dict] = []
    for record in records:
        if record["kind"] == "link":
            continue
        copied = dict(record)
        if copied["kind"] == "section":
            external = sorted(external_links_by_section.get(copied["id"], set()))
            copied["external_links"] = external
        out.append(copied)
    return out


def resolve_records(records_by_file: dict[Path, list[dict]]) -> dict[Path, list[dict]]:
    catalog = Catalog(records_by_file)
    external_links_by_section: dict[str, set[str]] = defaultdict(set)
    output_by_file: dict[Path, list[dict]] = {}
    emitted_relationship_ids: set[str] = set()

    # External links are section properties, so gather them before copying nodes.
    for records in records_by_file.values():
        page = next(record for record in records if record["kind"] == "page")
        for record in records:
            if record["kind"] != "link":
                continue
            resolved = resolve_href(record["href"], page["path"], catalog)
            if resolved["status"] == "external":
                external_links_by_section[record["section_id"]].add(record["href"])

    for path, records in records_by_file.items():
        page = next(record for record in records if record["kind"] == "page")
        out = _copy_nodes_with_external_links(records, external_links_by_section)

        for record in records:
            rel: dict | None = None
            kind = record["kind"]
            if kind == "section":
                if record.get("parent_section_id"):
                    rel = _relationship(
                        "HAS_SUBSECTION", record["parent_section_id"], record["id"]
                    )
                else:
                    rel = _relationship("HAS_SECTION", record["page_url"], record["id"])
            elif kind == "codeblock":
                rel = _relationship("CONTAINS_CODE", record["section_id"], record["id"])
            elif kind == "tablerow":
                rel = _relationship("CONTAINS_TABLE_ROW", record["section_id"], record["id"])
            elif kind == "callout":
                rel = _relationship("HAS_CALLOUT", record["section_id"], record["id"])
            if rel and rel["id"] not in emitted_relationship_ids:
                emitted_relationship_ids.add(rel["id"])
                out.append(rel)

        for record in records:
            if record["kind"] != "link":
                continue
            resolved = resolve_href(record["href"], page["path"], catalog)
            status = resolved["status"]
            source_kind = record.get("source_kind", "prose")
            if status in {"external", "dom_anchor"}:
                continue
            if status == "unresolved":
                out.append(
                    {
                        "kind": "unresolved_link",
                        "section_id": record["section_id"],
                        "href": record["href"],
                        "text": record["text"],
                        "reason": resolved.get("reason", "unknown"),
                        "target_page_url": resolved.get("page_url"),
                        "target_anchor": resolved.get("anchor"),
                    }
                )
                continue
            if status == "section" and record["section_id"] == resolved["section_id"]:
                continue

            if source_kind == "navigation_table":
                if status == "section":
                    target_id = resolved["section_id"]
                else:
                    target_section = catalog.first_section_for_page(resolved["page_url"])
                    if not target_section:
                        continue
                    target_id = target_section["id"]
                rel = _relationship(
                    "NAVIGATES_TO",
                    record["section_id"],
                    target_id,
                    href=record["href"],
                    text=record["text"],
                )
            elif status == "section":
                rel = _relationship(
                    "LINKS_TO",
                    record["section_id"],
                    resolved["section_id"],
                    href=record["href"],
                    text=record["text"],
                )
            else:
                rel = _relationship(
                    "LINKS_TO_PAGE",
                    record["section_id"],
                    resolved["page_url"],
                    href=record["href"],
                    text=record["text"],
                )
            if rel["id"] not in emitted_relationship_ids:
                emitted_relationship_ids.add(rel["id"])
                out.append(rel)

        codegroups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            if record["kind"] == "codeblock" and record.get("codegroup_id"):
                codegroups[record["codegroup_id"]].append(record)
        for members in codegroups.values():
            members.sort(key=lambda member: member["position"])
            for idx, left in enumerate(members):
                for right in members[idx + 1 :]:
                    if left.get("language") == right.get("language"):
                        continue
                    rel = _relationship("EQUIVALENT_TO", left["id"], right["id"])
                    if rel["id"] not in emitted_relationship_ids:
                        emitted_relationship_ids.add(rel["id"])
                        out.append(rel)

        output_by_file[path] = out

    return output_by_file


def load_records(root: Path) -> dict[Path, list[dict]]:
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL files found in {root}")
    return {path: _read_jsonl(path) for path in files}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 - resolve links and relationships.")
    parser.add_argument("--in", dest="in_dir", default=str(PARSED_ROOT), help="Stage 1 JSONL dir.")
    parser.add_argument("--out", default=str(RESOLVED_ROOT), help="Resolved JSONL output dir.")
    args = parser.parse_args(argv)

    in_root = Path(args.in_dir)
    out_root = Path(args.out)
    try:
        records_by_file = load_records(in_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_by_file = resolve_records(records_by_file)
    relationship_count = 0
    unresolved_count = 0
    for in_path, records in output_by_file.items():
        relationship_count += sum(1 for record in records if record["kind"] == "relationship")
        unresolved_count += sum(1 for record in records if record["kind"] == "unresolved_link")
        _write_jsonl(records, out_root / in_path.name)

    print(
        f"resolved {len(output_by_file)} pages -> {relationship_count} relationships "
        f"-> {out_root}"
    )
    if unresolved_count:
        print(f"warning: {unresolved_count} internal links did not resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
