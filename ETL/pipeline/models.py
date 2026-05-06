"""Data classes for Stage 1 parser output.

Each Page becomes one JSONL file under ./parsed/. Inside, one record per
node: page, section, codeblock, tablerow, callout, link.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


NodeKind = Literal["page", "section", "codeblock", "tablerow", "callout", "link"]
LinkSourceKind = Literal["prose", "table", "navigation_table"]


@dataclass
class Page:
    kind: Literal["page"]
    url: str                  # canonical URL, e.g. https://code.claude.com/docs/en/quickstart.md
    path: str                 # corpus-relative path, e.g. en/quickstart.md
    title: str                # H1 text
    description: str          # from llms.txt; fallback = post-H1 blockquote
    description_source: Literal["llms.txt", "blockquote", "missing"]
    content_hash: str         # sha256(file body)
    volatile: bool            # True for whats-new/* and changelog.md


@dataclass
class Section:
    kind: Literal["section"]
    id: str                   # "{page_url}#{anchor}"
    page_url: str
    anchor: str               # slugified heading
    breadcrumb: str           # "PageTitle > H2 > H3"
    level: int                # heading level, 2 through 6
    position: int             # 0-indexed within page
    text: str                 # prose only; code/tables/callouts stripped
    synthesized: bool = False   # True for <Update> cards in changelog/whats-new
    parent_section_id: str | None = None  # for H3/H4 nesting
    external_links: list[str] = field(default_factory=list)


@dataclass
class CodeBlock:
    kind: Literal["codeblock"]
    id: str
    section_id: str
    position: int
    language: str             # e.g. "python", "bash", "" if unfenced
    text: str                 # original whitespace preserved
    preceding_paragraph: str  # cached prefix input for Stage 4
    in_codegroup: bool        # member of a <CodeGroup>
    codegroup_id: str | None  # shared group id, for EQUIVALENT_TO pairing


@dataclass
class TableRow:
    kind: Literal["tablerow"]
    id: str
    section_id: str
    position: int             # 0-indexed within the table
    table_position: int       # which table within the section
    headers: list[str]        # column headers (lowercased, trimmed)
    cells: list[str]          # cell text in column order


@dataclass
class Callout:
    kind: Literal["callout"]
    id: str
    section_id: str
    position: int
    callout_kind: str         # "note" | "tip" | "info" | "warning" | "check"
    text: str


@dataclass
class Link:
    kind: Literal["link"]
    section_id: str
    href: str                 # raw href as written
    text: str                 # link text
    is_internal: bool         # /en/... vs external
    is_anchor_only: bool      # leading '#'
    source_kind: LinkSourceKind = "prose"


def to_dict(obj: Any) -> dict:
    return asdict(obj)
