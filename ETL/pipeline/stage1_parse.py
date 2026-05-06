"""Stage 1 — parse Mintlify MDX corpus into per-page JSONL.

Run:
    python -m pipeline.stage1_parse                  # full corpus
    python -m pipeline.stage1_parse en/quickstart.md # single file (relative to ./documents/)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.front_matter import front_matter_plugin

from .mdx_preprocess import preprocess, slugify
from .models import Callout, CodeBlock, Link, LinkSourceKind, Page, Section, TableRow


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "documents"
EN_ROOT = DOCS_ROOT / "en"
PARSED_ROOT = REPO_ROOT / "parsed"
LLMS_TXT = DOCS_ROOT / "llms.txt"

CANONICAL_URL_PREFIX = "https://code.claude.com/docs/"
VOLATILE_PATTERNS = ("en/whats-new/", "en/changelog.md")


# ---------------------------------------------------------------------------
# llms.txt loader
# ---------------------------------------------------------------------------

_LLMS_LINE_RE = re.compile(r"^- \[([^\]]+)\]\(([^)]+)\): (.+)$")


def load_llms_descriptions() -> dict[str, str]:
    """Return {canonical_url: description}. Empty if llms.txt is missing."""
    if not LLMS_TXT.exists():
        return {}
    out: dict[str, str] = {}
    for line in LLMS_TXT.read_text(encoding="utf-8").splitlines():
        m = _LLMS_LINE_RE.match(line.strip())
        if m:
            _, url, desc = m.groups()
            out[url] = desc.strip()
    return out


# ---------------------------------------------------------------------------
# Path -> URL
# ---------------------------------------------------------------------------

def path_to_url(rel_path: str) -> str:
    """en/quickstart.md -> https://code.claude.com/docs/en/quickstart.md"""
    return CANONICAL_URL_PREFIX + rel_path.replace("\\", "/")


# ---------------------------------------------------------------------------
# Markdown engine
# ---------------------------------------------------------------------------

def make_md() -> MarkdownIt:
    md = MarkdownIt("gfm-like", {"html": True}).enable("table").use(front_matter_plugin)
    return md


# ---------------------------------------------------------------------------
# Token-stream helpers
# ---------------------------------------------------------------------------

def _inline_to_text(tok: Token) -> str:
    """Concatenate text from an inline token's children, dropping link wrappers
    but keeping their text. Sentinel HTML comments are stripped."""
    parts: list[str] = []
    for child in tok.children or []:
        t = child.type
        if t == "text":
            parts.append(child.content)
        elif t == "code_inline":
            parts.append(f"`{child.content}`")
        elif t == "softbreak" or t == "hardbreak":
            parts.append(" ")
        elif t == "link_open":
            pass
        elif t == "link_close":
            pass
        elif t == "html_inline":
            # Drop any stray inline HTML.
            continue
        elif t == "image":
            alt = child.attrs.get("alt", "") if hasattr(child, "attrs") else ""
            parts.append(alt or "")
        else:
            if child.content:
                parts.append(child.content)
    return "".join(parts)


def _inline_links(tok: Token) -> Iterator[tuple[str, str]]:
    """Yield (href, text) for every link inside an inline token."""
    if not tok.children:
        return
    href = None
    buf: list[str] = []
    for child in tok.children:
        if child.type == "link_open":
            href = child.attrs.get("href", "") if hasattr(child, "attrs") else ""
            buf = []
        elif child.type == "link_close":
            if href is not None:
                yield href, "".join(buf)
            href = None
            buf = []
        elif href is not None:
            if child.type == "text":
                buf.append(child.content)
            elif child.type == "code_inline":
                buf.append(f"`{child.content}`")


# ---------------------------------------------------------------------------
# Sentinel-comment regexes
# ---------------------------------------------------------------------------

_CALLOUT_OPEN_RE = re.compile(r"<!--callout:([a-z]+)-->")
_CALLOUT_CLOSE_RE = re.compile(r"<!--/callout-->")
_UPDATE_OPEN_RE = re.compile(
    r'<!--update-start\s+anchor="([^"]*)"\s+label="([^"]*)"\s+'
    r'description="([^"]*)"\s+tags="([^"]*)"\s*-->'
)
_UPDATE_CLOSE_RE = re.compile(r"<!--update-end-->")
_CODEGROUP_RE = re.compile(r"<!--codegroup:(\d+)-->")
_IMAGE_RE = re.compile(r'<!--image data-path="([^"]*)" alt="([^"]*)"-->')


# ---------------------------------------------------------------------------
# Per-page parse
# ---------------------------------------------------------------------------

def parse_file(rel_path: str, raw: str, llms_desc: dict[str, str]) -> list[dict]:
    """Parse one corpus file. Returns a flat list of node dicts ready for
    JSONL. The first record is always the Page node."""
    url = path_to_url(rel_path)
    norm_path = rel_path.replace("\\", "/")
    is_volatile = any(norm_path.startswith(p) or norm_path == p for p in VOLATILE_PATTERNS)

    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    pre = preprocess(raw)

    md = make_md()
    tokens = md.parse(pre)

    # ---- extract H1 + post-H1 blockquote for fallback description ----
    h1_text = ""
    h1_idx = None
    for idx, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag == "h1":
            h1_idx = idx
            inline = tokens[idx + 1]
            h1_text = _inline_to_text(inline).strip()
            break

    blockquote_desc = ""
    if h1_idx is not None:
        # Look for the next blockquote_open within 4 tokens.
        for idx in range(h1_idx + 3, min(h1_idx + 12, len(tokens))):
            tok = tokens[idx]
            if tok.type == "blockquote_open":
                # Walk children for paragraph text.
                inner = []
                j = idx + 1
                while j < len(tokens) and tokens[j].type != "blockquote_close":
                    if tokens[j].type == "inline":
                        inner.append(_inline_to_text(tokens[j]))
                    j += 1
                blockquote_desc = " ".join(inner).strip()
                break
            if tok.type == "heading_open":
                break

    desc = llms_desc.get(url, "")
    if desc:
        desc_source = "llms.txt"
    elif blockquote_desc:
        desc, desc_source = blockquote_desc, "blockquote"
    else:
        desc_source = "missing"

    page = Page(
        kind="page",
        url=url,
        path=norm_path,
        title=h1_text or norm_path,
        description=desc,
        description_source=desc_source,
        content_hash=content_hash,
        volatile=is_volatile,
    )

    # ---- walk tokens, emitting Sections + child nodes ----
    records: list[dict] = [asdict(page)]
    walker = _SectionWalker(
        page_url=url,
        page_title=page.title,
        volatile=is_volatile,
    )
    walker.walk(tokens)
    records.extend(asdict(r) for r in walker.records)
    return records


class _SectionWalker:
    """Stateful walker over a markdown-it token stream.

    Maintains a section stack so child sections can reference parents, and a
    text buffer per section that excludes code/tables/callouts (those become
    their own atomic-block records)."""

    def __init__(self, page_url: str, page_title: str, volatile: bool) -> None:
        self.page_url = page_url
        self.page_title = page_title
        self.volatile = volatile
        self.records: list = []  # Section | CodeBlock | TableRow | Callout | Link
        self._section_stack: list[Section] = []  # H2/H3/H4 in order
        self._buf: list[str] = []                # current section prose
        self._section_counter = 0
        self._anchor_counts: dict[str, int] = {}
        self._in_update = False
        self._update_meta: dict = {}
        self._suppress_paragraph_emit = False    # while inside a callout

    # --- section open/close ---
    def _flush_buf_to_current(self) -> None:
        if not self._section_stack:
            self._buf.clear()
            return
        cur = self._section_stack[-1]
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(self._buf).strip())
        # Append, since walkers re-enter sections via close+reopen siblings.
        cur.text = (cur.text + "\n\n" + text).strip() if cur.text else text
        self._buf.clear()

    def _close_to_level(self, level: int) -> None:
        """Close all open sections with .level >= level."""
        self._flush_buf_to_current()
        while self._section_stack and self._section_stack[-1].level >= level:
            closed = self._section_stack.pop()
            self.records.append(closed)

    def _open_section(self, level: int, anchor: str, heading_text: str, synthesized: bool = False) -> Section:
        # Build breadcrumb from ancestors.
        ancestors = [self.page_title] + [s.heading_text for s in self._section_stack if s.level < level]
        ancestors.append(heading_text)
        breadcrumb = " > ".join(ancestors)
        sec_id = f"{self.page_url}#{anchor}"
        parent = self._section_stack[-1].id if self._section_stack else None
        sec = Section(
            kind="section",
            id=sec_id,
            page_url=self.page_url,
            anchor=anchor,
            breadcrumb=breadcrumb,
            level=level,
            position=self._section_counter,
            text="",
            synthesized=synthesized,
            parent_section_id=parent,
        )
        # Stash heading text for breadcrumb assembly of descendants.
        sec.heading_text = heading_text  # type: ignore[attr-defined]
        self._section_counter += 1
        self._section_stack.append(sec)
        return sec

    def _unique_anchor(self, anchor: str) -> str:
        """Return a page-unique anchor, suffixing repeated headings."""
        base = anchor or "section"
        count = self._anchor_counts.get(base, 0)
        self._anchor_counts[base] = count + 1
        if count == 0:
            return base
        return f"{base}-{count}"

    def _ensure_overview_section(self) -> None:
        """Create a synthetic H2 parent for top-level atomic content."""
        if self._section_stack:
            return
        self._open_section(2, self._unique_anchor("overview"), "Overview", synthesized=False)

    # --- main walk ---
    def walk(self, tokens: list[Token]) -> None:
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            t = tok.type

            if t == "html_block":
                i = self._handle_html_block(tokens, i)
                continue

            if t == "heading_open":
                level = int(tok.tag[1])
                inline = tokens[i + 1]
                heading_text = _inline_to_text(inline).strip()
                if level == 1:
                    i += 3  # heading_open, inline, heading_close
                    continue
                if 2 <= level <= 6:
                    self._close_to_level(level)
                    anchor = self._unique_anchor(slugify(heading_text))
                    synth = self._in_update and level == 2
                    self._open_section(level, anchor, heading_text, synthesized=synth)
                    if synth:
                        # Stamp update metadata onto the new section.
                        sec = self._section_stack[-1]
                        sec.text = ""
                        # Cache description in the prose buffer; the description
                        # itself is on Page-level for whats-new/index.md, but we
                        # surface it on the synthesized section text too.
                        if self._update_meta.get("description"):
                            self._buf.append(self._update_meta["description"])
                    i += 3
                    continue
                i += 3
                continue

            if t == "fence":
                self._ensure_overview_section()
                self._handle_fence(tok)
                i += 1
                continue

            if t == "code_block":
                # Indented code block.
                self._ensure_overview_section()
                self._handle_indented_code(tok)
                i += 1
                continue

            if t == "table_open":
                end = self._find_close(tokens, i, "table_close")
                self._handle_table(tokens[i : end + 1])
                i = end + 1
                continue

            if t == "blockquote_open":
                self._ensure_overview_section()
                end = self._find_close(tokens, i, "blockquote_close")
                # Treat as prose; recurse via inline tokens.
                inner_text = []
                for j in range(i + 1, end):
                    if tokens[j].type == "inline":
                        inner_text.append(_inline_to_text(tokens[j]))
                if inner_text:
                    self._buf.append("> " + " ".join(inner_text))
                i = end + 1
                continue

            if t == "paragraph_open":
                self._ensure_overview_section()
                inline = tokens[i + 1]
                text = _inline_to_text(inline).strip()
                if text and not self._suppress_paragraph_emit:
                    self._buf.append(text)
                self._collect_links(inline)
                i += 3
                continue

            if t in ("bullet_list_open", "ordered_list_open"):
                self._ensure_overview_section()
                end = self._find_close(tokens, i, t.replace("_open", "_close"))
                items = self._render_list(tokens[i : end + 1])
                if items and not self._suppress_paragraph_emit:
                    self._buf.append(items)
                i = end + 1
                continue

            if t == "hr":
                self._ensure_overview_section()
                self._buf.append("---")
                i += 1
                continue

            i += 1

        self._close_to_level(0)  # close everything
        self._drop_empty_leaf_sections()
        if not any(isinstance(record, Section) for record in self.records):
            self.records.append(
                Section(
                    kind="section",
                    id=f"{self.page_url}#overview",
                    page_url=self.page_url,
                    anchor="overview",
                    breadcrumb=f"{self.page_title} > Overview",
                    level=2,
                    position=self._section_counter,
                    text="",
                )
            )

    def _drop_empty_leaf_sections(self) -> None:
        used_section_ids: set[str] = set()
        for record in self.records:
            if isinstance(record, Section):
                if record.parent_section_id:
                    used_section_ids.add(record.parent_section_id)
                continue
            if isinstance(record, (CodeBlock, TableRow, Callout, Link)):
                used_section_ids.add(record.section_id)
        self.records = [
            record
            for record in self.records
            if not (
                isinstance(record, Section)
                and not record.text.strip()
                and record.id not in used_section_ids
            )
        ]

    # --- handlers ---
    def _handle_html_block(self, tokens: list[Token], i: int) -> int:
        content = tokens[i].content
        # Update card start.
        m = _UPDATE_OPEN_RE.search(content)
        if m:
            anchor, label, description, tags = m.groups()
            self._in_update = True
            self._update_meta = {
                "anchor": anchor, "label": label,
                "description": description, "tags": tags,
            }
            return i + 1
        if _UPDATE_CLOSE_RE.search(content):
            self._in_update = False
            self._update_meta = {}
            return i + 1
        # Callout open.
        m = _CALLOUT_OPEN_RE.search(content)
        if m:
            self._ensure_overview_section()
            kind = m.group(1)
            return self._handle_callout(tokens, i, kind)
        # Image sentinel.
        m = _IMAGE_RE.search(content)
        if m:
            self._ensure_overview_section()
            data_path, alt = m.groups()
            self._buf.append(f"[image: {alt}]({data_path})")
            return i + 1
        # codegroup sentinel: drop quietly; the next fence carries the group id.
        m = _CODEGROUP_RE.search(content)
        if m:
            self._next_codegroup_id = int(m.group(1))
            return i + 1
        # Other html_block: ignore.
        return i + 1

    def _handle_callout(self, tokens: list[Token], i: int, kind: str) -> int:
        """Walk forward until we see a `<!--/callout-->` html_block, gathering
        prose into a Callout node and suppressing it from the section buffer."""
        prose: list[str] = []
        j = i + 1
        while j < len(tokens):
            tj = tokens[j]
            if tj.type == "html_block" and _CALLOUT_CLOSE_RE.search(tj.content):
                break
            if tj.type == "paragraph_open":
                inline = tokens[j + 1]
                prose.append(_inline_to_text(inline).strip())
                self._collect_links(inline)
                j += 3
                continue
            if tj.type in ("bullet_list_open", "ordered_list_open"):
                end = self._find_close(tokens, j, tj.type.replace("_open", "_close"))
                prose.append(self._render_list(tokens[j : end + 1]))
                j = end + 1
                continue
            j += 1
        text = "\n\n".join(p for p in prose if p)
        if self._section_stack:
            sec = self._section_stack[-1]
            position = sum(1 for r in self.records if isinstance(r, Callout) and r.section_id == sec.id)
            cid = _atomic_id(sec.id, position, "callout-" + kind)
            self.records.append(
                Callout(kind="callout", id=cid, section_id=sec.id,
                        position=position, callout_kind=kind, text=text)
            )
        return j + 1

    def _handle_fence(self, tok: Token) -> None:
        if not self._section_stack:
            self._ensure_overview_section()
        sec = self._section_stack[-1]
        position = sum(1 for r in self.records if isinstance(r, CodeBlock) and r.section_id == sec.id)
        language = (tok.info or "").strip().split()[0] if tok.info else ""
        # Use last sentinel-set codegroup id, then clear it.
        codegroup_id = getattr(self, "_next_codegroup_id", None)
        in_codegroup = codegroup_id is not None
        if in_codegroup:
            self._next_codegroup_id = None
        # Preceding paragraph: last buffered prose, if any.
        preceding = self._buf[-1] if self._buf else ""
        cid = _atomic_id(sec.id, position, "code-" + language)
        self.records.append(
            CodeBlock(
                kind="codeblock",
                id=cid,
                section_id=sec.id,
                position=position,
                language=language,
                text=tok.content.rstrip("\n"),
                preceding_paragraph=preceding,
                in_codegroup=in_codegroup,
                codegroup_id=f"{sec.id}::cg{codegroup_id}" if codegroup_id else None,
            )
        )

    def _handle_indented_code(self, tok: Token) -> None:
        # Treat indented code as a fence with no language.
        if not self._section_stack:
            self._ensure_overview_section()
        sec = self._section_stack[-1]
        text = tok.content.rstrip("\n")
        if _looks_like_prose_code_block(text):
            self._buf.append(text)
            return
        position = sum(1 for r in self.records if isinstance(r, CodeBlock) and r.section_id == sec.id)
        preceding = self._buf[-1] if self._buf else ""
        cid = _atomic_id(sec.id, position, "code-")
        self.records.append(
            CodeBlock(
                kind="codeblock", id=cid, section_id=sec.id, position=position,
                language="", text=text,
                preceding_paragraph=preceding, in_codegroup=False, codegroup_id=None,
            )
        )

    def _handle_table(self, table_toks: list[Token]) -> None:
        if not self._section_stack:
            self._ensure_overview_section()
        sec = self._section_stack[-1]
        # Locate header cells and body rows.
        headers: list[str] = []
        rows: list[list[str]] = []
        cell_inlines: list[Token] = []
        in_thead, in_tbody, current_row = False, False, []
        for tok in table_toks:
            if tok.type == "thead_open":
                in_thead = True
            elif tok.type == "thead_close":
                in_thead = False
            elif tok.type == "tbody_open":
                in_tbody = True
            elif tok.type == "tbody_close":
                in_tbody = False
            elif tok.type == "tr_open":
                current_row = []
            elif tok.type == "tr_close":
                if in_thead:
                    headers = current_row[:]
                elif in_tbody:
                    rows.append(current_row[:])
            elif tok.type in ("th_open", "td_open"):
                pass
            elif tok.type in ("th_close", "td_close"):
                pass
            elif tok.type == "inline":
                current_row.append(_inline_to_text(tok).strip())
                cell_inlines.append(tok)

        # Classify: navigation table = predominantly link-only cells in body.
        link_cells = 0
        total_cells = 0
        for row in rows:
            for cell in row:
                total_cells += 1
                # Heuristic: cell is "link-only" if it begins with [...] and
                # contains no plain words besides the link text. We re-scan the
                # original raw markdown via the tokens later; here approximate
                # with `cell.startswith("`)` already lost — use the inline scan.
        # Simpler classification: count cells whose plain text equals what's
        # inside one [text](href) pair. We look back at the inline tokens.
        # For now: treat any table where the first column header is exactly
        # "Tool" / "Setting" / "Event" as a data table and emit rows.
        is_navigation = _looks_like_nav_table(headers, table_toks)
        table_index = sum(1 for r in self.records if isinstance(r, TableRow) and r.section_id == sec.id and r.position == 0)

        source_kind: LinkSourceKind = "navigation_table" if is_navigation else "table"
        for inline in cell_inlines:
            self._collect_links(inline, source_kind=source_kind)

        if is_navigation:
            # Skip emitting TableRows; would emit NAVIGATES_TO at Stage 2.
            table_text = _render_table_text(headers, rows)
            if table_text:
                self._buf.append(table_text)
            return

        for row_pos, row in enumerate(rows):
            tid = _atomic_id(sec.id, row_pos + 1000 * table_index, "row")
            self.records.append(
                TableRow(
                    kind="tablerow", id=tid, section_id=sec.id,
                    position=row_pos, table_position=table_index,
                    headers=[h.lower() for h in headers], cells=row,
                )
            )

    def _render_list(self, list_toks: list[Token]) -> str:
        """Render a bullet/ordered list to a flat markdown string for the buffer."""
        ordered = list_toks[0].type == "ordered_list_open"
        out: list[str] = []
        depth = 0
        item_idx = 0
        for tok in list_toks:
            if tok.type in ("bullet_list_open", "ordered_list_open"):
                depth += 1
            elif tok.type in ("bullet_list_close", "ordered_list_close"):
                depth -= 1
            elif tok.type == "list_item_open":
                pass
            elif tok.type == "list_item_close":
                item_idx += 1
            elif tok.type == "inline":
                prefix = "  " * (depth - 1) + (f"{item_idx + 1}. " if ordered else "- ")
                out.append(prefix + _inline_to_text(tok).strip())
                self._collect_links(tok)
        return "\n".join(out)

    def _collect_links(self, inline: Token, source_kind: LinkSourceKind = "prose") -> None:
        if not self._section_stack:
            return
        sec = self._section_stack[-1]
        for href, text in _inline_links(inline):
            is_internal = href.startswith("/en/") or href.startswith("./") or href.startswith("../")
            is_anchor_only = href.startswith("#")
            self.records.append(
                Link(
                    kind="link", section_id=sec.id, href=href, text=text,
                    is_internal=is_internal, is_anchor_only=is_anchor_only,
                    source_kind=source_kind,
                )
            )

    @staticmethod
    def _find_close(tokens: list[Token], start: int, close_type: str) -> int:
        """Find matching close token (same nesting level, naive depth count)."""
        open_type = close_type.replace("_close", "_open")
        depth = 1
        for j in range(start + 1, len(tokens)):
            if tokens[j].type == open_type:
                depth += 1
            elif tokens[j].type == close_type:
                depth -= 1
                if depth == 0:
                    return j
        return len(tokens) - 1


def _looks_like_nav_table(headers: list[str], table_toks: list[Token]) -> bool:
    """True if body cells are predominantly navigational links.

    Re-scans inline children instead of rendered cell text so regular prose
    tables with occasional links remain data tables, while reference matrices
    with link-heavy cells become NAVIGATES_TO edges.
    """
    normalized_headers = [h.strip().lower() for h in headers]
    data_table_first_headers = {
        "tool",
        "key",
        "event",
        "mode",
        "setting",
        "field",
        "parameter",
        "rule",
    }
    if normalized_headers and normalized_headers[0] in data_table_first_headers:
        return False

    body_cells = 0
    linked_cells = 0
    in_tbody = False
    for tok in table_toks:
        if tok.type == "tbody_open":
            in_tbody = True
        elif tok.type == "tbody_close":
            in_tbody = False
        elif in_tbody and tok.type == "inline":
            body_cells += 1
            children = tok.children or []
            link_count = 0
            for child in children:
                if child.type == "link_open":
                    link_count += 1
            if link_count >= 1:
                linked_cells += 1
    if body_cells == 0:
        return False
    return linked_cells / body_cells >= 0.5


def _render_table_text(headers: list[str], rows: list[list[str]]) -> str:
    rendered_rows: list[str] = []
    for row in rows:
        parts: list[str] = []
        for idx, cell in enumerate(row):
            if not cell:
                continue
            header = headers[idx] if idx < len(headers) else ""
            parts.append(f"{header}: {cell}" if header else cell)
        if parts:
            rendered_rows.append("- " + "; ".join(parts))
    return "\n".join(rendered_rows)


def _looks_like_prose_code_block(text: str) -> bool:
    """True for indented list continuations that markdown parsed as code."""
    stripped = text.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    code_prefixes = ("/", "$", ">", "{", "[", "|", "#", "<")
    if any(line.startswith(code_prefixes) for line in lines[:2]):
        return False
    prose_lines = [
        line
        for line in lines
        if len(line.split()) >= 5 and re.search(r"[A-Za-z]", line)
    ]
    if len(prose_lines) / len(lines) < 0.6:
        return False
    return any(re.search(r"[.!?](?:\s|$)", line) for line in prose_lines)


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

def _atomic_id(parent_section_id: str, position: int, kind_suffix: str) -> str:
    h = hashlib.sha256(f"{parent_section_id}|{position}|{kind_suffix}".encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_corpus_files(root: Path) -> Iterator[Path]:
    yield from sorted(p for p in root.rglob("*.md") if p.is_file())


def parse_one(rel_path: str, llms_desc: dict[str, str]) -> list[dict]:
    abs_path = DOCS_ROOT / rel_path
    raw = abs_path.read_text(encoding="utf-8")
    return parse_file(rel_path, raw, llms_desc)


def slug_for_path(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace("\\", "__").removesuffix(".md")


def write_jsonl(records: Iterable[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 — parse corpus to JSONL.")
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Optional path under documents/ (e.g. en/quickstart.md). "
             "If omitted, parse the full corpus.",
    )
    parser.add_argument(
        "--out", default=str(PARSED_ROOT),
        help="Output directory for JSONL files.",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    llms_desc = load_llms_descriptions()

    targets: list[Path]
    if args.target:
        candidate = (DOCS_ROOT / args.target).resolve()
        if not candidate.exists():
            print(f"error: {candidate} does not exist", file=sys.stderr)
            return 2
        targets = [candidate]
    else:
        targets = list(iter_corpus_files(EN_ROOT))

    n_pages, n_nodes = 0, 0
    for path in targets:
        rel = path.relative_to(DOCS_ROOT).as_posix()
        records = parse_one(rel, llms_desc)
        write_jsonl(records, out_root / (slug_for_path(rel) + ".jsonl"))
        n_pages += 1
        n_nodes += len(records)

    print(f"parsed {n_pages} pages -> {n_nodes} nodes -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
