"""MDX -> GFM preprocessor.

Implements the strip/unwrap/structural rules from CLAUDE.md "Source flavor".
Output is plain GFM markdown that markdown-it-py can parse, plus a small set
of HTML-comment sentinels that the Stage 1 parser recognises:

    <!--callout:note-->...<!--/callout-->
    <!--codegroup:N-->                    (placed before each member fence)
    <!--update-start anchor="week-17" description="..." tags="...":-->
    ...
    <!--update-end-->
    <!--image data-path="...":-->         (replaces <img> tags)

The preprocessor must run before AST parsing. Raw JSX must never leak into
prose.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Pass 1 — strip executable MDX
# ---------------------------------------------------------------------------

# Top-level `export const NAME = ...` and `export default ...`. These open
# with `export ` at column 0 and run to a balanced closing brace at column 0.
# Bodies routinely contain JSX, template literals, fetch() calls, useEffect,
# etc. We use brace counting (skipping strings, comments, and template-literal
# `${}` expressions) to find the close.
#
# Mintlify pages may also include self-closing script tags for client-side
# helpers. Strip them before markdown-it sees them; otherwise an unclosed
# `<script />` HTML block can swallow the rest of the document.

_EXPORT_RE = re.compile(r"^export\s+(const|let|var|function|default)\b", re.MULTILINE)


def _skip_string(src: str, i: int, quote: str) -> int:
    """Return index just past a string literal that started at src[i] == quote."""
    i += 1
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        if quote == "`" and c == "$" and i + 1 < len(src) and src[i + 1] == "{":
            # Template literal interpolation. Recurse on `{...}`.
            i = _skip_braces(src, i + 1)
            continue
        i += 1
    return i


def _skip_braces(src: str, i: int) -> int:
    """src[i] must be '{'. Return index just past the matching '}'."""
    assert src[i] == "{"
    depth = 0
    while i < len(src):
        c = src[i]
        if c in "\"'`":
            i = _skip_string(src, i, c)
            continue
        if c == "/" and i + 1 < len(src) and src[i + 1] == "/":
            nl = src.find("\n", i)
            i = nl + 1 if nl != -1 else len(src)
            continue
        if c == "/" and i + 1 < len(src) and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = end + 2 if end != -1 else len(src)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _skip_balanced(src: str, i: int, open_char: str, close_char: str) -> int:
    """Return index just past a balanced pair, skipping strings/comments."""
    assert src[i] == open_char
    depth = 0
    while i < len(src):
        c = src[i]
        if c in "\"'`":
            i = _skip_string(src, i, c)
            continue
        if c == "/" and i + 1 < len(src) and src[i + 1] == "/":
            nl = src.find("\n", i)
            i = nl + 1 if nl != -1 else len(src)
            continue
        if c == "/" and i + 1 < len(src) and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = end + 2 if end != -1 else len(src)
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _find_export_body_brace(src: str, start: int) -> int:
    """Find the executable body brace for a top-level export statement."""
    paragraph_end = src.find("\n\n", start)
    header_end = paragraph_end if paragraph_end != -1 else len(src)

    arrow = src.find("=>", start, header_end)
    if arrow != -1:
        return src.find("{", arrow, len(src))

    paren = src.find("(", start, header_end)
    if paren != -1:
        after_params = _skip_balanced(src, paren, "(", ")")
        body = src.find("{", after_params, len(src))
        if body != -1:
            return body

    assignment = src.find("=", start, header_end)
    if assignment != -1:
        return src.find("{", assignment, len(src))

    return src.find("{", start, header_end)


def _find_toplevel_export_close(src: str, start: int) -> int:
    m = re.search(r"(?m)^};?\s*$", src[start:])
    if not m:
        return -1
    line_end = src.find("\n", start + m.end())
    return line_end + 1 if line_end != -1 else start + m.end()


def _strip_export_blocks(src: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(src):
        m = _EXPORT_RE.search(src, i)
        if not m:
            out.append(src[i:])
            break
        out.append(src[i : m.start()])
        # Find the first `{` after the export to skip over an arrow-function or
        # object body. If the export has no body (e.g. `export default Foo;`),
        # fall back to next semicolon or newline.
        body_brace = _find_export_body_brace(src, m.start())
        if body_brace == -1:
            # No body — strip to the end of the statement.
            semi = src.find(";", m.start())
            nl = src.find("\n", m.start())
            end = min(x for x in (semi, nl, len(src)) if x != -1)
            i = end + 1
            continue
        end = _skip_braces(src, body_brace)
        fallback_end = _find_toplevel_export_close(src, body_brace)
        if fallback_end != -1 and (end == len(src) or fallback_end < end):
            end = fallback_end
        # Also consume any trailing semicolon and newline.
        if end < len(src) and src[end] == ";":
            end += 1
        if end < len(src) and src[end] == "\n":
            end += 1
        i = end
    return "".join(out)


_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_SCRIPT_BLOCK_RE = re.compile(
    r"<script\b[^>]*(?:/>\s*|>.*?</script\s*>)",
    re.DOTALL | re.IGNORECASE,
)


def _strip_style_blocks(src: str) -> str:
    return _STYLE_BLOCK_RE.sub("", src)


def _strip_script_blocks(src: str) -> str:
    return _SCRIPT_BLOCK_RE.sub("", src)


_TOPLEVEL_BRACE_LINE_RE = re.compile(r"^\{[^{}\n]*\}\s*$", re.MULTILINE)
_TOPLEVEL_BRACE_BLOCK_RE = re.compile(r"^\{(?:.|\n)*?\}\s*$", re.MULTILINE)


def _strip_toplevel_jsx_expressions(src: str) -> str:
    """Drop block-level `{expr}` lines that aren't inside a fence or tag.

    Conservative: only matches lines that start with `{` at column 0 and where
    the braces balance within the same logical block.
    """
    out_lines: list[str] = []
    in_fence = False
    fence_marker = ""
    i_lines = src.split("\n")
    i = 0
    while i < len(i_lines):
        line = i_lines[i]
        stripped = line.lstrip()
        # Track fenced code blocks so we don't strip inside them.
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            out_lines.append(line)
            i += 1
            continue
        if in_fence:
            out_lines.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
            i += 1
            continue
        if line.startswith("{"):
            # Find the matching close brace across lines.
            joined = line
            depth = line.count("{") - line.count("}")
            j = i
            while depth > 0 and j + 1 < len(i_lines):
                j += 1
                joined += "\n" + i_lines[j]
                depth += i_lines[j].count("{") - i_lines[j].count("}")
            if depth == 0:
                # Skip the whole block.
                i = j + 1
                continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Pass 2 — JSX tag tokenisation
# ---------------------------------------------------------------------------
#
# We need a small JSX tag scanner that respects fenced code blocks (so we
# don't molest tags inside `````jsx` examples) and string literals inside
# attribute values.

_TAG_OPEN_RE = re.compile(r"<([A-Za-z][A-Za-z0-9]*)\b")


@dataclass
class _Tag:
    name: str
    attrs: dict
    self_closing: bool
    start: int  # index in source where '<' lives
    end: int    # index just past '>'


def _parse_attrs(raw: str) -> dict:
    """Parse a raw attribute string like `title="X" foo={bar}` into a dict.

    JSX-style `{expr}` values are kept as the literal expression text (without
    braces). Boolean attrs (no '=') get value True.
    """
    attrs: dict = {}
    i = 0
    n = len(raw)
    while i < n:
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break
        # Attribute name.
        m = re.match(r"[A-Za-z_:][\w:.\-]*", raw[i:])
        if not m:
            i += 1
            continue
        name = m.group(0)
        i += len(name)
        if i >= n or raw[i] != "=":
            attrs[name] = True
            continue
        i += 1  # past '='
        if i >= n:
            break
        if raw[i] in "\"'":
            quote = raw[i]
            j = raw.find(quote, i + 1)
            if j == -1:
                attrs[name] = raw[i + 1 :]
                break
            attrs[name] = raw[i + 1 : j]
            i = j + 1
        elif raw[i] == "{":
            j = _skip_braces(raw, i)
            attrs[name] = raw[i + 1 : j - 1].strip()
            i = j
        else:
            m = re.match(r"\S+", raw[i:])
            if m:
                attrs[name] = m.group(0)
                i += len(m.group(0))
    return attrs


def _scan_tag(src: str, i: int) -> _Tag | None:
    """If src[i:] starts a JSX open tag, return _Tag describing it."""
    m = _TAG_OPEN_RE.match(src, i)
    if not m:
        return None
    name = m.group(1)
    j = m.end()
    # Walk forward, skipping string and JSX-expression attributes.
    while j < len(src):
        c = src[j]
        if c in "\"'":
            j = _skip_string(src, j, c)
            continue
        if c == "{":
            j = _skip_braces(src, j)
            continue
        if c == "/" and j + 1 < len(src) and src[j + 1] == ">":
            attrs_raw = src[m.end() : j]
            return _Tag(name, _parse_attrs(attrs_raw), True, i, j + 2)
        if c == ">":
            attrs_raw = src[m.end() : j]
            return _Tag(name, _parse_attrs(attrs_raw), False, i, j + 1)
        j += 1
    return None


def _find_close_tag(src: str, start: int, name: str) -> int:
    """Find index of matching `</name>` starting search at `start`.

    Respects nesting. Returns the index of the '<' or -1 if not found. Skips
    fenced code blocks and string literals inside attributes.
    """
    depth = 1
    open_re = re.compile(r"<" + re.escape(name) + r"\b")
    close_re = re.compile(r"</" + re.escape(name) + r"\s*>")
    i = start
    while i < len(src):
        # Skip fenced code blocks.
        if src.startswith("```", i):
            end = src.find("```", i + 3)
            if end == -1:
                return -1
            i = end + 3
            continue
        m_open = open_re.search(src, i)
        m_close = close_re.search(src, i)
        if not m_close:
            return -1
        if m_open and m_open.start() < m_close.start():
            # Need to skip past this open tag (could be self-closing).
            t = _scan_tag(src, m_open.start())
            if t is None:
                i = m_open.end()
                continue
            if not t.self_closing:
                depth += 1
            i = t.end
            continue
        depth -= 1
        if depth == 0:
            return m_close.start()
        i = m_close.end()
    return -1


# ---------------------------------------------------------------------------
# Pass 3 — Tag transforms (strip / unwrap / sentinel)
# ---------------------------------------------------------------------------

_STRIP_TAGS = {"Experiment", "Frame"}  # Frame: keep inner content (img)
_DROP_WITH_BODY_TAGS = {"Hero"}        # purely presentational

_UNWRAP_NO_TITLE = {
    "Tabs", "AccordionGroup", "CardGroup", "Steps",
    "RequestExample", "ResponseExample",
}

# title-bearing unwraps; title becomes a bold line
_UNWRAP_WITH_TITLE = {
    "Tab": "title",
    "Accordion": "title",
    "Card": "title",
    "Expandable": "title",
    "Step": "title",
    "ParamField": None,        # synthesised from path/type/required
    "ResponseField": None,
}

_CALLOUT_TAGS = {"Note", "Tip", "Info", "Warning", "Check"}

_HTML_TEXT_TAGS = {"p", "span"}
_HTML_STRONG_TAGS = {"strong", "b"}
_HTML_EM_TAGS = {"em", "i"}


def _render_paramfield_lead(attrs: dict) -> str:
    name = attrs.get("path") or attrs.get("name") or attrs.get("body") or ""
    typ = attrs.get("type", "")
    required = attrs.get("required") in (True, "true", "")
    parts = [f"`{name}`"] if name else []
    if typ:
        parts.append(f"({typ})")
    if required:
        parts.append("**required**")
    return "**" + " ".join(parts) + "**" if parts else ""


def _transform_tags(src: str) -> str:
    """Walk the source once, applying tag transforms.

    We rebuild into `out`. For each `<Tag>` we recognise we either:
      - strip the entire tag and its body
      - replace with sentinel-wrapped inner text
      - replace with a title line + inner text
    Unrecognised tags are left as-is for the parser to surface as raw HTML
    (and for check_parse to flag if they leak into prose).
    """
    out: list[str] = []
    i = 0
    codegroup_counter = 0

    while i < len(src):
        # Skip fenced code blocks verbatim.
        if src.startswith("```", i):
            end = src.find("```", i + 3)
            if end == -1:
                out.append(src[i:])
                break
            out.append(src[i : end + 3])
            i = end + 3
            continue

        if src[i] != "<":
            out.append(src[i])
            i += 1
            continue

        tag = _scan_tag(src, i)
        if tag is None:
            out.append(src[i])
            i += 1
            continue

        name = tag.name

        # ---- self-closing structural tags ----
        if tag.self_closing:
            if name == "img":
                data_path = tag.attrs.get("data-path") or tag.attrs.get("src", "")
                alt = tag.attrs.get("alt", "")
                out.append(f"<!--image data-path={_q(data_path)} alt={_q(alt)}-->")
                i = tag.end
                continue
            if name in _STRIP_TAGS or name in _DROP_WITH_BODY_TAGS:
                i = tag.end
                continue
            if name in _UNWRAP_WITH_TITLE:
                # Self-closing ParamField/ResponseField with no body — emit lead only.
                lead = _render_paramfield_lead(tag.attrs) if name in {"ParamField", "ResponseField"} else ""
                if lead:
                    out.append("\n\n" + lead + "\n\n")
                i = tag.end
                continue
            # Unknown self-closing tag: leave it.
            out.append(src[i : tag.end])
            i = tag.end
            continue

        # ---- paired tags: find close ----
        close_at = _find_close_tag(src, tag.end, name)
        if close_at == -1:
            # Unmatched: leave the open tag and move past it.
            out.append(src[i : tag.end])
            i = tag.end
            continue
        body = src[tag.end : close_at]
        end_of_close = src.find(">", close_at) + 1

        if name in _STRIP_TAGS:
            # Strip wrapper, keep body (recursively transformed).
            out.append(_transform_tags(_dedent_mdx_body(body)))
            i = end_of_close
            continue

        if name in _DROP_WITH_BODY_TAGS:
            i = end_of_close
            continue

        if name == "div":
            # Strip MDX-only div wrappers; recurse into body.
            slot = tag.attrs.get("data-gb-slot") or tag.attrs.get("className", "")
            if slot or "install-configurator-slot" in (tag.attrs.get("className") or ""):
                out.append(_transform_tags(_dedent_mdx_body(body)))
            else:
                # Generic div: unwrap, recurse.
                out.append(_transform_tags(_dedent_mdx_body(body)))
            i = end_of_close
            continue

        if name in _CALLOUT_TAGS:
            kind = name.lower()
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            out.append(f"\n\n<!--callout:{kind}-->\n{inner}\n<!--/callout-->\n\n")
            i = end_of_close
            continue

        if name == "CodeGroup":
            codegroup_counter += 1
            inner = _transform_tags(_dedent_mdx_body(body))
            inner = _tag_codegroup_fences(inner, codegroup_counter)
            out.append("\n\n" + inner + "\n\n")
            i = end_of_close
            continue

        if name == "Update":
            anchor = _slugify(tag.attrs.get("label", f"update-{i}"))
            description = tag.attrs.get("description", "")
            tags_attr = tag.attrs.get("tags", "")
            label = tag.attrs.get("label", "")
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            out.append(
                f"\n\n<!--update-start anchor={_q(anchor)} label={_q(label)} "
                f"description={_q(description)} tags={_q(tags_attr)}-->\n"
                f"## {label}\n\n{inner}\n"
                f"<!--update-end-->\n\n"
            )
            i = end_of_close
            continue

        if name in _UNWRAP_NO_TITLE:
            out.append(_transform_tags(_dedent_mdx_body(body)))
            i = end_of_close
            continue

        if name in _UNWRAP_WITH_TITLE:
            attr = _UNWRAP_WITH_TITLE[name]
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            if name in {"ParamField", "ResponseField"}:
                lead = _render_paramfield_lead(tag.attrs)
            else:
                title = tag.attrs.get(attr, "") if attr else ""
                lead = f"**{title}**" if title else ""
            if lead and inner:
                out.append(f"\n\n{lead}\n\n{inner}\n\n")
            elif lead:
                out.append(f"\n\n{lead}\n\n")
            else:
                out.append(_indent_for_step(name, inner))
            i = end_of_close
            continue

        if name in _HTML_TEXT_TAGS:
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            if name == "p":
                out.append(f"\n\n{inner}\n\n" if inner else "")
            else:
                out.append(inner)
            i = end_of_close
            continue

        if name in _HTML_STRONG_TAGS:
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            out.append(f"**{inner}**" if inner else "")
            i = end_of_close
            continue

        if name in _HTML_EM_TAGS:
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            out.append(f"*{inner}*" if inner else "")
            i = end_of_close
            continue

        if name == "code":
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            out.append(f"`{inner}`" if inner else "")
            i = end_of_close
            continue

        if name == "a":
            inner = _transform_tags(_dedent_mdx_body(body)).strip()
            href = tag.attrs.get("href", "")
            out.append(f"[{inner}]({href})" if href and inner else inner)
            i = end_of_close
            continue

        # Unknown tag: leave as-is, recurse into body.
        out.append(src[i : tag.end])
        out.append(_transform_tags(_dedent_mdx_body(body)))
        out.append(src[close_at:end_of_close])
        i = end_of_close

    return "".join(out)


def _indent_for_step(name: str, inner: str) -> str:
    if name == "Step":
        return "\n\n1. " + inner.replace("\n", "\n   ") + "\n\n"
    return "\n\n" + inner + "\n\n"


def _dedent_mdx_body(body: str) -> str:
    """Remove wrapper indentation introduced by MDX component nesting."""
    return textwrap.dedent(body).strip("\n")


def _tag_codegroup_fences(src: str, codegroup_id: int) -> str:
    """Place a sentinel before each opening fence in a CodeGroup body."""
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence_marker = stripped[:3]
            in_fence = True
            out.append(f"<!--codegroup:{codegroup_id}-->\n")
            out.append(stripped)
            continue
        if in_fence and stripped.startswith(fence_marker):
            in_fence = False
            out.append(stripped)
            continue
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Pass 4 — code-fence theme metadata
# ---------------------------------------------------------------------------

_FENCE_INFO_RE = re.compile(r"^(```|~~~)([^\n]*)$", re.MULTILINE)


def _strip_fence_metadata(src: str) -> str:
    """Strip `theme={...}` and any second-language label on fence info strings.

    Examples:
        ```python Python theme={null}     ->  ```python
        ```bash theme={null}              ->  ```bash
        ```js javascript                  ->  ```js
    """
    def repl(m: re.Match) -> str:
        marker = m.group(1)
        info = m.group(2).strip()
        if not info:
            return marker
        # Drop theme={...}.
        info = re.sub(r"\btheme=\{[^}]*\}", "", info).strip()
        # First whitespace-separated token is the language.
        tokens = info.split()
        return marker + (tokens[0] if tokens else "")
    return _FENCE_INFO_RE.sub(repl, src)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(value: object) -> str:
    """Quote a value for an HTML-comment sentinel."""
    s = str(value).replace('"', "&quot;").replace("\n", " ")
    return f'"{s}"'


_SLUG_DROP_RE = re.compile(r"[^\w\s-]")
_SLUG_SPACE_RE = re.compile(r"[\s_-]+")


def slugify(text: str) -> str:
    """Mintlify-style heading slug: lowercase, spaces/underscores -> '-', strip punctuation."""
    s = _SLUG_DROP_RE.sub("", text.lower())
    s = _SLUG_SPACE_RE.sub("-", s).strip("-")
    return s


_slugify = slugify


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess(source: str) -> str:
    """Run all preprocessing passes."""
    s = source
    s = _strip_export_blocks(s)
    s = _strip_style_blocks(s)
    s = _strip_script_blocks(s)
    s = _strip_toplevel_jsx_expressions(s)
    s = _transform_tags(s)
    s = _strip_fence_metadata(s)
    # Collapse 3+ newlines to 2 to keep the AST tidy.
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s
