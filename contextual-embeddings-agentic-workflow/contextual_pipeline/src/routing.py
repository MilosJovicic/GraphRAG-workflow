from typing import Literal
from .schemas import NodePayload

TRIVIAL_LABELS = {
    "Input:", "Output:", "Example:", "Migration:",
    "Windows PowerShell", "Windows CMD", "macOS",
    "macOS, Linux, WSL", "In code", ".mcp.json",
}


def decide_route(payload: NodePayload) -> Literal["skip", "template", "llm"]:
    if payload.label == "Section":
        if not payload.raw_text or payload.raw_text.strip() in {"", "---"}:
            return "skip"
        if payload.extras.get("synthesized") == "true":
            return "template"
        depth = len((payload.breadcrumb or "").split(" > "))
        if len(payload.raw_text) > 200 and depth >= 3:
            return "template"
        return "llm"

    if payload.label == "CodeBlock":
        pp = payload.extras.get("preceding", "") or ""
        if pp in TRIVIAL_LABELS:
            pp = ""
        if len(payload.raw_text) > 1000 and len(pp) > 200:
            return "template"
        return "llm"

    if payload.label in {
        "TableRow", "Callout", "Tool", "Hook",
        "SettingKey", "PermissionMode", "MessageType", "Provider",
    }:
        return "llm"

    return "skip"


def build_raw_text_repr(payload: NodePayload) -> str:
    if payload.label == "Section":
        return payload.raw_text

    if payload.label == "CodeBlock":
        lang = payload.extras.get("language", "code")
        return f"[{lang}]\n{payload.raw_text}"

    if payload.label == "TableRow":
        headers = payload.extras.get("headers", "").split("|")
        cells = payload.extras.get("cells", "").split("|")
        return f"{' | '.join(headers)}\n{' | '.join(cells)}"

    if payload.label == "Callout":
        kind = payload.extras.get("callout_kind", "note")
        return f"[{kind}] {payload.raw_text}"

    # Tool, Hook, SettingKey, PermissionMode, MessageType, Provider
    name = payload.extras.get("name", payload.node_id)
    return f"{name}: {payload.raw_text}"


def template_section(p: NodePayload) -> str:
    crumb = p.breadcrumb or ""
    if p.extras.get("synthesized") == "true":
        tail = crumb.split(" > ")[-1]
        return f"[changelog entry] Release notes for {tail}."
    tail = crumb.split(" > ")[-1]
    parent_crumb = crumb.rsplit(" > ", 1)[0] if " > " in crumb else crumb
    parent = (p.parent_text or "")[:120].strip()
    base = f"[{p.page_title} — {tail}] Subsection of \"{parent_crumb}\"."
    return f"{base} {parent}".strip()


def template_codeblock(p: NodePayload) -> str:
    lang = p.extras.get("language", "code")
    pp = (p.extras.get("preceding", "") or "")[:200]
    return f"[{lang}: example] In {p.breadcrumb}. {pp}".strip()


def templater_for(label: str):
    if label == "Section":
        return template_section
    if label == "CodeBlock":
        return template_codeblock
    raise ValueError(f"No template for label: {label}")
