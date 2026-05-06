from src.routing import template_section, template_codeblock, build_raw_text_repr
from src.schemas import NodePayload


def _section(extras=None, **kw):
    return NodePayload(
        node_id="s1", label="Section",
        raw_text=kw.get("raw_text", "Some text"),
        page_title=kw.get("page_title", "My Page"),
        breadcrumb=kw.get("breadcrumb", "My Page > Intro > Detail"),
        parent_text=kw.get("parent_text", "Parent paragraph."),
        extras=extras or {},
    )


def test_template_section_synthesized_contains_changelog():
    p = _section(extras={"synthesized": "true"},
                 breadcrumb="Changelog > 2.1.122")
    result = template_section(p)
    assert "changelog entry" in result
    assert "2.1.122" in result


def test_template_section_deterministic():
    p = _section(breadcrumb="My Page > Intro > Detail", raw_text="A" * 250)
    assert template_section(p) == template_section(p)


def test_template_section_contains_breadcrumb_tail():
    p = _section(breadcrumb="My Page > Intro > Detail")
    result = template_section(p)
    assert "Detail" in result


def test_template_section_contains_parent_breadcrumb():
    p = _section(breadcrumb="My Page > Intro > Detail")
    result = template_section(p)
    assert "My Page > Intro" in result


def test_template_codeblock_contains_language():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": "Install dependencies first."},
    )
    result = template_codeblock(p)
    assert "python" in result


def test_template_codeblock_deterministic():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": "Some preceding text."},
    )
    assert template_codeblock(p) == template_codeblock(p)


def test_template_codeblock_contains_breadcrumb():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": ""},
    )
    result = template_codeblock(p)
    assert "Page > Sec" in result


# --- build_raw_text_repr ---

def test_repr_section():
    p = NodePayload(node_id="s1", label="Section", raw_text="Hello world")
    assert build_raw_text_repr(p) == "Hello world"


def test_repr_codeblock():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="x=1",
                    extras={"language": "python"})
    assert build_raw_text_repr(p) == "[python]\nx=1"


def test_repr_codeblock_no_language_defaults_code():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="x=1", extras={})
    assert build_raw_text_repr(p).startswith("[code]")


def test_repr_tablerow():
    p = NodePayload(node_id="t1", label="TableRow", raw_text="",
                    extras={"headers": "Name|Value", "cells": "foo|bar"})
    result = build_raw_text_repr(p)
    assert "Name | Value" in result
    assert "foo | bar" in result


def test_repr_callout():
    p = NodePayload(node_id="cal1", label="Callout", raw_text="Be careful here.",
                    extras={"callout_kind": "warning"})
    result = build_raw_text_repr(p)
    assert "[warning]" in result
    assert "Be careful here." in result


def test_repr_tool():
    p = NodePayload(node_id="Bash", label="Tool", raw_text="Runs shell commands.",
                    extras={"name": "Bash"})
    result = build_raw_text_repr(p)
    assert result == "Bash: Runs shell commands."
