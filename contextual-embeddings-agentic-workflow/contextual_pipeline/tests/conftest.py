import pytest
from src.schemas import NodePayload


def make_section(
    raw_text: str = "Some section text",
    breadcrumb: str = "Page > Section",
    synthesized: str = "false",
    node_id: str = "s1",
) -> NodePayload:
    return NodePayload(
        node_id=node_id,
        label="Section",
        raw_text=raw_text,
        breadcrumb=breadcrumb,
        extras={"synthesized": synthesized},
    )


def make_codeblock(
    raw_text: str = "x = 1",
    preceding: str = "",
    language: str = "python",
    node_id: str = "c1",
) -> NodePayload:
    return NodePayload(
        node_id=node_id,
        label="CodeBlock",
        raw_text=raw_text,
        breadcrumb="Page > Section",
        extras={"preceding": preceding, "language": language},
    )
