import pytest
from src.routing import decide_route
from src.schemas import NodePayload


def section(raw_text, breadcrumb="Page > Sec", synthesized="false", node_id="s1"):
    return NodePayload(
        node_id=node_id, label="Section", raw_text=raw_text,
        breadcrumb=breadcrumb, extras={"synthesized": synthesized},
    )


def codeblock(raw_text, preceding="", node_id="c1"):
    return NodePayload(
        node_id=node_id, label="CodeBlock", raw_text=raw_text,
        breadcrumb="Page > Sec", extras={"preceding": preceding, "language": "python"},
    )


def ref_entity(label, node_id="e1"):
    return NodePayload(node_id=node_id, label=label, raw_text="desc",
                       extras={"name": node_id})


@pytest.mark.parametrize("payload,expected", [
    # --- Section: skip cases ---
    (section(""), "skip"),
    (section("   "), "skip"),
    (section("---"), "skip"),
    # --- Section: template cases ---
    (section("x", synthesized="true"), "template"),
    (section("A" * 201, "Page > A > B"), "template"),       # deep breadcrumb + long
    (section("A" * 201, synthesized="true"), "template"),   # synthesized wins even if long+deep
    # --- Section: llm cases ---
    (section("short text"), "llm"),
    (section("A" * 201, "Page > Sec"), "llm"),              # long but shallow breadcrumb (depth=2)
    (section("A" * 200, "Page > A > B"), "llm"),            # exactly 200 chars -> not > 200
    (section("A" * 201, "Page > A"), "llm"),                # long, depth=2 -> no template
    # --- CodeBlock: template cases ---
    (codeblock("X" * 1001, "P" * 201), "template"),
    # --- CodeBlock: llm — trivial labels treated as empty preceding ---
    (codeblock("X" * 1001, "Input:"), "llm"),
    (codeblock("X" * 1001, "Output:"), "llm"),
    (codeblock("X" * 1001, "Example:"), "llm"),
    (codeblock("X" * 1001, "Migration:"), "llm"),
    (codeblock("X" * 1001, "Windows PowerShell"), "llm"),
    (codeblock("X" * 1001, "Windows CMD"), "llm"),
    (codeblock("X" * 1001, "macOS"), "llm"),
    (codeblock("X" * 1001, "macOS, Linux, WSL"), "llm"),
    (codeblock("X" * 1001, "In code"), "llm"),
    (codeblock("X" * 1001, ".mcp.json"), "llm"),
    (codeblock("X" * 1001, "P" * 200), "llm"),             # preceding exactly 200 -> not > 200
    (codeblock("X" * 1000, "P" * 201), "llm"),             # code exactly 1000 -> not > 1000
    (codeblock("X" * 1001, ""), "llm"),
    # --- Reference entities: always llm ---
    (ref_entity("TableRow"), "llm"),
    (ref_entity("Callout"), "llm"),
    (ref_entity("Tool"), "llm"),
    (ref_entity("Hook"), "llm"),
    (ref_entity("SettingKey"), "llm"),
    (ref_entity("PermissionMode"), "llm"),
    (ref_entity("MessageType"), "llm"),
    (ref_entity("Provider"), "llm"),
])
def test_decide_route(payload, expected):
    assert decide_route(payload) == expected
