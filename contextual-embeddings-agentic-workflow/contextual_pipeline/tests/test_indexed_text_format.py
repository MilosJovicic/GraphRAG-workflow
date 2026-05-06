from src.routing import build_raw_text_repr
from src.schemas import NodePayload


def build_indexed_text(context: str, raw_text_repr: str) -> str:
    return context + "\n\n" + raw_text_repr


def test_indexed_text_exact_format_section():
    p = NodePayload(node_id="s1", label="Section", raw_text="The text content.")
    ctx = "This section explains tool permissions."
    repr_ = build_raw_text_repr(p)
    result = build_indexed_text(ctx, repr_)
    assert result == "This section explains tool permissions.\n\nThe text content."


def test_indexed_text_exact_format_codeblock():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="npm install",
                    extras={"language": "bash"})
    ctx = "Install command for the project."
    repr_ = build_raw_text_repr(p)
    result = build_indexed_text(ctx, repr_)
    assert result == "Install command for the project.\n\n[bash]\nnpm install"


def test_indexed_text_separator_is_double_newline():
    p = NodePayload(node_id="s1", label="Section", raw_text="content")
    result = build_indexed_text("context", build_raw_text_repr(p))
    assert "\n\n" in result
    parts = result.split("\n\n", 1)
    assert parts[0] == "context"
    assert parts[1] == "content"
