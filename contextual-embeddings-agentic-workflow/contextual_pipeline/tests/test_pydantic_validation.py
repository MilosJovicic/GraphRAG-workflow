import pytest
from pydantic import ValidationError
from src.schemas import ContextResult, NodePayload


# ---------- length / shape ----------

def test_context_result_valid():
    r = ContextResult(context="default: baseline permission mode granting read-only filesystem access.")
    assert r.context.startswith("default:")


def test_context_result_too_short():
    with pytest.raises(ValidationError):
        ContextResult(context="Too short")


def test_context_result_empty():
    with pytest.raises(ValidationError):
        ContextResult(context="")


def test_context_result_too_long():
    with pytest.raises(ValidationError):
        ContextResult(context="A" * 601)


def test_context_result_exactly_600():
    # 600 As is a degenerate case but length-only — must pass length check.
    # The boilerplate validator would not flag it, so this stays valid.
    r = ContextResult(context="A" * 600)
    assert len(r.context) == 600


# ---------- v2 boilerplate validator: first-word rejections ----------

@pytest.mark.parametrize("opener", [
    "This section explains tool permissions in Claude Code.",
    "this codeblock illustrates the retry policy used by tenacity.",
    "The node represents a permission row in the settings table.",
    "the section discusses how to configure hooks for tool calls.",
    "A node defining the default permission mode for sessions.",
    "An entry describing an environment variable used by the runtime.",
])
def test_context_rejects_meta_opener(opener):
    with pytest.raises(ValidationError) as exc:
        ContextResult(context=opener)
    assert "must not start" in str(exc.value).lower() or "forbidden" in str(exc.value).lower()


# ---------- v2 boilerplate validator: forbidden phrase rejections ----------

@pytest.mark.parametrize("phrase_ctx", [
    "default: baseline permission mode, distinct from related hints in adjacent rows.",
    "Bash: tool that runs shell commands. This node defines the canonical permission entry.",
    "PostToolUse: hook event firing after a tool call. This section discusses ordering.",
    "WarningCallout: surfaces a security caveat. This callout highlights bypass risk.",
    "default: baseline mode, in the documentation under Permissions.",
    "plan: read-only mode within the documentation, contrasting with auto.",
])
def test_context_rejects_forbidden_phrases(phrase_ctx):
    with pytest.raises(ValidationError) as exc:
        ContextResult(context=phrase_ctx)
    msg = str(exc.value).lower()
    assert "forbidden" in msg or "must not" in msg


# ---------- v2 boilerplate validator: things that should still pass ----------

@pytest.mark.parametrize("good_ctx", [
    "default: baseline permission mode granting read-only filesystem access. Contrasts with plan and acceptEdits.",
    "Bash: shell-command tool requiring permission, paired with the auto-approve allowedTools pattern.",
    "PostToolUse: hook event firing after a tool call succeeds. Counterpart to PostToolUseFailure.",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME: env var overriding the Opus model picker.",
    "Adds the cron expression `7 * * * *` to schedule a recurring agent every hour at minute 7.",
    "Cosine similarity in embedding space between sibling enum entries; sentence avoids forbidden boilerplate.",
])
def test_context_accepts_clean_outputs(good_ctx):
    r = ContextResult(context=good_ctx)
    assert r.context == good_ctx


def test_context_with_inline_this_is_ok():
    """'this' mid-sentence is fine; only leading-word + meta-phrases are rejected."""
    r = ContextResult(context="default: baseline mode used when entering a session via this command.")
    assert "this command" in r.context


def test_backticked_single_letter_identifier_is_allowed():
    r = ContextResult(context="`a`: Vim NORMAL-mode append command listed in mode switching.")
    assert r.context.startswith("`a`:")


# ---------- NodePayload (unchanged) ----------

def test_node_payload_defaults():
    p = NodePayload(node_id="x", label="Section", raw_text="hello")
    assert p.tags == []
    assert p.related_hints == []
    assert p.extras == {}


def test_node_payload_extras_stored():
    p = NodePayload(
        node_id="x", label="CodeBlock", raw_text="code",
        extras={"language": "python", "preceding": "Some text"}
    )
    assert p.extras["language"] == "python"
