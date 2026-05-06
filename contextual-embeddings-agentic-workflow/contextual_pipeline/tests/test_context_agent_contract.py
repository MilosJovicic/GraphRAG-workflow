import pytest
from pydantic import ValidationError

from src.agents import context_agent
from src.schemas import ContextResult, NodePayload


def test_entity_prompt_includes_definition_context_and_siblings():
    payload = NodePayload(
        node_id="default",
        label="PermissionMode",
        raw_text="Reads only",
        related_hints=["plan", "acceptEdits"],
        extras={
            "name": "default",
            "defined_in": "Choose a permission mode > Available modes",
            "definition_text": "Modes set the baseline. The table below shows each mode.",
        },
    )

    prompt = context_agent._build_user_prompt(payload)

    assert "default" in prompt
    assert "plan, acceptEdits" in prompt
    assert "Choose a permission mode > Available modes" in prompt
    assert "Modes set the baseline" in prompt


def test_entity_context_must_start_with_own_name():
    payload = NodePayload(
        node_id="default",
        label="PermissionMode",
        raw_text="Reads only",
        extras={"name": "default"},
    )
    result = ContextResult(context="PermissionMode: baseline read-only access.")

    with pytest.raises(ValidationError, match="must start with 'default:'"):
        context_agent._validate_context_for_payload(payload, result)


def test_entity_context_starting_with_name_passes_payload_validation():
    payload = NodePayload(
        node_id="default",
        label="PermissionMode",
        raw_text="Reads only",
        extras={"name": "default"},
    )
    result = ContextResult(context="default: baseline read-only access at session start.")

    assert context_agent._validate_context_for_payload(payload, result) is result


def test_codeblock_with_equivalent_hint_gets_cross_language_prefix():
    payload = NodePayload(
        node_id="cb1",
        label="CodeBlock",
        raw_text="print('hi')",
        related_hints=["typescript"],
    )
    result = ContextResult(context="Print call: Python statement emitting text.")

    validated = context_agent._validate_context_for_payload(payload, result)

    assert validated.context.startswith("Claude Code cross-language counterpart:")
    assert "Print call" in validated.context


def test_permissionmode_specific_semantic_requirements_are_enforced():
    default_payload = NodePayload(
        node_id="default",
        label="PermissionMode",
        raw_text="Reads only",
        related_hints=["plan", "acceptEdits"],
        extras={"name": "default"},
    )
    plan_payload = NodePayload(
        node_id="plan",
        label="PermissionMode",
        raw_text="Reads only",
        related_hints=["default", "acceptEdits"],
        extras={"name": "plan"},
    )

    with pytest.raises(ValidationError, match="baseline"):
        context_agent._validate_context_for_payload(
            default_payload,
            ContextResult(context="default: Reads only."),
        )

    with pytest.raises(ValidationError, match="planning"):
        context_agent._validate_context_for_payload(
            plan_payload,
            ContextResult(context="plan: Reads only."),
        )

    with pytest.raises(ValidationError, match="must not mention sibling mode"):
        context_agent._validate_context_for_payload(
            default_payload,
            ContextResult(
                context=(
                    "default: baseline read-only permission mode at session start. "
                    "Contrasts with plan."
                )
            ),
        )

    assert context_agent._validate_context_for_payload(
        default_payload,
        ContextResult(
            context="default: baseline read-only permission mode at session start."
        ),
    )
    assert context_agent._validate_context_for_payload(
        plan_payload,
        ContextResult(
            context=(
                "plan: EnterPlanMode read-only planning state for codebase "
                "exploration and approval-before-edit implementation proposals."
            )
        ),
    )
