"""Integration test for the validator -> tenacity retry chain.

Confirms that when the LLM produces a context that fails the new
ContextResult.reject_boilerplate validator, the activity retries (up
to 3 attempts) and surfaces a clean output if any attempt succeeds,
else raises a non-retryable ApplicationError.
"""
import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from temporalio.exceptions import ApplicationError

from src.schemas import ContextResult, NodePayload
from src.activities import generate_context as gc_module


def _payload() -> NodePayload:
    return NodePayload(
        node_id="t1",
        label="PermissionMode",
        raw_text="Reads only",
        breadcrumb="Settings > Permissions",
        extras={"name": "default"},
    )


def _make_validation_error() -> ValidationError:
    """Construct a real ValidationError by tripping the boilerplate rule."""
    try:
        ContextResult(context="This node defines the default permission mode.")
    except ValidationError as exc:
        return exc
    raise RuntimeError("expected ValidationError")


@pytest.mark.asyncio
async def test_retry_succeeds_after_one_validation_failure(monkeypatch):
    calls = {"n": 0}

    async def fake_agent_generate(_payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_validation_error()
        return ContextResult(context="default: baseline read-only permission mode at session start.")

    monkeypatch.setattr(gc_module, "_agent_generate", fake_agent_generate)
    # Bypass tenacity's exponential backoff so the test is fast.
    monkeypatch.setattr(gc_module._call_with_retry.retry, "wait", lambda *a, **kw: 0)

    result = await gc_module.generate_context(_payload())
    assert result.context.startswith("default:")
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_exhausts_after_three_validation_failures(monkeypatch):
    calls = {"n": 0}

    async def always_fails(_payload):
        calls["n"] += 1
        raise _make_validation_error()

    monkeypatch.setattr(gc_module, "_agent_generate", always_fails)
    monkeypatch.setattr(gc_module._call_with_retry.retry, "wait", lambda *a, **kw: 0)

    with pytest.raises(ApplicationError) as exc_info:
        await gc_module.generate_context(_payload())

    assert exc_info.value.non_retryable is True
    assert "after 3 attempts" in str(exc_info.value).lower() or "3" in str(exc_info.value)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_non_retryable_error_does_not_retry(monkeypatch):
    """Non-transient, non-validation errors must surface immediately as
    non_retryable=True without consuming retry budget."""
    calls = {"n": 0}

    async def boom(_payload):
        calls["n"] += 1
        raise RuntimeError("API key is malformed")  # not a transient signal

    monkeypatch.setattr(gc_module, "_agent_generate", boom)
    monkeypatch.setattr(gc_module._call_with_retry.retry, "wait", lambda *a, **kw: 0)

    with pytest.raises(ApplicationError) as exc_info:
        await gc_module.generate_context(_payload())

    assert exc_info.value.non_retryable is True
    assert calls["n"] == 1  # no retries on non-retryable error


@pytest.mark.asyncio
async def test_wrapped_output_validation_error_retries(monkeypatch):
    """PydanticAI wraps exhausted output validation in UnexpectedModelBehavior;
    that must still consume the outer retry budget."""
    calls = {"n": 0}

    async def fails_then_succeeds(_payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UnexpectedModelBehavior("Output validation failed: meta opener")
        return ContextResult(context="default: baseline read-only permission mode at session start.")

    monkeypatch.setattr(gc_module, "_agent_generate", fails_then_succeeds)
    monkeypatch.setattr(gc_module._call_with_retry.retry, "wait", lambda *a, **kw: 0)

    result = await gc_module.generate_context(_payload())

    assert result.context.startswith("default:")
    assert calls["n"] == 2
