from temporalio import activity
from temporalio.exceptions import ApplicationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from ..schemas import NodePayload, ContextResult
from ..agents.context_agent import generate_context as _agent_generate


class _RetryableError(Exception):
    pass


def _is_wrapped_output_validation_error(exc: UnexpectedModelBehavior) -> bool:
    msg = str(exc).lower()
    return (
        "validation" in msg
        or "validate" in msg
        or "validator" in msg
        or "retry" in msg
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RetryableError),
    reraise=True,
)
async def _call_with_retry(payload: NodePayload) -> ContextResult:
    try:
        return await _agent_generate(payload)
    except ValidationError as exc:
        raise _RetryableError(f"Schema validation failed: {exc}") from exc
    except UnexpectedModelBehavior as exc:
        if _is_wrapped_output_validation_error(exc):
            raise _RetryableError(f"Output validation failed: {exc}") from exc
        raise ApplicationError(
            f"Non-retryable LLM behavior for {payload.node_id}: {exc}",
            non_retryable=True,
        ) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("connection", "timeout", "502", "503", "429")):
            raise _RetryableError(f"Transient error: {exc}") from exc
        raise ApplicationError(
            f"Non-retryable LLM error for {payload.node_id}: {exc}",
            non_retryable=True,
        ) from exc


@activity.defn
async def generate_context(payload: NodePayload) -> ContextResult:
    try:
        return await _call_with_retry(payload)
    except _RetryableError as exc:
        raise ApplicationError(
            f"LLM failed after 3 attempts for {payload.node_id}: {exc}",
            non_retryable=True,
        ) from exc
