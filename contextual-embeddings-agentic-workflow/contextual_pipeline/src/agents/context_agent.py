from pathlib import Path
import re
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from ..schemas import NodePayload, ContextResult
from ..config import get_settings


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_PROMPT_FILE_BY_LABEL: dict[str, str] = {
    "Tool":           "entity_prompt.txt",
    "Hook":           "entity_prompt.txt",
    "SettingKey":     "entity_prompt.txt",
    "PermissionMode": "entity_prompt.txt",
    "MessageType":    "entity_prompt.txt",
    "Provider":       "entity_prompt.txt",
    "TableRow":       "tablerow_prompt.txt",
    "Callout":        "callout_prompt.txt",
    "Section":        "section_prompt.txt",
    "CodeBlock":      "codeblock_prompt.txt",
}

_PROMPT_TEMPLATES: dict[str, str] = {
    label: (_PROMPTS_DIR / fname).read_text(encoding="utf-8")
    for label, fname in _PROMPT_FILE_BY_LABEL.items()
}

_settings = get_settings()
_provider = OpenAIProvider(api_key=_settings.openai_api_key)
_model = OpenAIModel(_settings.llm_model, provider=_provider)

agent = Agent(_model, output_type=ContextResult)


_ENTITY_LABELS = {"Tool", "Hook", "SettingKey", "PermissionMode", "MessageType", "Provider"}
_CODEBLOCK_EQUIVALENT_PREFIX = "Claude Code cross-language counterpart: "


def _context_validation_error(message: str, context: str) -> ValidationError:
    return ValidationError.from_exception_data(
        "ContextResult",
        [
            {
                "type": "value_error",
                "loc": ("context",),
                "input": context,
                "ctx": {"error": ValueError(message)},
            }
        ],
    )


def _entity_name(payload: NodePayload) -> str:
    return payload.extras.get("name", payload.node_id)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _mentions_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text) is not None


def _prefix_equivalent_codeblock_context(result: ContextResult) -> ContextResult:
    if result.context.startswith(_CODEBLOCK_EQUIVALENT_PREFIX):
        return result

    context = _CODEBLOCK_EQUIVALENT_PREFIX + result.context
    if len(context) > 600:
        context = context[:600].rstrip()
        if not context.endswith((".", "!", "?")):
            context = context[:599].rstrip() + "."
    return ContextResult(context=context)


def _validate_context_for_payload(
    payload: NodePayload,
    result: ContextResult,
) -> ContextResult:
    if payload.label == "CodeBlock" and payload.related_hints:
        return _prefix_equivalent_codeblock_context(result)

    if payload.label not in _ENTITY_LABELS:
        return result

    expected_prefix = f"{_entity_name(payload)}:"
    if not result.context.startswith(expected_prefix):
        raise _context_validation_error(
            f"context must start with {expected_prefix!r}",
            result.context,
        )

    if payload.label == "PermissionMode":
        name = _entity_name(payload)
        lowered = result.context.lower()
        sibling_hits = [
            sibling for sibling in payload.related_hints
            if sibling.lower() != name.lower()
            and _mentions_term(result.context, sibling)
        ]
        if sibling_hits:
            raise _context_validation_error(
                f"PermissionMode context must not mention sibling mode: {sibling_hits[0]}",
                result.context,
            )
        read_only = _has_any(lowered, ("read-only", "read only"))
        if name == "default" and not (
            "baseline" in lowered
            and read_only
            and _has_any(lowered, ("session", "start"))
        ):
            raise _context_validation_error(
                "default PermissionMode context must mention baseline/session-start read-only role",
                result.context,
            )
        if name == "plan" and not (
            read_only
            and "enterplanmode" in lowered
            and _has_any(
                lowered,
                ("approval", "before edit", "before-edit", "implementation", "proposal", "strategy"),
            )
        ):
            raise _context_validation_error(
                "plan PermissionMode context must mention deliberate planning/EnterPlanMode read-only role",
                result.context,
            )
        if name == "bypassPermissions" and "protected path" not in lowered:
            raise _context_validation_error(
                "bypassPermissions context must mention protected paths",
                result.context,
            )

    return result


def _build_user_prompt(payload: NodePayload) -> str:
    template = _PROMPT_TEMPLATES.get(payload.label)
    if template is None:
        raise ValueError(f"No prompt template registered for label: {payload.label}")

    common = {
        "node_type": payload.label,
        "page_title": payload.page_title or "unknown",
        "page_description": payload.page_description or "",
        "breadcrumb": payload.breadcrumb or "unknown",
        "parent_text": (payload.parent_text or "")[:300],
        "raw_text": payload.raw_text[:2000],
    }

    if payload.label in _ENTITY_LABELS:
        return template.format(
            node_type=payload.label,
            name=_entity_name(payload),
            description=payload.raw_text[:600],
            breadcrumb=common["breadcrumb"],
            defined_in=payload.extras.get("defined_in", "unknown"),
            definition_text=(payload.extras.get("definition_text", "") or "")[:1200],
            siblings=", ".join(payload.related_hints) if payload.related_hints else "(none specified)",
        )

    if payload.label == "TableRow":
        return template.format(
            page_title=common["page_title"],
            page_description=common["page_description"],
            breadcrumb=common["breadcrumb"],
            headers=payload.extras.get("headers", ""),
            cells=payload.extras.get("cells", ""),
            parent_text=common["parent_text"],
        )

    if payload.label == "Callout":
        return template.format(
            page_title=common["page_title"],
            page_description=common["page_description"],
            breadcrumb=common["breadcrumb"],
            callout_kind=payload.extras.get("callout_kind", "note"),
            parent_text=common["parent_text"],
            raw_text=common["raw_text"],
        )

    if payload.label == "Section":
        return template.format(
            page_title=common["page_title"],
            page_description=common["page_description"],
            breadcrumb=common["breadcrumb"],
            parent_text=common["parent_text"],
            raw_text=common["raw_text"],
        )

    if payload.label == "CodeBlock":
        return template.format(
            language=payload.extras.get("language", "code"),
            page_title=common["page_title"],
            page_description=common["page_description"],
            breadcrumb=common["breadcrumb"],
            preceding=(payload.extras.get("preceding", "") or "")[:300],
            raw_text=common["raw_text"],
        )

    raise ValueError(f"Unhandled label: {payload.label}")


async def generate_context(payload: NodePayload) -> ContextResult:
    user_prompt = _build_user_prompt(payload)
    result = await agent.run(user_prompt)
    return _validate_context_for_payload(payload, result.output)
