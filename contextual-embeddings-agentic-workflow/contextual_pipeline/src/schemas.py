from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class NodePayload(BaseModel):
    node_id: str
    label: Literal[
        "Section", "CodeBlock", "TableRow", "Callout",
        "Tool", "Hook", "SettingKey", "PermissionMode",
        "MessageType", "Provider"
    ]
    raw_text: str
    page_title: Optional[str] = None
    page_description: Optional[str] = None
    breadcrumb: Optional[str] = None
    parent_text: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    related_hints: list[str] = Field(default_factory=list)
    extras: dict[str, str] = Field(default_factory=dict)


_FORBIDDEN_OPENERS = {"this", "the", "a", "an"}

_FORBIDDEN_PHRASES = (
    "related hints",
    "this node defines",
    "this section discusses",
    "this callout highlights",
    "this tablerow",
    "this codeblock",
    "within the documentation",
    "in the documentation",
    "distinct from related",
)


class ContextResult(BaseModel):
    context: str = Field(
        min_length=10,
        max_length=600,
        description="1-3 sentence retrieval context",
    )

    @field_validator("context")
    @classmethod
    def reject_boilerplate(cls, v: str) -> str:
        tokens = v.split()
        if not tokens:
            raise ValueError("context must not be empty")
        first = tokens[0].lower().rstrip(":,.;!?")
        if first in _FORBIDDEN_OPENERS:
            raise ValueError(
                f"context must not start with a meta opener "
                f"({tokens[0]!r}); first word should be the entity/subject"
            )
        v_lower = v.lower()
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in v_lower:
                raise ValueError(f"context contains forbidden phrase: {phrase!r}")
        return v
