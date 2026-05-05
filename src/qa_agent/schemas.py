from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NODE_LABELS = [
    "Section",
    "Chunk",
    "CodeBlock",
    "TableRow",
    "Callout",
    "Tool",
    "Hook",
    "SettingKey",
    "PermissionMode",
    "MessageType",
    "Provider",
]
NodeLabel = Literal[
    "Section",
    "Chunk",
    "CodeBlock",
    "TableRow",
    "Callout",
    "Tool",
    "Hook",
    "SettingKey",
    "PermissionMode",
    "MessageType",
    "Provider",
]
ExpansionName = Literal["siblings", "parent_page", "links", "defines", "navigates_to"]


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    debug: bool = False


class Citation(BaseModel):
    id: int
    node_id: str
    node_label: NodeLabel
    url: str | None = None
    anchor: str | None = None
    title: str | None = None
    snippet: str
    score: float


class SubQuery(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    target_labels: list[NodeLabel] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    bm25_keywords: list[str] = Field(default_factory=list)


class ExpansionPattern(BaseModel):
    name: ExpansionName
    max_per_seed: int = Field(default=3, ge=1, le=5)


class Plan(BaseModel):
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=4)
    expansion_patterns: list[ExpansionPattern] = Field(default_factory=list)
    notes: str = ""

    @classmethod
    def fallback_for(cls, question: str) -> Plan:
        return cls(
            sub_queries=[
                SubQuery(
                    text=question,
                    target_labels=["Section", "CodeBlock", "TableRow", "Callout"],
                )
            ],
            expansion_patterns=[],
            notes="synthetic-fallback",
        )


class Candidate(BaseModel):
    node_id: str
    node_label: str
    indexed_text: str
    raw_text: str
    url: str | None = None
    anchor: str | None = None
    breadcrumb: str | None = None
    title: str | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    expansion_origin: str | None = None


class ExpandRequest(BaseModel):
    seeds: list[Candidate]
    patterns: list[ExpansionPattern]


class RerankRequest(BaseModel):
    question: str
    candidates: list[Candidate]
    top_k: int = 8


class GenerateRequest(BaseModel):
    question: str
    evidence: list[Candidate]
    plan: Plan


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    plan: Plan
    retrieved: list[Candidate] | None = None
    latency_ms: dict[str, int] = Field(default_factory=dict)
    fallback_used: bool = False
    no_results: bool = False

    @classmethod
    def empty(cls, plan: Plan, fallback_used: bool = False) -> QAResponse:
        return cls(
            answer=(
                "I couldn't find anything relevant in the knowledge base for that question. "
                "Try rephrasing, or ask about a more specific topic."
            ),
            citations=[],
            plan=plan,
            fallback_used=fallback_used,
            no_results=True,
        )


class AnswerWithCitations(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    used_citation_ids: list[int] = Field(default_factory=list)
