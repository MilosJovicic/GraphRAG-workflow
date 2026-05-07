"""Direct entity lookup over closed-vocabulary graph labels.

Entity nodes (Tool, Hook, SettingKey, PermissionMode, Provider, MessageType)
have no fulltext or vector index, so this module matches them by their
canonical name/key against tokens drawn from the sub-query text and
bm25_keywords.
"""

from __future__ import annotations

from qa_agent.neo4j_client import run_cypher
from qa_agent.schemas import Candidate, SubQuery

ENTITY_LABELS = {
    "Tool",
    "Hook",
    "SettingKey",
    "PermissionMode",
    "Provider",
    "MessageType",
}

_MIN_TERM_LEN = 2


def _template_for(label: str) -> str:
    return f"lookup_{label.lower()}.cypher"


def _build_terms(sq: SubQuery) -> list[str]:
    raw: list[str] = []
    raw.extend(sq.text.split())
    for kw in sq.bm25_keywords:
        raw.extend(kw.split())

    seen: set[str] = set()
    out: list[str] = []
    for token in raw:
        cleaned = token.strip(",.;:?!()[]{}\"'")
        if len(cleaned) < _MIN_TERM_LEN:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


async def entity_lookup(sq: SubQuery, label: str, limit: int) -> list[Candidate]:
    if label not in ENTITY_LABELS:
        raise ValueError(f"Entity lookup not supported for label {label!r}")

    terms = _build_terms(sq)
    if not terms:
        return []

    rows = await run_cypher(
        _template_for(label),
        {"terms": terms, "limit": limit},
    )
    return [Candidate(**row) for row in rows]
