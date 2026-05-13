"""BM25 fulltext retrieval over searchable graph labels."""

from __future__ import annotations

from qa_agent.neo4j_client import run_cypher
from qa_agent.schemas import Candidate, SubQuery

SEARCHABLE_LABELS = {"Section", "Chunk", "CodeBlock", "TableRow", "Callout"}
ALLOWED_FILTERS = {"language", "page_path_prefix"}
LUCENE_SPECIAL_CHARS = frozenset('+-&|!(){}[]^"~*?:\\/')


def _template_for(label: str) -> str:
    return f"fulltext_{label.lower()}.cypher"


def _build_query_string(sq: SubQuery) -> str:
    parts = [sq.text.strip()]
    if sq.bm25_keywords:
        parts.append(" ".join(sq.bm25_keywords))
    query = " ".join(part for part in parts if part)
    return _escape_lucene_query(query)


def _escape_lucene_query(query: str) -> str:
    return "".join(
        f"\\{char}" if char in LUCENE_SPECIAL_CHARS else char for char in query
    )


def _build_filter_params(filters: dict[str, str]) -> dict[str, str | None]:
    params: dict[str, str | None] = {key: None for key in ALLOWED_FILTERS}
    for key, value in filters.items():
        if key in ALLOWED_FILTERS:
            params[key] = value
    return params


async def bm25_search(sq: SubQuery, label: str, limit: int) -> list[Candidate]:
    if label not in SEARCHABLE_LABELS:
        raise ValueError(f"BM25 search not supported for label {label!r}")

    params = {
        "query": _build_query_string(sq),
        "limit": limit,
        **_build_filter_params(sq.filters),
    }
    rows = await run_cypher(_template_for(label), params)
    return [Candidate(**row) for row in rows]
