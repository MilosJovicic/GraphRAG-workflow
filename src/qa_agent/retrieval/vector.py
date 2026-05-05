"""Vector retrieval over searchable graph labels."""

from __future__ import annotations

from qa_agent.embeddings import embed_one
from qa_agent.neo4j_client import run_cypher
from qa_agent.retrieval.bm25 import (
    SEARCHABLE_LABELS,
    _build_filter_params,
)
from qa_agent.schemas import Candidate, SubQuery

__all__ = ["vector_search"]


def _template_for(label: str) -> str:
    return f"vector_{label.lower()}.cypher"


async def vector_search(sq: SubQuery, label: str, limit: int) -> list[Candidate]:
    if label not in SEARCHABLE_LABELS:
        raise ValueError(f"Vector search not supported for label {label!r}")

    query_vector = await embed_one(sq.text)
    params = {
        "query_vector": query_vector,
        "limit": limit,
        **_build_filter_params(sq.filters),
    }
    rows = await run_cypher(_template_for(label), params)
    return [Candidate(**row) for row in rows]
