"""Cohere rerank wrapper for retrieval candidates."""

from __future__ import annotations

from functools import lru_cache

import cohere

from qa_agent.config import get_settings
from qa_agent.schemas import Candidate


@lru_cache
def _get_client() -> cohere.AsyncClientV2:
    settings = get_settings()
    if not settings.cohere_api_key or settings.cohere_api_key == "changeme":
        raise RuntimeError("COHERE_API_KEY is not configured")
    return cohere.AsyncClientV2(api_key=settings.cohere_api_key)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


async def cohere_rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int,
    doc_chars: int,
) -> list[Candidate]:
    if not candidates:
        return []

    settings = get_settings()
    client = _get_client()
    documents = [
        _truncate(candidate.indexed_text or candidate.raw_text or "", doc_chars)
        for candidate in candidates
    ]
    response = await client.rerank(
        model=settings.cohere_rerank_model,
        query=query,
        documents=documents,
        top_n=min(top_k, len(candidates)),
    )

    out: list[Candidate] = []
    for result in response.results:
        candidate = candidates[result.index].model_copy(
            update={"rerank_score": float(result.relevance_score)}
        )
        out.append(candidate)
    return out
