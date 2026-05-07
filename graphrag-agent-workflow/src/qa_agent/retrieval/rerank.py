"""Cohere rerank wrapper for retrieval candidates."""

from __future__ import annotations

from functools import lru_cache

import cohere

from qa_agent.config import get_settings
from qa_agent.retrieval.entity import ENTITY_LABELS
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


def _is_entity(c: Candidate) -> bool:
    return c.node_label in ENTITY_LABELS


def _is_lookup_entity(c: Candidate) -> bool:
    """An entity is a lookup hit if it came from entity_lookup (which sets
    bm25_score from the lookup_*.cypher relevance) rather than from graph
    expansion (which only sets expansion_origin).
    """
    return _is_entity(c) and c.bm25_score is not None and c.expansion_origin is None


def _apply_entity_budget(
    scored: list[Candidate],
    top_k: int,
    entity_budget: int,
) -> list[Candidate]:
    """Two-lane top_k allocation. `scored` arrives in Cohere score order.

    The entity lane reserves up to `entity_budget` slots and is filled with
    lookup entities first (closed-vocabulary matches from entity_lookup),
    then any other entity by Cohere score. The document lane fills the
    remaining (top_k - filled_entity_slots) slots with the highest-scored
    non-entity candidates. Final order is by Cohere score across both lanes.
    """
    if entity_budget <= 0 or top_k <= 0:
        return scored[:top_k]

    entities = [c for c in scored if _is_entity(c)]
    documents = [c for c in scored if not _is_entity(c)]

    # Rank entities by tier (lookup first), preserving Cohere order within tier.
    lookup_entities = [c for c in entities if _is_lookup_entity(c)]
    other_entities = [c for c in entities if not _is_lookup_entity(c)]
    ranked_entities = lookup_entities + other_entities

    n_entity_slots = min(entity_budget, len(ranked_entities), top_k)
    chosen_entities = ranked_entities[:n_entity_slots]
    chosen_documents = documents[: max(0, top_k - n_entity_slots)]

    combined = chosen_entities + chosen_documents
    # Final order: highest Cohere score first.
    combined.sort(key=lambda c: -(c.rerank_score if c.rerank_score is not None else 0.0))
    return combined


async def cohere_rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int,
    doc_chars: int,
    entity_budget: int = 0,
) -> list[Candidate]:
    if not candidates:
        return []

    settings = get_settings()
    client = _get_client()
    documents = [
        _truncate(candidate.indexed_text or candidate.raw_text or "", doc_chars)
        for candidate in candidates
    ]
    request_top_n = (
        len(candidates) if entity_budget > 0 else min(top_k, len(candidates))
    )
    response = await client.rerank(
        model=settings.cohere_rerank_model,
        query=query,
        documents=documents,
        top_n=request_top_n,
    )

    scored: list[Candidate] = []
    for result in response.results:
        candidate = candidates[result.index].model_copy(
            update={"rerank_score": float(result.relevance_score)}
        )
        scored.append(candidate)

    return _apply_entity_budget(scored, top_k=top_k, entity_budget=entity_budget)
