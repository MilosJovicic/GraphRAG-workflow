"""Temporal activities: hybrid_search and naive_hybrid_fallback."""

from __future__ import annotations

import asyncio

import logfire
from temporalio import activity

from qa_agent.config import get_settings
from qa_agent.retrieval.bm25 import SEARCHABLE_LABELS, bm25_search
from qa_agent.retrieval.entity import ENTITY_LABELS, entity_lookup
from qa_agent.retrieval.fusion import rrf_fuse
from qa_agent.retrieval.rerank import cohere_rerank
from qa_agent.retrieval.vector import vector_search
from qa_agent.schemas import Candidate, SubQuery


def _merge_legs(bm25: list[Candidate], vector: list[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[str, str], Candidate] = {}
    for candidates in (bm25, vector):
        for candidate in candidates:
            key = (candidate.node_id, candidate.node_label)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                continue

            by_key[key] = existing.model_copy(
                update={
                    "bm25_score": (
                        existing.bm25_score
                        if existing.bm25_score is not None
                        else candidate.bm25_score
                    ),
                    "vector_score": (
                        existing.vector_score
                        if existing.vector_score is not None
                        else candidate.vector_score
                    ),
                }
            )
    return list(by_key.values())


async def _hybrid_for_label(sub_query: SubQuery, label: str) -> list[Candidate]:
    settings = get_settings()
    if label in ENTITY_LABELS:
        return await entity_lookup(sub_query, label=label, limit=settings.bm25_top_k)
    bm25_task = bm25_search(sub_query, label=label, limit=settings.bm25_top_k)
    vector_task = vector_search(sub_query, label=label, limit=settings.vector_top_k)
    bm25, vector = await asyncio.gather(bm25_task, vector_task)
    return _merge_legs(bm25, vector)


@activity.defn(name="hybrid_search")
async def hybrid_search(sub_query: SubQuery) -> list[Candidate]:
    with logfire.span("hybrid_search", sub_query=sub_query.text[:200]):
        labels = [
            label
            for label in sub_query.target_labels
            if label in SEARCHABLE_LABELS or label in ENTITY_LABELS
        ]
        if not labels:
            return []

        per_label = await asyncio.gather(
            *[_hybrid_for_label(sub_query, label) for label in labels]
        )
        merged: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidates in per_label:
            for candidate in candidates:
                key = (candidate.node_id, candidate.node_label)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)

        logfire.info("hybrid_search.done", labels=labels, count=len(merged))
        return merged


@activity.defn(name="naive_hybrid_fallback")
async def naive_hybrid_fallback(question: str) -> list[Candidate]:
    with logfire.span("naive_hybrid_fallback", question=question[:200]):
        settings = get_settings()
        sub_query = SubQuery(
            text=question,
            target_labels=["Section", "CodeBlock", "TableRow", "Callout"],
        )
        candidates = await hybrid_search(sub_query)
        if not candidates:
            return []

        fused = rrf_fuse([candidates], k=60, top_n=settings.rrf_top_n)
        try:
            return await cohere_rerank(
                question,
                fused,
                top_k=settings.rerank_top_k,
                doc_chars=settings.rerank_doc_chars,
                entity_budget=settings.rerank_entity_budget,
            )
        except Exception as exc:
            logfire.warning(
                "fallback rerank failed; returning RRF top-K",
                error=str(exc),
            )
            return fused[: settings.rerank_top_k]
