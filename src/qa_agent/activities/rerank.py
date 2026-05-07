"""Temporal activity: rerank."""

from __future__ import annotations

import logfire
from temporalio import activity
from temporalio.exceptions import ApplicationError

from qa_agent.config import get_settings
from qa_agent.retrieval.rerank import cohere_rerank
from qa_agent.schemas import Candidate, RerankRequest


@activity.defn(name="rerank")
async def rerank(req: RerankRequest) -> list[Candidate]:
    with logfire.span("rerank", count=len(req.candidates), top_k=req.top_k):
        if not req.candidates:
            return []

        settings = get_settings()
        if not settings.cohere_api_key or settings.cohere_api_key == "changeme":
            logfire.warning("Cohere API key missing; workflow will use RRF fallback")
            raise ApplicationError(
                "Cohere API key is not configured",
                non_retryable=True,
                type="CohereMissingKey",
            )

        try:
            return await cohere_rerank(
                query=req.question,
                candidates=req.candidates,
                top_k=req.top_k,
                doc_chars=settings.rerank_doc_chars,
                entity_budget=settings.rerank_entity_budget,
            )
        except Exception as exc:
            message = str(exc)
            if any(
                token in message
                for token in ("401", "403", "429", "InvalidApiKey", "Unauthorized")
            ):
                logfire.warning("Cohere 4xx; non-retryable", error=message)
                raise ApplicationError(
                    f"Cohere rerank refused: {message}",
                    non_retryable=True,
                    type="CohereClientError",
                ) from exc
            raise
