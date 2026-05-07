"""Temporal activity: expand_graph."""

from __future__ import annotations

import logfire
from temporalio import activity

from qa_agent.config import get_settings
from qa_agent.retrieval.expansion import expand
from qa_agent.schemas import Candidate, ExpandRequest


@activity.defn(name="expand_graph")
async def expand_graph(req: ExpandRequest) -> list[Candidate]:
    with logfire.span(
        "expand_graph",
        seed_count=len(req.seeds),
        patterns=[pattern.name for pattern in req.patterns],
    ):
        settings = get_settings()
        result = await expand(
            seeds=req.seeds,
            patterns=req.patterns,
            total_cap=settings.max_expansion_total,
        )
        logfire.info("expand_graph.done", count=len(result))
        return result
