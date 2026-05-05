"""Temporal activity: generate_answer."""

from __future__ import annotations

import logfire
from temporalio import activity

from qa_agent.agents.answerer import (
    answer_with_citations,
    extract_citation_ids,
    validate_citations,
)
from qa_agent.schemas import Candidate, Citation, GenerateRequest, QAResponse

_SNIPPET_CHARS = 280


def _candidate_to_citation(idx: int, candidate: Candidate) -> Citation:
    return Citation(
        id=idx,
        node_id=candidate.node_id,
        node_label=candidate.node_label,
        url=candidate.url,
        anchor=candidate.anchor,
        title=candidate.title or candidate.breadcrumb or candidate.node_id,
        snippet=(candidate.indexed_text or candidate.raw_text or "")[:_SNIPPET_CHARS],
        score=float(
            candidate.rerank_score
            if candidate.rerank_score is not None
            else candidate.rrf_score
            if candidate.rrf_score is not None
            else 0.0
        ),
    )


@activity.defn(name="generate_answer")
async def generate_answer(req: GenerateRequest) -> QAResponse:
    with logfire.span(
        "generate_answer",
        evidence_count=len(req.evidence),
        question=req.question[:200],
    ):
        result = await answer_with_citations(req.question, req.evidence)
        ok, missing, out_of_range = validate_citations(
            result.answer,
            evidence_count=len(req.evidence),
            used_ids=result.used_citation_ids,
        )
        if not ok:
            logfire.warning(
                "answer citation mismatch; accepting and trimming",
                missing=missing,
                out_of_range=out_of_range,
            )

        used_in_text = extract_citation_ids(result.answer)
        valid_ids = [idx for idx in used_in_text if 1 <= idx <= len(req.evidence)]
        citations = [
            _candidate_to_citation(idx, req.evidence[idx - 1]) for idx in valid_ids
        ]

        return QAResponse(
            answer=result.answer,
            citations=citations,
            plan=req.plan,
            latency_ms={},
            fallback_used=False,
            no_results=not citations,
        )
