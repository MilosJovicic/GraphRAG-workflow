"""QAWorkflow orchestrates GraphRAG retrieval and answer generation."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from qa_agent.retrieval.fusion import rrf_fuse
    from qa_agent.schemas import (
        Candidate,
        ExpandRequest,
        GenerateRequest,
        Plan,
        QARequest,
        QAResponse,
        RerankRequest,
    )

_DEFAULT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


def _is_non_retryable_application_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, ActivityError)
        and isinstance(exc.cause, ApplicationError)
        and exc.cause.non_retryable
    )


@workflow.defn
class QAWorkflow:
    @workflow.run
    async def run(self, req: QARequest) -> QAResponse:
        plan: Plan
        planner_failed = False
        try:
            plan = await workflow.execute_activity(
                "plan_query",
                req.question,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
        except ActivityError as exc:
            if not _is_non_retryable_application_error(exc):
                raise
            planner_failed = True
            plan = Plan.fallback_for(req.question)

        if planner_failed:
            top: list[Candidate] = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
            if not top:
                return QAResponse.empty(plan, fallback_used=True)

            response = await workflow.execute_activity(
                "generate_answer",
                GenerateRequest(question=req.question, evidence=top, plan=plan),
                start_to_close_timeout=timedelta(seconds=45),
                retry_policy=_DEFAULT_RETRY,
            )
            return response.model_copy(update={"fallback_used": True})

        per_subquery: list[list[Candidate]] = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "hybrid_search",
                    sub_query,
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=_DEFAULT_RETRY,
                )
                for sub_query in plan.sub_queries
            ]
        )

        fused = rrf_fuse(per_subquery, k=60, top_n=40)
        expanded = await workflow.execute_activity(
            "expand_graph",
            ExpandRequest(seeds=fused, patterns=plan.expansion_patterns),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=_DEFAULT_RETRY,
        )

        fallback_used = False
        try:
            top = await workflow.execute_activity(
                "rerank",
                RerankRequest(question=req.question, candidates=expanded, top_k=8),
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=_DEFAULT_RETRY,
            )
        except ActivityError as exc:
            if not _is_non_retryable_application_error(exc):
                raise
            fallback_used = True
            top = expanded[:8]

        if not top:
            fallback_used = True
            top = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )

        if not top:
            return QAResponse.empty(plan, fallback_used=fallback_used)

        response = await workflow.execute_activity(
            "generate_answer",
            GenerateRequest(question=req.question, evidence=top, plan=plan),
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=_DEFAULT_RETRY,
        )
        return response.model_copy(update={"fallback_used": fallback_used})
