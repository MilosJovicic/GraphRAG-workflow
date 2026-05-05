"""QAWorkflow orchestrates GraphRAG retrieval and answer generation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

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


def _ms_since(start: datetime) -> int:
    return int((workflow.now() - start).total_seconds() * 1000)


@workflow.defn
class QAWorkflow:
    @workflow.run
    async def run(self, req: QARequest) -> QAResponse:
        latency_ms: dict[str, int] = {}
        wf_start = workflow.now()

        plan: Plan
        planner_failed = False
        try:
            t = workflow.now()
            plan = await workflow.execute_activity(
                "plan_query",
                req.question,
                result_type=Plan,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
            latency_ms["plan"] = _ms_since(t)
        except ActivityError as exc:
            if not _is_non_retryable_application_error(exc):
                raise
            planner_failed = True
            plan = Plan.fallback_for(req.question)

        if planner_failed:
            t = workflow.now()
            top: list[Candidate] = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
            latency_ms["naive_fallback"] = _ms_since(t)
            if not top:
                latency_ms["total"] = _ms_since(wf_start)
                return QAResponse.empty(plan, fallback_used=True).model_copy(
                    update={
                        "latency_ms": latency_ms,
                        "retrieved": top if req.debug else None,
                    }
                )

            t = workflow.now()
            response = await workflow.execute_activity(
                "generate_answer",
                GenerateRequest(question=req.question, evidence=top, plan=plan),
                result_type=QAResponse,
                start_to_close_timeout=timedelta(seconds=45),
                retry_policy=_DEFAULT_RETRY,
            )
            latency_ms["generate"] = _ms_since(t)
            latency_ms["total"] = _ms_since(wf_start)
            return response.model_copy(
                update={
                    "fallback_used": True,
                    "latency_ms": latency_ms,
                    "retrieved": top if req.debug else None,
                }
            )

        t = workflow.now()
        per_subquery: list[list[Candidate]] = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "hybrid_search",
                    sub_query,
                    result_type=list[Candidate],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=_DEFAULT_RETRY,
                )
                for sub_query in plan.sub_queries
            ]
        )
        latency_ms["hybrid_search"] = _ms_since(t)

        fused = rrf_fuse(per_subquery, k=60, top_n=40)

        t = workflow.now()
        expanded = await workflow.execute_activity(
            "expand_graph",
            ExpandRequest(seeds=fused, patterns=plan.expansion_patterns),
            result_type=list[Candidate],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=_DEFAULT_RETRY,
        )
        latency_ms["expand"] = _ms_since(t)

        fallback_used = False
        t = workflow.now()
        try:
            top = await workflow.execute_activity(
                "rerank",
                RerankRequest(question=req.question, candidates=expanded, top_k=8),
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=_DEFAULT_RETRY,
            )
        except ActivityError as exc:
            if not _is_non_retryable_application_error(exc):
                raise
            fallback_used = True
            top = expanded[:8]
        latency_ms["rerank"] = _ms_since(t)

        if not top:
            fallback_used = True
            t = workflow.now()
            top = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
            latency_ms["naive_fallback"] = _ms_since(t)

        if not top:
            latency_ms["total"] = _ms_since(wf_start)
            return QAResponse.empty(plan, fallback_used=fallback_used).model_copy(
                update={
                    "latency_ms": latency_ms,
                    "retrieved": top if req.debug else None,
                }
            )

        t = workflow.now()
        response = await workflow.execute_activity(
            "generate_answer",
            GenerateRequest(question=req.question, evidence=top, plan=plan),
            result_type=QAResponse,
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=_DEFAULT_RETRY,
        )
        latency_ms["generate"] = _ms_since(t)
        latency_ms["total"] = _ms_since(wf_start)
        return response.model_copy(
            update={
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
                "retrieved": top if req.debug else None,
            }
        )
