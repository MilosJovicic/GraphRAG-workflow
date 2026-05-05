"""Temporal activity: plan_query."""

from __future__ import annotations

import logfire
from pydantic import ValidationError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from qa_agent.agents.planner import plan_question
from qa_agent.schemas import Plan


@activity.defn(name="plan_query")
async def plan_query(question: str) -> Plan:
    with logfire.span("plan_query", question=question[:200]):
        try:
            return await plan_question(question)
        except ValidationError as exc:
            raise ApplicationError(
                f"planner produced invalid Plan: {exc}",
                non_retryable=True,
                type="PlannerValidationError",
            ) from exc
