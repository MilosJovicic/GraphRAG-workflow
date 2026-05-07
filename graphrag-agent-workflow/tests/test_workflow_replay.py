"""Determinism replay test for QAWorkflow."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from qa_agent.schemas import (
    Candidate,
    ExpandRequest,
    ExpansionPattern,
    GenerateRequest,
    Plan,
    QARequest,
    QAResponse,
    RerankRequest,
    SubQuery,
)
from qa_agent.workflows.qa import QAWorkflow


def _candidate(node_id: str, score: float = 1.0) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label="Section",
        indexed_text="x",
        raw_text="y",
        bm25_score=score,
        vector_score=score,
        rerank_score=score,
    )


@activity.defn(name="plan_query")
async def stub_plan(question: str) -> Plan:
    return Plan(
        sub_queries=[SubQuery(text=question, target_labels=["Section"])],
        expansion_patterns=[ExpansionPattern(name="parent_page")],
    )


@activity.defn(name="hybrid_search")
async def stub_hybrid(sq: SubQuery) -> list[Candidate]:
    return [_candidate("a", 10.0), _candidate("b", 8.0)]


@activity.defn(name="expand_graph")
async def stub_expand(req: ExpandRequest) -> list[Candidate]:
    return req.seeds + [_candidate("c", 1.0)]


@activity.defn(name="rerank")
async def stub_rerank(req: RerankRequest) -> list[Candidate]:
    return req.candidates[: req.top_k]


@activity.defn(name="naive_hybrid_fallback")
async def stub_fallback(question: str) -> list[Candidate]:
    return [_candidate("z", 0.5)]


@activity.defn(name="generate_answer")
async def stub_generate(req: GenerateRequest) -> QAResponse:
    return QAResponse(
        answer="ok [1].",
        citations=[],
        plan=req.plan,
        latency_ms={},
    )


@pytest.mark.asyncio
async def test_qaworkflow_replays_cleanly():
    workflow_id = f"qa-replay-test-{uuid.uuid4()}"
    task_queue = f"qa-replay-tq-{uuid.uuid4()}"

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        client: Client = env.client
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[QAWorkflow],
            activities=[
                stub_plan,
                stub_hybrid,
                stub_expand,
                stub_rerank,
                stub_fallback,
                stub_generate,
            ],
        ):
            handle = await client.start_workflow(
                QAWorkflow.run,
                QARequest(question="hello"),
                id=workflow_id,
                task_queue=task_queue,
                execution_timeout=timedelta(seconds=30),
            )
            result: QAResponse = await handle.result()
            assert result.answer.startswith("ok")
            history = await handle.fetch_history()

    replayer = Replayer(
        workflows=[QAWorkflow],
        data_converter=pydantic_data_converter,
    )
    await replayer.replay_workflow(history)
