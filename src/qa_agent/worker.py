"""Temporal worker entrypoint for the Q&A agent."""

from __future__ import annotations

import asyncio

import logfire
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from qa_agent.activities.expand import expand_graph
from qa_agent.activities.generate import generate_answer
from qa_agent.activities.plan import plan_query
from qa_agent.activities.rerank import rerank
from qa_agent.activities.retrieve import hybrid_search, naive_hybrid_fallback
from qa_agent.config import get_settings
from qa_agent.workflows.qa import QAWorkflow


async def main() -> None:
    settings = get_settings()
    logfire.configure(service_name="qa-agent-worker", send_to_logfire=False)

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue=settings.qa_task_queue,
        workflows=[QAWorkflow],
        activities=[
            plan_query,
            hybrid_search,
            naive_hybrid_fallback,
            expand_graph,
            rerank,
            generate_answer,
        ],
    )
    print(
        f"qa-agent worker listening on task queue {settings.qa_task_queue!r} "
        f"({settings.temporal_host}, namespace={settings.temporal_namespace})"
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
