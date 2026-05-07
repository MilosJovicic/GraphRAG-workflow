"""CLI starter: run a single question through QAWorkflow."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow


async def ask(question: str, debug: bool = False) -> QAResponse:
    settings = get_settings()
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )
    return await client.execute_workflow(
        QAWorkflow.run,
        QARequest(question=question, debug=debug),
        id=f"qa-{uuid.uuid4().hex[:12]}",
        task_queue=settings.qa_task_queue,
        result_type=QAResponse,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m qa_agent.starter "<question>"', file=sys.stderr)
        return 2

    question = " ".join(sys.argv[1:])
    response = asyncio.run(ask(question))
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
