"""Integration smoke test against live Temporal worker and retrieval services."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import yaml
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("QA_RUN_INTEGRATION") != "1",
        reason="set QA_RUN_INTEGRATION=1 with Temporal worker and services running",
    ),
]


def _load_eval():
    path = Path(__file__).resolve().parents[1] / "ragas_evals" / "gold_questions.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", _load_eval(), ids=lambda entry: entry["id"])
async def test_recall_at_8(entry):
    settings = get_settings()
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )
    response: QAResponse = await client.execute_workflow(
        QAWorkflow.run,
        QARequest(question=entry["question"], debug=True),
        id=f"qa-eval-{entry['id']}-{uuid.uuid4().hex[:6]}",
        task_queue=settings.qa_task_queue,
        result_type=QAResponse,
    )

    cited_ids = {citation.node_id for citation in response.citations}
    retrieved_ids = {candidate.node_id for candidate in (response.retrieved or [])}
    seen = cited_ids | retrieved_ids
    expected = set(entry["expected_context_ids"])
    overlap = seen & expected
    assert overlap, (
        f"Q[{entry['id']}] {entry['question']!r}: none of expected={expected} "
        f"found in cited={cited_ids} or retrieved={retrieved_ids}"
    )
