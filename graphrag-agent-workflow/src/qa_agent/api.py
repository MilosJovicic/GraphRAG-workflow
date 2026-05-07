"""Flask API: POST /ask, GET /health."""

from __future__ import annotations

import asyncio
import time
import uuid

from flask import Flask, jsonify, request
from pydantic import ValidationError
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow


async def run_workflow(req: QARequest) -> QAResponse:
    settings = get_settings()
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )
    return await client.execute_workflow(
        QAWorkflow.run,
        req,
        id=f"qa-{uuid.uuid4().hex[:12]}",
        task_queue=settings.qa_task_queue,
        result_type=QAResponse,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        try:
            req = QARequest(**payload)
        except ValidationError as exc:
            return jsonify({"error": "invalid request", "details": exc.errors()}), 400

        started = time.monotonic()
        try:
            response = asyncio.run(run_workflow(req))
        except Exception as exc:
            return jsonify({"error": "workflow_failure", "detail": str(exc)}), 500

        total_ms = int((time.monotonic() - started) * 1000)
        latency = dict(response.latency_ms or {})
        latency["total"] = total_ms
        return jsonify(response.model_copy(update={"latency_ms": latency}).model_dump())

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
