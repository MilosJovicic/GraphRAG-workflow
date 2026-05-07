"""Chainlit chat UI for the GraphRAG Q&A agent.

Logic lives in pure helpers so it can be unit-tested without a Chainlit
runtime context. ``render_qa_response`` returns plain dicts for citation
elements; the on_message handler converts them to ``cl.Text`` instances at
the moment of sending, when the runtime context exists.
"""

from __future__ import annotations

import logging
import os

import httpx

QA_API_URL = os.environ.get("QA_API_URL", "http://qa-api:5000")
QA_API_TIMEOUT = float(os.environ.get("QA_API_TIMEOUT", "120"))

log = logging.getLogger(__name__)


async def call_qa_api(client: httpx.AsyncClient, url: str, question: str) -> dict:
    """POST one question to the Flask QA API and return the parsed JSON body.

    Raises ``httpx.HTTPStatusError`` on non-2xx and other ``httpx`` exceptions
    on transport failures.
    """
    resp = await client.post(
        f"{url}/ask",
        json={"question": question},
        timeout=QA_API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def render_qa_response(payload: dict) -> tuple[str, list[dict]]:
    """Convert a QA API response into a message body and citation element specs.

    Each element spec is a ``dict`` with ``name``, ``content``, and ``display``
    keys, ready to be splatted into ``cl.Text(**spec)`` once a Chainlit
    runtime context exists.
    """
    answer = payload["answer"]
    if payload.get("no_results"):
        return answer, []

    elements: list[dict] = []
    for citation in payload.get("citations", []):
        label = (
            citation.get("title")
            or citation.get("url")
            or citation.get("node_id", "?")
        )
        elements.append(
            {
                "name": f"[{citation['id']}] {label}",
                "content": citation["snippet"],
                "display": "side",
            }
        )
    return answer, elements
