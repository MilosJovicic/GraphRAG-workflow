"""Tests for the Chainlit app pure helpers.

The helpers return plain dicts for citation elements rather than cl.Text
instances, because cl.Text construction requires a live Chainlit context that
is only available inside a running Chainlit server. The on_message handler is
responsible for converting the dicts to cl.Text objects.
"""

from __future__ import annotations

from app import render_qa_response


def test_render_qa_response_with_citations_uses_title():
    payload = {
        "answer": "Set the permissionMode in settings.json [1].",
        "citations": [
            {
                "id": 1,
                "node_id": "n1",
                "node_label": "Section",
                "url": "https://example.com/x#a",
                "title": "Permissions",
                "snippet": "Use permissionMode...",
                "score": 0.9,
            }
        ],
        "no_results": False,
    }

    content, elements = render_qa_response(payload)

    assert content == payload["answer"]
    assert elements == [
        {
            "name": "[1] Permissions",
            "content": "Use permissionMode...",
            "display": "side",
        }
    ]


def test_render_qa_response_falls_back_to_url_when_title_missing():
    payload = {
        "answer": "answer [1]",
        "citations": [
            {
                "id": 1,
                "node_id": "n1",
                "node_label": "Chunk",
                "url": "https://example.com/y",
                "snippet": "...",
                "score": 0.5,
            }
        ],
        "no_results": False,
    }

    _, elements = render_qa_response(payload)

    assert elements[0]["name"] == "[1] https://example.com/y"


def test_render_qa_response_falls_back_to_node_id_when_url_missing():
    payload = {
        "answer": "answer [1]",
        "citations": [
            {
                "id": 1,
                "node_id": "section-abc",
                "node_label": "Section",
                "snippet": "...",
                "score": 0.5,
            }
        ],
        "no_results": False,
    }

    _, elements = render_qa_response(payload)

    assert elements[0]["name"] == "[1] section-abc"


def test_render_qa_response_no_results_returns_empty_elements():
    payload = {
        "answer": "I couldn't find anything relevant in the knowledge base.",
        "citations": [],
        "no_results": True,
    }

    content, elements = render_qa_response(payload)

    assert content == payload["answer"]
    assert elements == []
