from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.api import create_app
from qa_agent.schemas import Plan, QAResponse, SubQuery


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("COHERE_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    from qa_agent.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_ask_validates_payload(client):
    response = client.post("/ask", json={})

    assert response.status_code == 400


def test_ask_invokes_workflow(client):
    fake_response = QAResponse(
        answer="hi [1].",
        citations=[],
        plan=Plan(sub_queries=[SubQuery(text="hi", target_labels=["Section"])]),
    )

    with patch("qa_agent.api.run_workflow", new=AsyncMock(return_value=fake_response)):
        response = client.post("/ask", json={"question": "hello?"})

    assert response.status_code == 200
    assert response.get_json()["answer"] == "hi [1]."
