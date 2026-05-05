import pytest
from pydantic_ai.models.test import TestModel

from qa_agent.agents.planner import build_planner_agent, plan_question
from qa_agent.schemas import Plan


@pytest.mark.asyncio
async def test_planner_returns_valid_plan_with_test_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("COHERE_API_KEY", "x")

    from qa_agent.config import get_settings

    get_settings.cache_clear()
    agent = build_planner_agent(model=TestModel())
    plan = await plan_question("How do I configure Bash?", agent=agent)

    assert isinstance(plan, Plan)
    assert len(plan.sub_queries) >= 1
