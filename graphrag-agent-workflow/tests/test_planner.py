import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIChatModel

from qa_agent.agents.planner import build_planner_agent, plan_question
from qa_agent.schemas import Plan


@pytest.mark.asyncio
async def test_planner_returns_valid_plan_with_test_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
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


def test_planner_default_model_uses_openai_small_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")

    from qa_agent.config import get_settings

    get_settings.cache_clear()
    agent = build_planner_agent()

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "gpt-5.4-mini"
