"""Planner agent: turns a user question into a retrieval Plan."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from qa_agent.config import get_settings
from qa_agent.schemas import Plan

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_planner_agent(model: object | None = None) -> Agent[None, Plan]:
    if model is None:
        settings = get_settings()
        model = OpenAIChatModel(
            settings.openai_model,
            provider=OpenAIProvider(api_key=settings.openai_api_key),
        )

    return Agent(
        model=model,
        output_type=Plan,
        system_prompt=_load_prompt(),
    )


async def plan_question(question: str, agent: Agent[None, Plan] | None = None) -> Plan:
    planner = agent if agent is not None else build_planner_agent()
    result = await planner.run(question)
    return result.output
