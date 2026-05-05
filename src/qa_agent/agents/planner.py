"""Planner agent: turns a user question into a retrieval Plan."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from qa_agent.config import get_settings
from qa_agent.schemas import Plan

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_planner_agent(model: object | None = None) -> Agent[None, Plan]:
    if model is None:
        settings = get_settings()
        model = GoogleModel(
            settings.gemini_model,
            provider=GoogleProvider(api_key=settings.gemini_api_key),
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
