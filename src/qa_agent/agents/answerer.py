"""Answerer agent: cited answer from numbered evidence."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic_ai import Agent

from qa_agent.config import get_settings
from qa_agent.schemas import AnswerWithCitations, Candidate

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "answerer.txt"
_CITATION_RE = re.compile(r"\[(\d+)\]")
_EVIDENCE_TEXT_CHARS = 800


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_evidence(candidates: list[Candidate]) -> str:
    lines: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        title = candidate.title or candidate.breadcrumb or candidate.node_id
        url_part = ""
        if candidate.url:
            anchor = f"#{candidate.anchor}" if candidate.anchor else ""
            url_part = f"\n    url: {candidate.url}{anchor}"
        body = (candidate.indexed_text or candidate.raw_text or "")[:_EVIDENCE_TEXT_CHARS]
        lines.append(f"[{idx}] {candidate.node_label} {title}{url_part}\n    {body}")
    return "\n\n".join(lines)


def _render_prompt(question: str, evidence: list[Candidate]) -> str:
    template = _load_prompt_template()
    return (
        template.replace("{question}", question).replace(
            "{evidence_block}", render_evidence(evidence)
        )
    )


def extract_citation_ids(answer: str) -> list[int]:
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        citation_id = int(match.group(1))
        if citation_id not in seen:
            seen.append(citation_id)
    return seen


def validate_citations(
    answer: str,
    evidence_count: int,
    used_ids: list[int],
) -> tuple[bool, list[int], list[int]]:
    extracted = extract_citation_ids(answer)
    out_of_range = [n for n in extracted if n < 1 or n > evidence_count]
    extracted_set = set(extracted)
    missing = [used_id for used_id in used_ids if used_id not in extracted_set]
    return not out_of_range and not missing, missing, out_of_range


def build_answerer_agent(model: object | None = None) -> Agent[None, AnswerWithCitations]:
    if model is None:
        settings = get_settings()
        model = f"google-gla:{settings.gemini_model}"

    return Agent(
        model=model,
        output_type=AnswerWithCitations,
        system_prompt="Answer with cited JSON only.",
    )


async def answer_with_citations(
    question: str,
    evidence: list[Candidate],
    agent: Agent[None, AnswerWithCitations] | None = None,
) -> AnswerWithCitations:
    answerer = agent if agent is not None else build_answerer_agent()
    result = await answerer.run(_render_prompt(question, evidence))
    return result.output
