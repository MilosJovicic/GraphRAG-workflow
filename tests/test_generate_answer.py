from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.activities.generate import generate_answer
from qa_agent.schemas import (
    AnswerWithCitations,
    Candidate,
    ExpansionPattern,
    GenerateRequest,
    Plan,
    SubQuery,
)


def _evidence(node_id: str = "s1") -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label="Section",
        indexed_text="x",
        raw_text="y",
        rerank_score=0.5,
    )


def _plan() -> Plan:
    return Plan(
        sub_queries=[SubQuery(text="q", target_labels=["Section"])],
        expansion_patterns=[ExpansionPattern(name="parent_page")],
    )


@pytest.mark.asyncio
async def test_generate_answer_does_not_flag_no_results_when_evidence_present_but_answerer_refuses():
    refusal = "The provided documentation does not contain a clear answer to this question."
    request = GenerateRequest(
        question="Show me a Python example of using the Edit tool.",
        evidence=[_evidence("e1"), _evidence("e2")],
        plan=_plan(),
    )

    with patch(
        "qa_agent.activities.generate.answer_with_citations",
        new=AsyncMock(return_value=AnswerWithCitations(answer=refusal, used_citation_ids=[])),
    ):
        response = await generate_answer(request)

    assert response.answer == refusal
    assert response.citations == []
    assert response.no_results is False


@pytest.mark.asyncio
async def test_generate_answer_flags_no_results_when_evidence_is_empty():
    request = GenerateRequest(
        question="anything",
        evidence=[],
        plan=_plan(),
    )

    with patch(
        "qa_agent.activities.generate.answer_with_citations",
        new=AsyncMock(
            return_value=AnswerWithCitations(answer="anything", used_citation_ids=[]),
        ),
    ):
        response = await generate_answer(request)

    assert response.no_results is True


@pytest.mark.asyncio
async def test_generate_answer_returns_citations_when_answer_uses_them():
    request = GenerateRequest(
        question="q",
        evidence=[_evidence("e1"), _evidence("e2")],
        plan=_plan(),
    )

    with patch(
        "qa_agent.activities.generate.answer_with_citations",
        new=AsyncMock(
            return_value=AnswerWithCitations(
                answer="The answer is X [1] and Y [2].",
                used_citation_ids=[1, 2],
            )
        ),
    ):
        response = await generate_answer(request)

    assert response.no_results is False
    assert [c.id for c in response.citations] == [1, 2]
