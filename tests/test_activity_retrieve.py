from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.activities.retrieve import _hybrid_for_label, _merge_legs
from qa_agent.schemas import Candidate, SubQuery


def _candidate(
    node_id: str,
    *,
    bm25: float | None = None,
    vector: float | None = None,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label="Section",
        indexed_text="x",
        raw_text="y",
        bm25_score=bm25,
        vector_score=vector,
    )


@pytest.mark.asyncio
async def test_hybrid_for_label_routes_entity_label_to_entity_lookup():
    sq = SubQuery(
        text="configure Bash tool permission mode",
        target_labels=["Tool"],
        bm25_keywords=["Bash", "permission"],
    )

    with (
        patch(
            "qa_agent.activities.retrieve.entity_lookup",
            new=AsyncMock(return_value=[_candidate("Tool:Bash")]),
        ) as mocked_entity,
        patch(
            "qa_agent.activities.retrieve.bm25_search",
            new=AsyncMock(return_value=[]),
        ) as mocked_bm25,
        patch(
            "qa_agent.activities.retrieve.vector_search",
            new=AsyncMock(return_value=[]),
        ) as mocked_vector,
    ):
        out = await _hybrid_for_label(sq, "Tool")

    assert [c.node_id for c in out] == ["Tool:Bash"]
    mocked_entity.assert_awaited_once()
    mocked_bm25.assert_not_awaited()
    mocked_vector.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_for_label_routes_searchable_label_to_bm25_and_vector():
    sq = SubQuery(
        text="configure Bash tool permission mode",
        target_labels=["Section"],
        bm25_keywords=["Bash"],
    )

    with (
        patch(
            "qa_agent.activities.retrieve.entity_lookup",
            new=AsyncMock(return_value=[]),
        ) as mocked_entity,
        patch(
            "qa_agent.activities.retrieve.bm25_search",
            new=AsyncMock(return_value=[_candidate("s1", bm25=10.0)]),
        ) as mocked_bm25,
        patch(
            "qa_agent.activities.retrieve.vector_search",
            new=AsyncMock(return_value=[_candidate("s1", vector=0.9)]),
        ) as mocked_vector,
    ):
        out = await _hybrid_for_label(sq, "Section")

    mocked_entity.assert_not_awaited()
    mocked_bm25.assert_awaited_once()
    mocked_vector.assert_awaited_once()
    assert len(out) == 1
    assert out[0].bm25_score == 10.0
    assert out[0].vector_score == 0.9


def test_merge_legs_dedupes_and_combines_scores():
    bm25 = [_candidate("a", bm25=10.0), _candidate("b", bm25=5.0)]
    vector = [_candidate("a", vector=0.9), _candidate("c", vector=0.5)]

    out = _merge_legs(bm25, vector)

    by_id = {candidate.node_id: candidate for candidate in out}
    assert by_id["a"].bm25_score == 10.0
    assert by_id["a"].vector_score == 0.9
    assert by_id["b"].bm25_score == 5.0
    assert by_id["b"].vector_score is None
    assert by_id["c"].bm25_score is None
    assert by_id["c"].vector_score == 0.5
    assert len(out) == 3
