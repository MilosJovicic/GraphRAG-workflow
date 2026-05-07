from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.schemas import SubQuery
from qa_agent.retrieval.vector import vector_search


@pytest.mark.asyncio
async def test_vector_search_embeds_then_queries():
    sq = SubQuery(text="permission mode", target_labels=["Section"])
    fake_records = [
        {
            "node_id": "s1",
            "node_label": "Section",
            "indexed_text": "x",
            "raw_text": "y",
            "url": "u",
            "anchor": "a",
            "breadcrumb": "b",
            "title": "t",
            "vector_score": 0.91,
        }
    ]

    with (
        patch(
            "qa_agent.retrieval.vector.embed_one",
            new=AsyncMock(return_value=[0.1] * 1024),
        ) as embed,
        patch(
            "qa_agent.retrieval.vector.run_cypher",
            new=AsyncMock(return_value=fake_records),
        ) as run,
    ):
        out = await vector_search(sq, label="Section", limit=50)

    embed.assert_awaited_once_with("permission mode")
    args, _ = run.call_args
    assert args[0] == "vector_section.cypher"
    assert args[1]["limit"] == 50
    assert len(args[1]["query_vector"]) == 1024
    assert out[0].vector_score == 0.91


@pytest.mark.asyncio
async def test_vector_search_unknown_label_raises():
    sq = SubQuery(text="x", target_labels=["Section"])

    with pytest.raises(ValueError):
        await vector_search(sq, label="Page", limit=50)
