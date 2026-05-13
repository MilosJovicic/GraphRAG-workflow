from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.schemas import SubQuery
from qa_agent.retrieval.bm25 import bm25_search


@pytest.mark.asyncio
async def test_bm25_search_section_label_calls_correct_template():
    sq = SubQuery(
        text="how to configure",
        target_labels=["Section"],
        bm25_keywords=["bash"],
    )
    fake_records = [
        {
            "node_id": "s1",
            "node_label": "Section",
            "indexed_text": "x",
            "raw_text": "y",
            "url": "https://example.com",
            "anchor": "h1",
            "breadcrumb": "Home > Bash",
            "title": "Bash",
            "bm25_score": 12.0,
        }
    ]

    with patch(
        "qa_agent.retrieval.bm25.run_cypher",
        new=AsyncMock(return_value=fake_records),
    ) as mocked_run:
        out = await bm25_search(sq, label="Section", limit=50)

    assert len(out) == 1
    assert out[0].node_id == "s1"
    assert out[0].bm25_score == 12.0

    args, _ = mocked_run.call_args
    assert args[0] == "fulltext_section.cypher"
    assert "how to configure" in args[1]["query"]
    assert "bash" in args[1]["query"]
    assert args[1]["limit"] == 50


@pytest.mark.asyncio
async def test_bm25_search_unknown_label_raises():
    sq = SubQuery(text="x", target_labels=["Section"])

    with pytest.raises(ValueError):
        await bm25_search(sq, label="Page", limit=50)


@pytest.mark.asyncio
async def test_bm25_search_drops_unknown_filter_keys():
    sq = SubQuery(
        text="x",
        target_labels=["Section"],
        filters={"language": "python", "bogus": "v"},
    )

    with patch(
        "qa_agent.retrieval.bm25.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await bm25_search(sq, label="CodeBlock", limit=50)

    args, _ = mocked_run.call_args
    params = args[1]
    assert params.get("language") == "python"
    assert "bogus" not in params


@pytest.mark.asyncio
async def test_bm25_search_escapes_lucene_special_characters():
    sq = SubQuery(
        text="Claude Code slash command clear conversation",
        target_labels=["Section"],
        bm25_keywords=["/clear", "conversation", "clear"],
    )

    with patch(
        "qa_agent.retrieval.bm25.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await bm25_search(sq, label="Section", limit=50)

    args, _ = mocked_run.call_args
    query = args[1]["query"]
    assert r"\/clear" in query
    assert " /clear" not in query
