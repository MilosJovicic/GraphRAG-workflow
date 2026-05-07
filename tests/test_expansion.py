from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.schemas import Candidate, ExpansionPattern
from qa_agent.retrieval.expansion import expand


def _seed(node_id: str, label: str = "Section") -> Candidate:
    return Candidate(node_id=node_id, node_label=label, indexed_text="x", raw_text="y")


@pytest.mark.asyncio
async def test_expand_dispatches_per_pattern_and_dedupes():
    seeds = [_seed("s1"), _seed("s2")]
    patterns = [ExpansionPattern(name="links", max_per_seed=2)]
    fake_links_rows = [
        {
            "node_id": "t1",
            "node_label": "Section",
            "indexed_text": "x",
            "raw_text": "y",
            "url": None,
            "anchor": None,
            "breadcrumb": None,
            "title": "T1",
            "seed_id": "s1",
        },
        {
            "node_id": "t1",
            "node_label": "Section",
            "indexed_text": "x",
            "raw_text": "y",
            "url": None,
            "anchor": None,
            "breadcrumb": None,
            "title": "T1",
            "seed_id": "s2",
        },
    ]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=fake_links_rows),
    ) as mocked_run:
        out = await expand(seeds, patterns, total_cap=20)

    assert [c.node_id for c in out[:2]] == ["s1", "s2"]
    expanded = [c for c in out if c.node_id == "t1"]
    assert len(expanded) == 1
    assert expanded[0].expansion_origin
    assert expanded[0].expansion_origin.startswith("links:")

    args, _ = mocked_run.call_args
    assert args[0] == "expand_links.cypher"


@pytest.mark.asyncio
async def test_expand_total_cap_truncates_expansions_but_not_seeds():
    seeds = [_seed(f"s{i}") for i in range(3)]
    patterns = [ExpansionPattern(name="siblings", max_per_seed=5)]
    rows = [
        {
            "node_id": f"e{i}",
            "node_label": "Chunk",
            "indexed_text": "x",
            "raw_text": "y",
            "url": None,
            "anchor": None,
            "breadcrumb": None,
            "title": None,
            "seed_id": "s0",
        }
        for i in range(50)
    ]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=rows),
    ):
        out = await expand(seeds, patterns, total_cap=4)

    assert len(out) == 7
    assert {c.node_id for c in out[:3]} == {"s0", "s1", "s2"}


@pytest.mark.asyncio
async def test_expand_no_patterns_returns_seeds_only():
    seeds = [_seed("s1")]

    out = await expand(seeds, [], total_cap=20)

    assert [c.node_id for c in out] == ["s1"]


@pytest.mark.asyncio
async def test_expand_dispatches_code_examples_to_correct_template():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [ExpansionPattern(name="code_examples", max_per_seed=4)]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    assert args[0] == "expand_code_examples.cypher"


@pytest.mark.asyncio
async def test_expand_passes_language_param_for_code_examples():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [
        ExpansionPattern(name="code_examples", max_per_seed=6, language="python")
    ]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    assert args[0] == "expand_code_examples.cypher"
    params = args[1]
    assert params["language"] == "python"
    assert params["cap"] == 6


@pytest.mark.asyncio
async def test_expand_passes_null_language_when_unset():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [ExpansionPattern(name="code_examples", max_per_seed=4)]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    params = args[1]
    assert params["language"] is None
