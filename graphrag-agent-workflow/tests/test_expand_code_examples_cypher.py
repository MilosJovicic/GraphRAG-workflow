"""Graph-integration test for the code_examples expansion template.

Gated on QA_RUN_INTEGRATION=1; requires a live Neo4j with the Claude Code
docs graph loaded.
"""

from __future__ import annotations

import os

import pytest

from qa_agent.neo4j_client import get_async_driver, run_cypher

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("QA_RUN_INTEGRATION") != "1",
        reason="set QA_RUN_INTEGRATION=1 with Neo4j running",
    ),
]

GOLD_IDS = {"dd7564204c2946d7", "5f644bc9ef3abd5f"}


@pytest.fixture(autouse=True)
async def _fresh_async_driver():
    """Each test gets a fresh AsyncDriver bound to its own event loop.

    pytest-asyncio gives each test function a new loop; the lru_cached
    driver from a previous test is bound to a closed loop and crashes on
    Windows proactor cleanup. Clearing the cache and closing the driver
    after each test isolates the loops.
    """
    yield
    driver = get_async_driver()
    await driver.close()
    get_async_driver.cache_clear()


@pytest.mark.asyncio
async def test_code_examples_returns_gold_edit_codeblocks_in_top_6():
    rows = await run_cypher(
        "expand_code_examples.cypher",
        {
            "seeds": [{"node_id": "Edit", "node_label": "Tool"}],
            "cap": 6,
            "language": "python",
        },
    )
    ids = [row["node_id"] for row in rows]
    hits = GOLD_IDS.intersection(ids)
    assert hits, f"Neither gold codeblock found in top 6 results. Got: {ids}"


@pytest.mark.asyncio
async def test_code_examples_is_noop_for_non_entity_seed():
    rows = await run_cypher(
        "expand_code_examples.cypher",
        {
            "seeds": [
                {
                    "node_id": "https://code.claude.com/docs/en/quickstart.md",
                    "node_label": "Section",
                }
            ],
            "cap": 6,
            "language": "python",
        },
    )
    assert rows == []
