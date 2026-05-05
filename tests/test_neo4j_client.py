import pytest

from qa_agent import neo4j_client
from qa_agent.neo4j_client import load_cypher, run_cypher


def test_load_cypher_existing_file():
    txt = load_cypher("vector_section.cypher")
    assert "db.index.vector.queryNodes" in txt
    assert "section_embedding" in txt


def test_load_cypher_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_cypher("does_not_exist.cypher")


@pytest.mark.asyncio
async def test_run_cypher_loads_template_and_delegates(monkeypatch):
    calls = {}

    async def fake_run_query_async(cypher: str, parameters: dict | None = None) -> list[dict]:
        calls["cypher"] = cypher
        calls["parameters"] = parameters
        return [{"ok": True}]

    monkeypatch.setattr(neo4j_client, "run_query_async", fake_run_query_async)
    out = await run_cypher("vector_section.cypher", {"limit": 1})

    assert out == [{"ok": True}]
    assert "section_embedding" in calls["cypher"]
    assert calls["parameters"] == {"limit": 1}
