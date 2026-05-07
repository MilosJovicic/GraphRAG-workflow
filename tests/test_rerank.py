from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qa_agent.schemas import Candidate
from qa_agent.retrieval.rerank import cohere_rerank


def _candidate(
    node_id: str,
    text: str,
    node_label: str = "Section",
    *,
    bm25_score: float | None = None,
    expansion_origin: str | None = None,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label=node_label,
        indexed_text=text,
        raw_text=text,
        bm25_score=bm25_score,
        expansion_origin=expansion_origin,
    )


def _set_envs(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")


@pytest.mark.asyncio
async def test_cohere_rerank_empty_input_returns_empty():
    out = await cohere_rerank("hello", [], top_k=8, doc_chars=1500)

    assert out == []


@pytest.mark.asyncio
async def test_cohere_rerank_truncates_documents_and_reorders(monkeypatch):
    _set_envs(monkeypatch)
    cands = [
        _candidate("a", "A" * 3000),
        _candidate("b", "B" * 100),
        _candidate("c", "C" * 100),
    ]
    fake_resp = MagicMock(
        results=[
            MagicMock(index=2, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.6),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank("q", cands, top_k=2, doc_chars=1500)

    assert [candidate.node_id for candidate in out] == ["c", "a"]
    assert out[0].rerank_score == 0.9
    assert out[1].rerank_score == 0.6

    _, kwargs = fake_client.rerank.call_args
    docs = kwargs["documents"]
    assert all(len(doc) <= 1500 for doc in docs)
    assert kwargs["top_n"] == 2
    assert kwargs["query"] == "q"


@pytest.mark.asyncio
async def test_cohere_rerank_promotes_entity_when_top_k_otherwise_excludes_it(monkeypatch):
    """When entity_budget>=1, the highest-scored entity must appear in the
    returned top_k even if Cohere ranked it below all the doc candidates."""
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200, node_label="Section"),
        _candidate("doc-b", "B" * 200, node_label="Section"),
        _candidate("doc-c", "C" * 200, node_label="Section"),
        _candidate("Tool:Bash", "Bash tool", node_label="Tool"),
    ]
    # Cohere scores all four; entity comes last
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.80),
            MagicMock(index=2, relevance_score=0.60),
            MagicMock(index=3, relevance_score=0.30),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=3, doc_chars=1500, entity_budget=1
        )

    ids = [c.node_id for c in out]
    assert "Tool:Bash" in ids, ids
    assert len(out) == 3
    # The lowest-scored doc (doc-c) should have been swapped for the entity
    assert "doc-c" not in ids


@pytest.mark.asyncio
async def test_cohere_rerank_does_not_swap_when_entity_already_in_top_k(monkeypatch):
    _set_envs(monkeypatch)
    cands = [
        _candidate("Tool:Bash", "Bash", node_label="Tool"),
        _candidate("doc-a", "A" * 200, node_label="Section"),
        _candidate("doc-b", "B" * 200, node_label="Section"),
    ]
    # Entity scored highest naturally
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.80),
            MagicMock(index=2, relevance_score=0.60),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=2, doc_chars=1500, entity_budget=1
        )

    ids = [c.node_id for c in out]
    assert ids == ["Tool:Bash", "doc-a"], ids


@pytest.mark.asyncio
async def test_cohere_rerank_caps_swaps_at_entity_budget(monkeypatch):
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200, node_label="Section"),
        _candidate("doc-b", "B" * 200, node_label="Section"),
        _candidate("doc-c", "C" * 200, node_label="Section"),
        _candidate("Tool:Bash", "Bash", node_label="Tool"),
        _candidate("SettingKey:permissions", "permissions", node_label="SettingKey"),
        _candidate("PermissionMode:acceptEdits", "acceptEdits", node_label="PermissionMode"),
    ]
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.90),
            MagicMock(index=2, relevance_score=0.85),
            MagicMock(index=3, relevance_score=0.40),
            MagicMock(index=4, relevance_score=0.35),
            MagicMock(index=5, relevance_score=0.30),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=4, doc_chars=1500, entity_budget=2
        )

    ids = [c.node_id for c in out]
    entity_ids = [
        i for i in ids if i.startswith(("Tool:", "SettingKey:", "PermissionMode:"))
    ]
    # Exactly 2 entities promoted (the two highest-scoring entities)
    assert len(entity_ids) == 2, ids
    assert "Tool:Bash" in entity_ids
    assert "SettingKey:permissions" in entity_ids
    # Top docs preserved
    assert "doc-a" in ids
    assert "doc-b" in ids


@pytest.mark.asyncio
async def test_cohere_rerank_no_entities_in_input_unchanged(monkeypatch):
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200),
        _candidate("doc-b", "B" * 200),
    ]
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.5),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=1, doc_chars=1500, entity_budget=2
        )

    assert [c.node_id for c in out] == ["doc-a"]


@pytest.mark.asyncio
async def test_cohere_rerank_prefers_lookup_entities_over_expansion_entities(monkeypatch):
    """An entity from entity_lookup (bm25_score set) should fill the budget
    before an entity from graph expansion (expansion_origin set), even if
    Cohere ranked the expansion entity higher."""
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200, node_label="Section"),
        _candidate("doc-b", "B" * 200, node_label="Section"),
        _candidate("doc-c", "C" * 200, node_label="Section"),
        _candidate(
            "PermissionMode:default",
            "Default permission mode prompts before tool actions.",
            node_label="PermissionMode",
            expansion_origin="defines:doc-a",
        ),
        _candidate(
            "Tool:Bash",
            "Bash",
            node_label="Tool",
            bm25_score=3.0,
        ),
    ]
    # Cohere ranks the rich-text expansion entity above the terse lookup entity
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.95),
            MagicMock(index=1, relevance_score=0.85),
            MagicMock(index=2, relevance_score=0.75),
            MagicMock(index=3, relevance_score=0.55),  # PermissionMode (expansion)
            MagicMock(index=4, relevance_score=0.30),  # Tool:Bash (lookup)
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=4, doc_chars=1500, entity_budget=1
        )

    ids = [c.node_id for c in out]
    assert "Tool:Bash" in ids, ids
    assert "PermissionMode:default" not in ids, ids


@pytest.mark.asyncio
async def test_cohere_rerank_falls_back_to_expansion_entities_when_no_lookup_hits(
    monkeypatch,
):
    """If only expansion entities exist (no lookup hits), they still get the
    budget — better than no entities at all."""
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200, node_label="Section"),
        _candidate("doc-b", "B" * 200, node_label="Section"),
        _candidate(
            "Hook:PostToolUse",
            "Hook description from defines",
            node_label="Hook",
            expansion_origin="defines:doc-a",
        ),
    ]
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.8),
            MagicMock(index=2, relevance_score=0.4),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        out = await cohere_rerank(
            "q", cands, top_k=2, doc_chars=1500, entity_budget=1
        )

    assert "Hook:PostToolUse" in [c.node_id for c in out]


@pytest.mark.asyncio
async def test_cohere_rerank_requests_full_scoring_when_entity_budget_set(monkeypatch):
    """With entity_budget>0 we need scores for all candidates, not just top_k,
    so the reranker call must request top_n=len(candidates)."""
    _set_envs(monkeypatch)
    cands = [
        _candidate("doc-a", "A" * 200),
        _candidate("doc-b", "B" * 200),
        _candidate("Tool:Bash", "Bash", node_label="Tool"),
    ]
    fake_resp = MagicMock(
        results=[
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.5),
            MagicMock(index=2, relevance_score=0.2),
        ]
    )
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        await cohere_rerank("q", cands, top_k=2, doc_chars=1500, entity_budget=1)

    _, kwargs = fake_client.rerank.call_args
    assert kwargs["top_n"] == 3
