from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qa_agent.schemas import Candidate
from qa_agent.retrieval.rerank import cohere_rerank


def _candidate(node_id: str, text: str) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label="Section",
        indexed_text=text,
        raw_text=text,
    )


@pytest.mark.asyncio
async def test_cohere_rerank_empty_input_returns_empty():
    out = await cohere_rerank("hello", [], top_k=8, doc_chars=1500)

    assert out == []


@pytest.mark.asyncio
async def test_cohere_rerank_truncates_documents_and_reorders(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
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
