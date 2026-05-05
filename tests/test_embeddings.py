from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qa_agent.embeddings import embed_batch, embed_one


def _set_required_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")


@pytest.mark.asyncio
async def test_embed_one_returns_vector(monkeypatch):
    _set_required_env(monkeypatch)
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1024)]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    with patch("qa_agent.embeddings._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        v = await embed_one("hello world")

    assert len(v) == 1024
    assert v[0] == 0.1
    fake_client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_input(monkeypatch):
    _set_required_env(monkeypatch)
    fake_response = MagicMock()
    fake_response.data = [MagicMock(index=i, embedding=[float(i)] * 4) for i in range(3)]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    with patch("qa_agent.embeddings._get_client", return_value=fake_client):
        from qa_agent.config import get_settings

        get_settings.cache_clear()
        vs = await embed_batch(["a", "b", "c"])

    assert len(vs) == 3
    assert vs[0] == [0.0, 0.0, 0.0, 0.0]
    assert vs[2] == [2.0, 2.0, 2.0, 2.0]


@pytest.mark.asyncio
async def test_embed_batch_empty_returns_empty():
    assert await embed_batch([]) == []
