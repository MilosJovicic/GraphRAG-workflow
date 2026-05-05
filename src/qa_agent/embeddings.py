"""Ollama OpenAI-compatible embedding client for Qwen 3 0.6B."""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from qa_agent.config import get_settings


@lru_cache
def _get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(base_url=s.ollama_base_url, api_key="ollama")


async def embed_one(text: str) -> list[float]:
    s = get_settings()
    client = _get_client()
    resp = await client.embeddings.create(model=s.embedding_model, input=text)
    return list(resp.data[0].embedding)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = _get_client()
    resp = await client.embeddings.create(model=s.embedding_model, input=texts)
    by_index = {d.index: list(d.embedding) for d in resp.data}
    return [by_index[i] for i in range(len(texts))]
