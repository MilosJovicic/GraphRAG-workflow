from temporalio import activity
from temporalio.exceptions import ApplicationError
from openai import AsyncOpenAI
from ..config import get_settings


def _get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(base_url=s.embedding_base_url, api_key="ollama")


@activity.defn
async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = _get_client()
    batch_size = s.embedding_batch_size
    results: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        activity.heartbeat(f"embedding chunk {i // batch_size + 1}")
        try:
            response = await client.embeddings.create(
                model=s.embedding_model,
                input=chunk,
            )
        except Exception as exc:
            raise ApplicationError(
                f"Embedding API error at chunk {i}: {exc}"
            ) from exc

        for item in sorted(response.data, key=lambda x: x.index):
            results.append(item.embedding)

    return results
