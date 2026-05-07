from functools import lru_cache
from pathlib import Path

from neo4j import AsyncGraphDatabase, GraphDatabase

from qa_agent.config import get_settings

_CYPHER_DIR = Path(__file__).parent / "cypher"


def load_cypher(relative_path: str) -> str:
    path = _CYPHER_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Cypher template not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache
def get_driver():
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


@lru_cache
def get_async_driver():
    s = get_settings()
    return AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def run_query(cypher: str, parameters: dict | None = None) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, parameters or {})
        return [dict(record) for record in result]


async def run_query_async(cypher: str, parameters: dict | None = None) -> list[dict]:
    driver = get_async_driver()
    async with driver.session() as session:
        result = await session.run(cypher, parameters or {})
        return [dict(record) async for record in result]


async def run_cypher(filename: str, parameters: dict | None = None) -> list[dict]:
    """Load a parameterized .cypher template and execute it against Neo4j."""
    return await run_query_async(load_cypher(filename), parameters)
