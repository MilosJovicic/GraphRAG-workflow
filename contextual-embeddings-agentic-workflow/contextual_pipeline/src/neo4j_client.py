from pathlib import Path
from functools import lru_cache
from neo4j import GraphDatabase, AsyncGraphDatabase
from .config import get_settings

_CYPHER_DIR = Path(__file__).parent / "cypher"


def load_cypher(relative_path: str) -> str:
    return (_CYPHER_DIR / relative_path).read_text(encoding="utf-8")


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
