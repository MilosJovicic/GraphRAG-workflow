from temporalio import activity
from temporalio.exceptions import ApplicationError
from ..neo4j_client import load_cypher, run_query_async

_WRITE_CYPHER = load_cypher("write_context.cypher")


@activity.defn
async def write_results(rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        result = await run_query_async(_WRITE_CYPHER, {"rows": rows})
    except Exception as exc:
        raise ApplicationError(f"Neo4j write failed: {exc}") from exc

    updated = result[0]["updated"] if result else 0
    expected = len(rows)
    if updated != expected:
        raise ApplicationError(
            f"Neo4j write updated {updated} of {expected} rows",
            non_retryable=True,
        )

    activity.logger.info(f"Wrote {updated} nodes to Neo4j")
    return updated
