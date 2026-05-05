"""Graph expansion from seed candidates through fixed Cypher patterns."""

from __future__ import annotations

from qa_agent.neo4j_client import run_cypher
from qa_agent.schemas import Candidate, ExpansionPattern

_PATTERN_TEMPLATE = {
    "siblings": "expand_siblings.cypher",
    "parent_page": "expand_parent_page.cypher",
    "links": "expand_links.cypher",
    "defines": "expand_defines.cypher",
    "navigates_to": "expand_navigates_to.cypher",
}


def _row_to_candidate(row: dict, pattern: str) -> Candidate:
    return Candidate(
        node_id=row["node_id"],
        node_label=row["node_label"],
        indexed_text=row.get("indexed_text") or "",
        raw_text=row.get("raw_text") or "",
        url=row.get("url"),
        anchor=row.get("anchor"),
        breadcrumb=row.get("breadcrumb"),
        title=row.get("title"),
        expansion_origin=f"{pattern}:{row.get('seed_id', '')}",
    )


async def expand(
    seeds: list[Candidate],
    patterns: list[ExpansionPattern],
    total_cap: int,
) -> list[Candidate]:
    """Return seeds first, then deduped expansions capped separately."""
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []

    for seed in seeds:
        key = (seed.node_id, seed.node_label)
        if key not in seen:
            seen.add(key)
            out.append(seed)

    if not seeds or not patterns or total_cap <= 0:
        return out

    expansions: list[Candidate] = []
    seed_payload = [
        {"node_id": seed.node_id, "node_label": seed.node_label} for seed in seeds
    ]
    for pattern in patterns:
        template = _PATTERN_TEMPLATE.get(pattern.name)
        if not template:
            continue

        rows = await run_cypher(
            template,
            {
                "seeds": seed_payload,
                "cap": pattern.max_per_seed * len(seeds),
            },
        )
        for row in rows:
            candidate = _row_to_candidate(row, pattern.name)
            key = (candidate.node_id, candidate.node_label)
            if key in seen:
                continue
            seen.add(key)
            expansions.append(candidate)
            if len(expansions) >= total_cap:
                break

        if len(expansions) >= total_cap:
            break

    out.extend(expansions[:total_cap])
    return out
