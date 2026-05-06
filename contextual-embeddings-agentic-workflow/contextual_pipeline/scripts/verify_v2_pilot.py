"""Verify v2 contextualization quality gates against Neo4j.

Read-only script. It works for both the corrected pilot and the full v2 rerun.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)


def print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("  (zero rows returned)")
        return
    for row in rows:
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))


with driver.session() as s:
    print("=== V2 COUNTS (context_version=2) ===")
    counts = s.run("""
        MATCH (n) WHERE n.context_version = 2
        WITH labels(n)[0] AS label, n.context_source AS source, count(*) AS total
        RETURN label, source, total
        ORDER BY label, source
    """).data()
    total = sum(row["total"] for row in counts)
    for row in counts:
        print(f"  {row['label']:<16} {row['source']:<9} {row['total']}")
    print(f"  TOTAL: {total}")
    print("  Expected corrected pilot: about 20 per LLM label, all 6 PermissionMode nodes.")
    if total == 48:
        print("  Previous 48-node pilot detected; this is not the corrected 20-node pilot.")

    print("\n=== 1. BOILERPLATE RATES ===")
    boilerplate = s.run("""
        MATCH (n) WHERE n.context_source = 'llm' AND n.context_version = 2
        WITH labels(n)[0] AS label,
             count(*) AS total,
             count(CASE WHEN toLower(n.context) CONTAINS 'related hints' THEN 1 END) AS leak,
             count(CASE WHEN n.context STARTS WITH 'This ' THEN 1 END) AS this_opener,
             count(CASE WHEN toLower(n.context) CONTAINS 'this node defines' THEN 1 END) AS this_defines
        RETURN label, total, leak, this_opener, this_defines,
               round(100.0 * leak / total, 1) AS leak_pct,
               round(100.0 * this_opener / total, 1) AS this_pct
        ORDER BY total DESC
    """).data()
    for row in boilerplate:
        print(
            f"  {row['label']:<16} total={row['total']:<5} "
            f"leak_pct={row['leak_pct']:<4} this_pct={row['this_pct']:<4} "
            f"leak={row['leak']} this={row['this_opener']} this_defines={row['this_defines']}"
        )
    print("  Target: leak_pct < 1%; this_pct < 5%")

    print("\n=== 2. ENTITY FIRST TOKEN ===")
    entity = s.run("""
        MATCH (n) WHERE labels(n)[0] IN ['Tool','Hook','SettingKey','PermissionMode','MessageType','Provider']
          AND n.context_source = 'llm' AND n.context_version = 2
        WITH labels(n)[0] AS label,
             coalesce(n.name, n.key) AS entity_name,
             split(n.context, ':')[0] AS first_token
        RETURN label,
               count(*) AS total,
               count(CASE WHEN first_token = entity_name THEN 1 END) AS starts_with_name
        ORDER BY label
    """).data()
    for row in entity:
        print(f"  {row['label']:<16} {row['starts_with_name']}/{row['total']}")
    print("  Target: starts_with_name = total")

    print("\n=== 3. PERMISSIONMODE COSINE ===")
    pm = s.run("""
        MATCH (pm1:PermissionMode {name:'default'}), (pm2:PermissionMode {name:'plan'})
        WHERE pm1.context_version = 2 AND pm2.context_version = 2
          AND pm1.embedding IS NOT NULL AND pm2.embedding IS NOT NULL
        WITH pm1, pm2,
             reduce(s = 0.0, i IN range(0, size(pm1.embedding)-1) |
                    s + pm1.embedding[i] * pm2.embedding[i]) AS dot,
             sqrt(reduce(s = 0.0, x IN pm1.embedding | s + x*x)) AS n1,
             sqrt(reduce(s = 0.0, x IN pm2.embedding | s + x*x)) AS n2
        RETURN dot/(n1*n2) AS cosine, pm1.context AS default_context, pm2.context AS plan_context
    """).data()
    print_rows("PermissionMode default vs plan", pm)
    print("  Target: < 0.70")

    print("\n=== 4. EQUIVALENT_TO CODEBLOCK COSINE ===")
    equiv = s.run("""
        MATCH (c1:CodeBlock)-[:EQUIVALENT_TO]->(c2:CodeBlock)
        WHERE c1.language = 'python' AND c2.language = 'typescript'
          AND c1.context_version = 2 AND c2.context_version = 2
          AND c1.embedding IS NOT NULL AND c2.embedding IS NOT NULL
        WITH c1, c2,
             reduce(s = 0.0, i IN range(0, size(c1.embedding)-1) |
                    s + c1.embedding[i] * c2.embedding[i]) AS dot,
             sqrt(reduce(s = 0.0, x IN c1.embedding | s + x*x)) AS n1,
             sqrt(reduce(s = 0.0, x IN c2.embedding | s + x*x)) AS n2
        WITH dot/(n1*n2) AS cosine
        RETURN count(*) AS pairs, avg(cosine) AS mean_cosine, percentileCont(cosine, 0.5) AS median
    """).data()
    print_rows("EQUIVALENT_TO Python to TypeScript", equiv)
    print("  Target: mean stays in [0.75, 0.85]")

    print("\n=== 5. FORBIDDEN PHRASES ===")
    forbidden = s.run("""
        MATCH (n) WHERE n.context_source = 'llm' AND n.context_version = 2
        WITH n,
             [phrase IN ['related hints', 'this node defines',
                         'within the documentation', 'in the documentation']
              WHERE toLower(n.context) CONTAINS phrase | phrase] AS hits
        WHERE size(hits) > 0
        RETURN labels(n)[0] AS label, count(*) AS violations
    """).data()
    print_rows("Forbidden phrase violations", forbidden)
    print("  Target: zero rows returned")

    print("\n=== TABLEROW EXAMPLE LOOKUPS ===")
    examples = s.run("""
        MATCH (t:TableRow)
        WHERE t.context_version = 2
          AND t.context_source = 'llm'
          AND (
            any(cell IN coalesce(t.cells, [])
                WHERE toString(cell) CONTAINS 'ANTHROPIC_DEFAULT_OPUS_MODEL_NAME'
                   OR toString(cell) CONTAINS '7 * * * *')
            OR any(header IN coalesce(t.headers, [])
                WHERE toString(header) CONTAINS 'environment variable')
            OR coalesce(t.indexed_text, '') CONTAINS 'ANTHROPIC_DEFAULT_OPUS_MODEL_NAME'
          )
        RETURN t.id AS id, t.headers AS headers, t.cells AS cells, t.context AS context
        LIMIT 10
    """).data()
    for row in examples:
        print(f"  {row['id']}: {row['headers']} | {row['cells']} :: {row['context']}")

driver.close()
