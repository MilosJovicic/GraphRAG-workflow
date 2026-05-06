"""Verify v2-plan criticism's claims against actual Neo4j state."""
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# .env lives at project root (parent of contextual_pipeline)
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")

if not URI:
    sys.exit("NEO4J_URI not loaded")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

with driver.session() as s:
    print("=== COUNTS BY SOURCE & LABEL ===")
    res = s.run("""
        MATCH (n) WHERE n.context IS NOT NULL
        WITH labels(n)[0] AS label, n.context_source AS source, count(*) AS c
        RETURN label, source, c ORDER BY label, source
    """).data()
    for r in res:
        print(f"  {r['label']:<20} {r['source']:<10} {r['c']}")

    print("\n=== BOILERPLATE PHRASE RATES ON LLM-SOURCED CONTEXTS ===")
    print(f"  {'label':<16} {'total':>6} {'this_op':>8} {'rel_hint':>9} {'this_def':>9} {'distinct':>9}")
    res = s.run("""
        MATCH (n) WHERE n.context_source = 'llm'
        WITH labels(n)[0] AS label, n.context AS ctx
        RETURN label,
               count(*) AS total,
               sum(CASE WHEN ctx STARTS WITH 'This ' OR ctx STARTS WITH 'this ' THEN 1 ELSE 0 END) AS this_op,
               sum(CASE WHEN toLower(ctx) CONTAINS 'related hints' THEN 1 ELSE 0 END) AS rel_hint,
               sum(CASE WHEN toLower(ctx) CONTAINS 'this node defines' THEN 1 ELSE 0 END) AS this_def,
               sum(CASE WHEN toLower(ctx) CONTAINS 'distinct from' THEN 1 ELSE 0 END) AS distinct_p
        ORDER BY total DESC
    """).data()
    for r in res:
        t = max(r['total'], 1)
        print(
            f"  {r['label']:<16} {r['total']:>6} "
            f"{r['this_op']:>4} ({100*r['this_op']/t:>4.1f}%) "
            f"{r['rel_hint']:>4} ({100*r['rel_hint']/t:>4.1f}%) "
            f"{r['this_def']:>4} ({100*r['this_def']/t:>4.1f}%) "
            f"{r['distinct_p']:>4} ({100*r['distinct_p']/t:>4.1f}%)"
        )

    print("\n=== EXAMPLES CITED IN CRITICISM ===")
    print("\n-- ANTHROPIC_DEFAULT_OPUS_MODEL_NAME tablerow --")
    r = s.run("""
        MATCH (t:TableRow) WHERE t.text CONTAINS 'ANTHROPIC_DEFAULT_OPUS_MODEL_NAME'
        RETURN t.text AS text, t.context AS ctx LIMIT 1
    """).single()
    if r:
        print(f"  text: {r['text'][:140]}")
        print(f"  ctx:  {r['ctx']}")
    else:
        print("  (not found)")

    print("\n-- 'cron' tablerows --")
    r = s.run("""
        MATCH (t:TableRow) WHERE t.text CONTAINS 'cron' OR t.text CONTAINS '* * *'
        RETURN t.text AS text, t.context AS ctx LIMIT 2
    """).data()
    for x in r:
        print(f"  text: {x['text'][:140]}")
        print(f"  ctx:  {x['ctx']}\n")

    print("-- PermissionMode default vs plan --")
    r = s.run("""
        MATCH (pm:PermissionMode) WHERE pm.name IN ['default', 'plan']
        RETURN pm.name AS name, pm.description AS desc, pm.context AS ctx
    """).data()
    for x in r:
        print(f"  {x['name']}:")
        print(f"    desc: {x['desc']}")
        print(f"    ctx:  {x['ctx']}\n")

    print("=== ALL PERMISSIONMODES ===")
    r = s.run("MATCH (pm:PermissionMode) RETURN pm.name AS name, pm.description AS desc ORDER BY pm.name").data()
    for x in r:
        print(f"  {x['name']}: {x['desc']}")

    print("\n=== COSINE: default vs plan ===")
    r = s.run("""
        MATCH (pm1:PermissionMode {name:'default'}), (pm2:PermissionMode {name:'plan'})
        WHERE pm1.embedding IS NOT NULL AND pm2.embedding IS NOT NULL
        WITH pm1, pm2,
             reduce(s=0.0, i IN range(0,size(pm1.embedding)-1) | s + pm1.embedding[i]*pm2.embedding[i]) AS dot,
             sqrt(reduce(s=0.0, x IN pm1.embedding | s + x*x)) AS n1,
             sqrt(reduce(s=0.0, x IN pm2.embedding | s + x*x)) AS n2
        RETURN dot/(n1*n2) AS cosine
    """).single()
    print(f"  cosine(default, plan) = {r['cosine'] if r else 'no embeddings'}")

    print("\n=== EQUIVALENT_TO Python<->TS pairs (sanity) ===")
    r = s.run("""
        MATCH (c1:CodeBlock)-[:EQUIVALENT_TO]->(c2:CodeBlock)
        WHERE c1.language = 'python' AND c2.language = 'typescript'
          AND c1.embedding IS NOT NULL AND c2.embedding IS NOT NULL
        WITH c1, c2,
             reduce(s=0.0, i IN range(0,size(c1.embedding)-1) | s + c1.embedding[i]*c2.embedding[i]) AS dot,
             sqrt(reduce(s=0.0, x IN c1.embedding | s + x*x)) AS n1,
             sqrt(reduce(s=0.0, x IN c2.embedding | s + x*x)) AS n2
        WITH dot/(n1*n2) AS cosine
        RETURN avg(cosine) AS mean, percentileCont(cosine, 0.5) AS median, count(*) AS n
    """).single()
    if r:
        print(f"  py<->ts pairs: n={r['n']}, mean={r['mean']:.3f}, median={r['median']:.3f}")

driver.close()
