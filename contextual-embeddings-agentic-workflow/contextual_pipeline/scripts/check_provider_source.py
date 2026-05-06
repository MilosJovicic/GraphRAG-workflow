import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

with driver.session() as s:
    print("=== Provider source descriptions ===")
    for x in s.run("MATCH (p:Provider) RETURN p.name AS name, p.description AS desc").data():
        print(f"  {x['name']}: {x['desc']!r}")

    print("\n=== PermissionMode parent_text/page (any docs context?) ===")
    for x in s.run("""
        MATCH (pm:PermissionMode {name:'default'})
        OPTIONAL MATCH (pm)<-[r]-(other)
        RETURN type(r) AS rel, labels(other)[0] AS lbl, coalesce(other.text, other.title, other.name) AS preview LIMIT 10
    """).data():
        print(f"  {x['rel']}  ({x['lbl']})  {str(x['preview'])[:120] if x['preview'] else '(no preview)'}")
driver.close()
