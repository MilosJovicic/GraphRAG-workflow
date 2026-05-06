"""Smoke-test the configured local Neo4j connection."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.neo4j_connector import Neo4jConnector, config_from_env  # noqa: E402


def main() -> int:
    try:
        config = config_from_env()
        with Neo4jConnector(config) as connector:
            connector.verify_connectivity()
            ok = connector.smoke_check()
    except Exception as exc:
        print(f"check_neo4j.py FAILED: {exc}")
        return 1
    if ok != 1:
        print(f"check_neo4j.py FAILED: expected 1, got {ok}")
        return 1
    print("check_neo4j.py OK: Neo4j returned 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
