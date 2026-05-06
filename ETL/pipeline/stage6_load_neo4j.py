"""Stage 6 - Neo4j connector utilities.

Run:
    python -m pipeline.stage6_load_neo4j check
    python -m pipeline.stage6_load_neo4j schema
    python -m pipeline.stage6_load_neo4j load
    python -m pipeline.stage6_load_neo4j wipe --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .neo4j_connector import Neo4jConnector, config_from_env
from .stage1_parse import REPO_ROOT


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uri", default=None, help="Neo4j URI. Defaults to NEO4J_URI.")
    parser.add_argument("--user", default=None, help="Neo4j user. Defaults to NEO4J_USER.")
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password. Prefer NEO4J_PASSWORD to avoid shell history.",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Optional database name. Defaults to NEO4J_DATABASE.",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 - Neo4j connector and loader.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify Neo4j connectivity.")
    _add_connection_args(check_parser)

    schema_parser = subparsers.add_parser("schema", help="Apply schema.cypher.")
    _add_connection_args(schema_parser)
    schema_parser.add_argument(
        "--schema",
        default=str(REPO_ROOT / "schema.cypher"),
        help="Path to a Cypher schema file.",
    )

    load_parser = subparsers.add_parser("load", help="Load Stage 1-3 outputs into Neo4j.")
    _add_connection_args(load_parser)
    load_parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        help="Input dir, defaults to ./entities/.",
    )
    load_parser.add_argument("--batch-size", type=int, default=500)
    load_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate without executing Cypher.",
    )

    wipe_parser = subparsers.add_parser("wipe", help="Delete all loader-owned label nodes.")
    _add_connection_args(wipe_parser)
    wipe_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation. Required for non-interactive use.",
    )

    args = parser.parse_args(argv)
    try:
        config = config_from_env(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        with Neo4jConnector(config) as connector:
            if args.command == "check":
                connector.verify_connectivity()
                ok = connector.smoke_check()
                print(f"Neo4j connection OK: RETURN 1 -> {ok}")
                return 0
            if args.command == "schema":
                count = connector.apply_schema(Path(args.schema))
                print(f"Applied {count} schema statements")
                return 0
            if args.command == "load":
                from .stage6_load import ENTITIES_ROOT, load

                in_dir = Path(args.in_dir) if args.in_dir else ENTITIES_ROOT
                counts = load(connector, in_dir, batch_size=args.batch_size, dry_run=args.dry_run)
                _print_load_summary(counts, dry_run=args.dry_run)
                return 0
            if args.command == "wipe":
                if not args.yes:
                    print("error: refusing to wipe without --yes", file=sys.stderr)
                    return 1
                from .stage6_load import wipe

                wipe(connector)
                print("wiped all loader-owned label nodes")
                return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _print_load_summary(counts: dict, *, dry_run: bool) -> None:
    prefix = "would load" if dry_run else "loaded"
    nodes = counts["nodes"]
    rels = counts["relationships"]
    node_total = sum(nodes.values())
    rel_total = sum(rels.values())
    node_breakdown = ", ".join(f"{key}={value}" for key, value in sorted(nodes.items()) if value)
    rel_breakdown = ", ".join(f"{key}={value}" for key, value in sorted(rels.items()) if value)
    print(
        f"{prefix} {counts['pages_loaded']} pages "
        f"({counts['pages_skipped']} skipped, {counts['pages_replaced']} replaced): "
        f"{node_total} nodes ({node_breakdown}), {rel_total} relationships ({rel_breakdown})"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
