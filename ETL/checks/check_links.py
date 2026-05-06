"""Validate Stage 2 link and relationship resolution."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage1_parse import PARSED_ROOT, main as parse_main  # noqa: E402
from pipeline.stage2_resolve import RESOLVED_ROOT, main as resolve_main  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="graphrag-stage2-") as tmp:
        tmp_path = Path(tmp)
        parsed_dir = tmp_path / "parsed"
        resolved_dir = tmp_path / "resolved"

        parse_code = parse_main(["--out", str(parsed_dir)])
        if parse_code != 0:
            print(f"check_links.py FAILED: stage1 parse exited {parse_code}")
            return 1

        resolve_code = resolve_main(["--in", str(parsed_dir), "--out", str(resolved_dir)])
        if resolve_code != 0:
            print(f"check_links.py FAILED: stage2 resolve exited {resolve_code}")
            return 1

        records: list[dict] = []
        for path in sorted(resolved_dir.glob("*.jsonl")):
            records.extend(_read_jsonl(path))

    counts = Counter(record["kind"] for record in records)
    rel_counts = Counter(
        record["type"] for record in records if record["kind"] == "relationship"
    )
    unresolved = [record for record in records if record["kind"] == "unresolved_link"]
    bad_relationships = [
        record
        for record in records
        if record["kind"] == "relationship"
        and record["type"] in {"LINKS_TO", "LINKS_TO_PAGE", "NAVIGATES_TO"}
        and not record.get("target_id")
    ]

    errors: list[str] = []
    if counts["relationship"] == 0:
        errors.append("expected relationship records")
    if rel_counts["LINKS_TO"] == 0:
        errors.append("expected at least one LINKS_TO relationship")
    if rel_counts["LINKS_TO_PAGE"] == 0:
        errors.append("expected at least one LINKS_TO_PAGE relationship")
    if rel_counts["NAVIGATES_TO"] == 0:
        errors.append("expected at least one NAVIGATES_TO relationship")
    if rel_counts["EQUIVALENT_TO"] == 0:
        errors.append("expected at least one EQUIVALENT_TO relationship")
    if bad_relationships:
        errors.append(f"{len(bad_relationships)} relationships have no target_id")

    if errors:
        print("check_links.py FAILED")
        for error in errors:
            print(f"- {error}")
        if unresolved:
            print(f"warnings: {len(unresolved)} unresolved internal links")
        return 1

    print(
        "check_links.py OK: "
        f"{counts['relationship']} relationships "
        f"({', '.join(f'{k}={v}' for k, v in sorted(rel_counts.items()))}); "
        f"{len(unresolved)} unresolved internal links warned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
