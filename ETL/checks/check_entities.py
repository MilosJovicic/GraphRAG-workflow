"""Validate Stage 3 closed-vocabulary entity extraction."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage1_parse import main as parse_main  # noqa: E402
from pipeline.stage2_resolve import main as resolve_main  # noqa: E402
from pipeline.stage3_entities import main as entities_main  # noqa: E402


ALLOWED_ENTITY_LABELS = {
    "Tool",
    "SettingKey",
    "Provider",
    "MessageType",
    "Hook",
    "PermissionMode",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="graphrag-stage3-") as tmp:
        tmp_path = Path(tmp)
        parsed_dir = tmp_path / "parsed"
        resolved_dir = tmp_path / "resolved"
        entities_dir = tmp_path / "entities"

        parse_code = parse_main(["--out", str(parsed_dir)])
        if parse_code != 0:
            print(f"check_entities.py FAILED: stage1 parse exited {parse_code}")
            return 1

        resolve_code = resolve_main(["--in", str(parsed_dir), "--out", str(resolved_dir)])
        if resolve_code != 0:
            print(f"check_entities.py FAILED: stage2 resolve exited {resolve_code}")
            return 1

        entities_code = entities_main(["--in", str(resolved_dir), "--out", str(entities_dir)])
        if entities_code != 0:
            print(f"check_entities.py FAILED: stage3 entities exited {entities_code}")
            return 1

        records: list[dict] = []
        for path in sorted(entities_dir.glob("*.jsonl")):
            records.extend(_read_jsonl(path))

    entity_records = [record for record in records if record["kind"] == "entity"]
    relationship_records = [
        record for record in records if record["kind"] == "relationship"
    ]
    entity_counts = Counter(record["label"] for record in entity_records)
    rel_counts = Counter(record["type"] for record in relationship_records)
    entity_ids = [record["id"] for record in entity_records]
    entity_id_set = set(entity_ids)

    errors: list[str] = []
    if len(entity_ids) != len(entity_id_set):
        errors.append("duplicate entity ids found")

    unexpected_labels = set(entity_counts) - ALLOWED_ENTITY_LABELS
    if unexpected_labels:
        errors.append(f"unexpected entity labels: {sorted(unexpected_labels)}")

    minimums = {
        "Tool": 30,
        "SettingKey": 60,
        "Hook": 25,
        "PermissionMode": 6,
        "MessageType": 5,
    }
    for label, minimum in minimums.items():
        if entity_counts[label] < minimum:
            errors.append(f"expected at least {minimum} {label} entities, got {entity_counts[label]}")

    if entity_counts["Provider"] != 3:
        errors.append(f"expected exactly 3 Provider entities, got {entity_counts['Provider']}")

    required_ids = {
        "Tool:Bash",
        "SettingKey:permissions",
        "Hook:SessionStart",
        "PermissionMode:default",
        "Provider:bedrock",
        "MessageType:SystemMessage",
    }
    missing = sorted(required_ids - entity_id_set)
    if missing:
        errors.append(f"missing required entities: {missing}")

    if rel_counts["DEFINES"] < 50:
        errors.append(f"expected at least 50 DEFINES relationships, got {rel_counts['DEFINES']}")
    if rel_counts["MENTIONS"] < 100:
        errors.append(f"expected at least 100 MENTIONS relationships, got {rel_counts['MENTIONS']}")

    for relationship in relationship_records:
        if relationship["type"] in {"DEFINES", "MENTIONS"} and relationship["target_id"] not in entity_id_set:
            errors.append(
                f"{relationship['type']} targets missing entity {relationship['target_id']}"
            )
            break

    if errors:
        print("check_entities.py FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "check_entities.py OK: "
        f"{sum(entity_counts.values())} entities "
        f"({', '.join(f'{k}={v}' for k, v in sorted(entity_counts.items()))}); "
        f"DEFINES={rel_counts['DEFINES']}, MENTIONS={rel_counts['MENTIONS']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
