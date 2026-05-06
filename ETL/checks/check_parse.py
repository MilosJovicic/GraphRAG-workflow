"""Validate Stage 1 parser output against the CLAUDE.md contract."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage1_parse import (  # noqa: E402
    EN_ROOT,
    iter_corpus_files,
    load_llms_descriptions,
    parse_one,
)


LEAK_PATTERNS = [
    re.compile(r"\bexport\s+(const|default|function|let|var)\b"),
    re.compile(r"</?style\b", re.IGNORECASE),
    re.compile(
        r"</?(Tabs|Tab|Note|Tip|Info|Warning|Check|Steps|Step|Accordion|"
        r"AccordionGroup|Card|CardGroup|Expandable|ParamField|ResponseField|"
        r"RequestExample|ResponseExample|CodeGroup|Frame|Experiment|Update)\b"
    ),
    re.compile(r"<div\b[^>]*(data-gb-slot|install-configurator-slot)", re.IGNORECASE),
    re.compile(r"^\{[A-Za-z_$][\w$.\[\]'\"]*\.map\(", re.MULTILINE),
]

EXPECTED_UPDATE_COUNTS = {
    "en/changelog.md": 276,
    "en/whats-new/index.md": 5,
}
EXPECTED_MIN_TABLE_ROWS = {
    "en/tools-reference.md": 30,
}


def _count_update_cards(rel_path: str) -> int:
    text = (REPO_ROOT / "documents" / rel_path).read_text(encoding="utf-8")
    return len(re.findall(r"^<Update\b", text, flags=re.MULTILINE))


def _jsx_leaks(text: str) -> list[str]:
    text = re.sub(r"`[^`]*`", "", text)
    return [pattern.pattern for pattern in LEAK_PATTERNS if pattern.search(text)]


def main() -> int:
    llms_desc = load_llms_descriptions()
    parsed_urls: set[str] = set()
    errors: list[str] = []
    totals = Counter()

    for path in iter_corpus_files(EN_ROOT):
        rel = path.relative_to(EN_ROOT.parent).as_posix()
        try:
            records = parse_one(rel, llms_desc)
        except Exception as exc:  # pragma: no cover - shown in CLI output.
            errors.append(f"{rel}: parser raised {type(exc).__name__}: {exc}")
            continue

        counts = Counter(record["kind"] for record in records)
        totals.update(counts)

        if counts["page"] != 1:
            errors.append(f"{rel}: expected exactly one Page, got {counts['page']}")
            continue
        if counts["section"] < 1:
            errors.append(f"{rel}: expected at least one Section")

        page = next(record for record in records if record["kind"] == "page")
        parsed_urls.add(page["url"])

        section_ids = [record["id"] for record in records if record["kind"] == "section"]
        duplicate_section_ids = [
            section_id for section_id, count in Counter(section_ids).items() if count > 1
        ]
        if duplicate_section_ids:
            preview = ", ".join(duplicate_section_ids[:3])
            errors.append(f"{rel}: duplicate Section ids: {preview}")

        for section in (record for record in records if record["kind"] == "section"):
            leaks = _jsx_leaks(section.get("text", ""))
            if leaks:
                errors.append(
                    f"{rel}: JSX/MDX leakage in section {section['anchor']}: "
                    + ", ".join(leaks)
                )

        if rel in EXPECTED_UPDATE_COUNTS:
            raw_count = _count_update_cards(rel)
            expected = EXPECTED_UPDATE_COUNTS[rel]
            synthesized = sum(
                1
                for record in records
                if record["kind"] == "section" and record.get("synthesized")
            )
            if raw_count != expected:
                errors.append(
                    f"{rel}: CLAUDE.md expects {expected} <Update> cards, "
                    f"but source has {raw_count}"
                )
            if synthesized != raw_count:
                errors.append(
                    f"{rel}: expected {raw_count} synthesized update Sections, "
                    f"got {synthesized}"
                )

        if rel in EXPECTED_MIN_TABLE_ROWS:
            minimum = EXPECTED_MIN_TABLE_ROWS[rel]
            if counts["tablerow"] < minimum:
                errors.append(
                    f"{rel}: expected at least {minimum} data TableRows, "
                    f"got {counts['tablerow']}"
                )

    missing_urls = sorted(set(llms_desc) - parsed_urls)
    for url in missing_urls:
        errors.append(f"llms.txt URL was not parsed: {url}")

    if errors:
        print("check_parse.py FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "check_parse.py OK: "
        f"{totals['page']} pages, {totals['section']} sections, "
        f"{totals['codeblock']} code blocks, {totals['tablerow']} table rows, "
        f"{totals['callout']} callouts; {len(llms_desc)} llms.txt URLs resolved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
