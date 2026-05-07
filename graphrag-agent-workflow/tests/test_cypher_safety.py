"""Ensure Cypher stays in parameterized templates, not generated Python strings."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qa_agent"

FORBIDDEN = [
    re.compile(r'f"\s*MATCH\b', re.IGNORECASE),
    re.compile(r'f"\s*MERGE\b', re.IGNORECASE),
    re.compile(r'f"\s*CREATE\b', re.IGNORECASE),
    re.compile(r'f"\s*CALL\s+db\.', re.IGNORECASE),
    re.compile(r'f"\s*RETURN\b', re.IGNORECASE),
    re.compile(r'f"\s*WHERE\b', re.IGNORECASE),
    re.compile(r'f"\s*UNWIND\b', re.IGNORECASE),
    re.compile(r'f"\s*WITH\b', re.IGNORECASE),
    re.compile(r'\+\s*"MATCH\b', re.IGNORECASE),
    re.compile(r'\+\s*"MERGE\b', re.IGNORECASE),
    re.compile(r'\+\s*"CALL\s+db\.', re.IGNORECASE),
]


def test_no_f_string_cypher_in_src():
    offenders: list[tuple[Path, int, str]] = []
    for py in SRC.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            for pat in FORBIDDEN:
                if pat.search(line):
                    offenders.append((py.relative_to(SRC.parent.parent), i, line.strip()))
    assert offenders == [], "Found f-string Cypher in source:\n" + "\n".join(
        f"  {p}:{i}  {line}" for p, i, line in offenders
    )
