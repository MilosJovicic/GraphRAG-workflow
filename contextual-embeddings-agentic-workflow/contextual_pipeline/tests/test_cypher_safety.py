"""Asserts no inline Cypher is constructed via f-strings or string concatenation in src/."""
import re
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"

DANGEROUS_PATTERNS = [
    r'f"MATCH',
    r"f'MATCH",
    r'f"CREATE',
    r"f'CREATE",
    r'f"MERGE',
    r"f'MERGE",
    r'f"SET',
    r"f'SET",
    r'" MATCH',
    r"' MATCH",
    r'" CREATE',
    r"' CREATE",
]


def _find_pattern(pattern: str) -> list[str]:
    hits = []
    for py_file in SRC_DIR.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(re.escape(pattern), line):
                hits.append(f"{py_file.relative_to(SRC_DIR)}:{i}: {line.strip()}")
    return hits


def test_no_fstring_cypher_match():
    assert _find_pattern('f"MATCH') == [], "Found f-string MATCH"
    assert _find_pattern("f'MATCH") == [], "Found f-string MATCH"


def test_no_fstring_cypher_create():
    assert _find_pattern('f"CREATE') == [], "Found f-string CREATE"
    assert _find_pattern("f'CREATE") == [], "Found f-string CREATE"


def test_no_fstring_cypher_merge():
    assert _find_pattern('f"MERGE') == [], "Found f-string MERGE"
    assert _find_pattern("f'MERGE") == [], "Found f-string MERGE"


def test_no_fstring_cypher_set():
    assert _find_pattern('f"SET') == [], "Found f-string SET"
    assert _find_pattern("f'SET") == [], "Found f-string SET"


def test_no_concat_cypher_match():
    hits = _find_pattern('" MATCH') + _find_pattern("' MATCH")
    assert hits == [], f"Found string-concat MATCH:\n" + "\n".join(hits)


def test_no_concat_cypher_create():
    hits = _find_pattern('" CREATE') + _find_pattern("' CREATE")
    assert hits == [], f"Found string-concat CREATE:\n" + "\n".join(hits)
