from pathlib import Path


SCRIPT = Path(__file__).parent.parent / "scripts" / "verify_v2_pilot.py"


def test_v2_verifier_contains_all_gate_queries():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "leak_pct < 1%" in text
    assert "this_pct < 5%" in text
    assert "starts_with_name = total" in text
    assert "Target: < 0.70" in text
    assert "mean stays in [0.75, 0.85]" in text
    assert "zero rows returned" in text


def test_v2_verifier_searches_tablerow_cells_not_null_text():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "any(cell IN coalesce(t.cells, [])" in text
    assert "any(header IN coalesce(t.headers, [])" in text
    assert "t.context_version = 2" in text
    assert "t.context_source = 'llm'" in text
    assert "t.text CONTAINS" not in text


def test_v2_verifier_reports_pilot_size_context():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Expected corrected pilot" in text
    assert "48-node pilot" in text
