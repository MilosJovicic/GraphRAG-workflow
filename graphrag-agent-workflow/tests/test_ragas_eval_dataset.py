from __future__ import annotations

from pathlib import Path

import yaml


GOLD_PATH = Path(__file__).resolve().parents[1] / "ragas_evals" / "gold_questions.yaml"


def test_gold_questions_file_exists_and_has_seven_curated_samples():
    assert GOLD_PATH.exists()

    samples = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))

    assert isinstance(samples, list)
    assert len(samples) == 7


def _flatten_expected_ids(entries: list) -> list[str]:
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, list):
            for member in entry:
                assert isinstance(member, str), "or-group members must be strings"
                out.append(member)
        else:
            raise AssertionError(f"unexpected gold id type: {type(entry).__name__}")
    return out


def test_gold_questions_have_required_fields_and_no_placeholders():
    samples = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    seen_ids: set[str] = set()

    for sample in samples:
        assert sample["id"] not in seen_ids
        seen_ids.add(sample["id"])
        assert sample["category"].strip()
        assert len(sample["question"].strip()) >= 10
        assert len(sample["reference_answer"].strip()) >= 40
        assert sample["expected_context_ids"]

        flat_ids = _flatten_expected_ids(sample["expected_context_ids"])
        joined = "\n".join(
            [
                sample["id"],
                sample["category"],
                sample["question"],
                sample["reference_answer"],
                *flat_ids,
            ]
        ).lower()
        assert "placeholder" not in joined
        assert "<" not in joined
        assert ">" not in joined
