import pytest
from pydantic import ValidationError

from qa_agent.schemas import (
    NODE_LABELS,
    Candidate,
    Citation,
    ExpansionPattern,
    Plan,
    QARequest,
    SubQuery,
)


def test_subquery_requires_target_labels():
    with pytest.raises(ValidationError):
        SubQuery(text="hello", target_labels=[])


def test_subquery_accepts_known_label():
    sq = SubQuery(text="hello", target_labels=["Section"])
    assert sq.target_labels == ["Section"]


def test_plan_min_one_subquery():
    with pytest.raises(ValidationError):
        Plan(sub_queries=[])


def test_plan_max_four_subqueries():
    sqs = [SubQuery(text=f"q{i}", target_labels=["Section"]) for i in range(5)]
    with pytest.raises(ValidationError):
        Plan(sub_queries=sqs)


def test_expansion_pattern_max_per_seed_bounds():
    with pytest.raises(ValidationError):
        ExpansionPattern(name="siblings", max_per_seed=10)
    ExpansionPattern(name="siblings", max_per_seed=5)


def test_expansion_pattern_unknown_name_rejected():
    with pytest.raises(ValidationError):
        ExpansionPattern(name="bogus")  # type: ignore[arg-type]


def test_expansion_pattern_accepts_code_examples_name():
    pattern = ExpansionPattern(name="code_examples", max_per_seed=4)
    assert pattern.name == "code_examples"


def test_expansion_pattern_max_per_seed_accepts_8():
    pattern = ExpansionPattern(name="code_examples", max_per_seed=8)
    assert pattern.max_per_seed == 8


def test_expansion_pattern_max_per_seed_rejects_9():
    with pytest.raises(ValidationError):
        ExpansionPattern(name="code_examples", max_per_seed=9)


def test_expansion_pattern_language_default_is_none():
    pattern = ExpansionPattern(name="code_examples")
    assert pattern.language is None


def test_expansion_pattern_language_round_trip():
    pattern = ExpansionPattern(
        name="code_examples", max_per_seed=6, language="python"
    )
    assert pattern.language == "python"
    dumped = pattern.model_dump()
    rehydrated = ExpansionPattern.model_validate(dumped)
    assert rehydrated.language == "python"


def test_qa_request_min_length():
    with pytest.raises(ValidationError):
        QARequest(question="")


def test_node_labels_constant_complete():
    expected = {
        "Section",
        "Chunk",
        "CodeBlock",
        "TableRow",
        "Callout",
        "Tool",
        "Hook",
        "SettingKey",
        "PermissionMode",
        "MessageType",
        "Provider",
    }
    assert set(NODE_LABELS) == expected


def test_candidate_default_scores_none():
    c = Candidate(node_id="n1", node_label="Section", indexed_text="x", raw_text="y")
    assert c.bm25_score is None
    assert c.vector_score is None
    assert c.rrf_score is None


def test_candidate_normalizes_missing_text_fields():
    c = Candidate(node_id="n1", node_label="Section", indexed_text=None, raw_text=None)

    assert c.indexed_text == ""
    assert c.raw_text == ""


def test_citation_accepts_page_label_from_expansion():
    citation = Citation(
        id=1,
        node_id="page-1",
        node_label="Page",
        snippet="page evidence",
        score=0.5,
    )

    assert citation.node_label == "Page"
