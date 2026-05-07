from pydantic_ai.models.openai import OpenAIChatModel

from qa_agent.agents.answerer import extract_citation_ids, validate_citations
from qa_agent.agents.answerer import build_answerer_agent


def test_extract_citation_ids_simple():
    assert extract_citation_ids("Use foo [1] and bar [2][3].") == [1, 2, 3]


def test_extract_citation_ids_dedup_preserves_first_seen_order():
    assert extract_citation_ids("[2] x [1] y [2]") == [2, 1]


def test_extract_citation_ids_empty():
    assert extract_citation_ids("no citations here.") == []


def test_validate_citations_passes_when_in_range():
    ok, missing, out_of_range = validate_citations(
        "Foo [1][2].",
        evidence_count=3,
        used_ids=[1, 2],
    )

    assert ok is True
    assert missing == []
    assert out_of_range == []


def test_validate_citations_flags_out_of_range():
    ok, missing, out_of_range = validate_citations(
        "Foo [9].",
        evidence_count=3,
        used_ids=[9],
    )

    assert ok is False
    assert out_of_range == [9]


def test_validate_citations_flags_used_but_not_in_text():
    ok, missing, out_of_range = validate_citations(
        "Foo [1].",
        evidence_count=3,
        used_ids=[1, 2],
    )

    assert ok is False
    assert missing == [2]
    assert out_of_range == []


def test_answerer_default_model_uses_openai_small_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("NEO4J_URI", "x")
    monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")

    from qa_agent.config import get_settings

    get_settings.cache_clear()
    agent = build_answerer_agent()

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "gpt-5.4-mini"
