from __future__ import annotations

from qa_agent.schemas import Candidate, Citation, Plan, QAResponse, SubQuery
from ragas_evals.run_ragas_eval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    build_retrieved_contexts,
    compute_id_metrics,
    ensure_openai_api_key,
)


def _response() -> QAResponse:
    plan = Plan(sub_queries=[SubQuery(text="How do hooks work?", target_labels=["Section"])])
    return QAResponse(
        answer="Hooks run around tool use [1].",
        citations=[
            Citation(
                id=1,
                node_id="Hook:PostToolUse",
                node_label="Hook",
                snippet="PostToolUse runs after a tool call.",
                score=0.91,
            )
        ],
        plan=plan,
        retrieved=[
            Candidate(
                node_id="Hook:PostToolUse",
                node_label="Hook",
                indexed_text="PostToolUse runs after a tool call.",
                raw_text="PostToolUse raw text",
                title="PostToolUse",
                rerank_score=0.91,
            ),
            Candidate(
                node_id="noise",
                node_label="Section",
                indexed_text="Unrelated chunk.",
                raw_text="Unrelated raw text.",
                title="Noise",
                rrf_score=0.2,
            ),
        ],
        latency_ms={"total": 1234, "rerank": 42},
    )


def test_build_retrieved_contexts_prefers_indexed_text_with_metadata_prefix():
    contexts = build_retrieved_contexts(_response())

    assert contexts == [
        "[Hook] PostToolUse\nPostToolUse runs after a tool call.",
        "[Section] Noise\nUnrelated chunk.",
    ]


def test_compute_id_metrics_reports_hit_recall_and_latency():
    metrics = compute_id_metrics(
        _response(),
        expected_context_ids=["Hook:PostToolUse", "Hook:PreToolUse"],
    )

    assert metrics["id_hit_at_8"] == 1.0
    assert metrics["id_recall_at_8"] == 0.5
    assert metrics["cited_expected_hit"] == 1.0
    assert metrics["citation_count"] == 1
    assert metrics["no_results"] == 0.0
    assert metrics["fallback_used"] == 0.0
    assert metrics["latency_total_ms"] == 1234
    assert metrics["latency_rerank_ms"] == 42


def test_compute_id_metrics_or_groups_count_each_group_once():
    metrics = compute_id_metrics(
        _response(),
        expected_context_ids=[
            ["Hook:PostToolUse", "Hook:Synonym"],
            "Hook:PreToolUse",
        ],
    )

    assert metrics["id_hit_at_8"] == 1.0
    # 1 of 2 groups satisfied (the one containing PostToolUse)
    assert metrics["id_recall_at_8"] == 0.5
    assert metrics["cited_expected_hit"] == 1.0


def test_compute_id_metrics_or_groups_satisfied_by_any_alternate():
    metrics = compute_id_metrics(
        _response(),
        expected_context_ids=[
            ["Hook:DoesNotExist", "Hook:PostToolUse"],
        ],
    )

    assert metrics["id_hit_at_8"] == 1.0
    assert metrics["id_recall_at_8"] == 1.0
    assert metrics["cited_expected_hit"] == 1.0


def test_compute_id_metrics_or_groups_unsatisfied_when_no_alternate_matches():
    metrics = compute_id_metrics(
        _response(),
        expected_context_ids=[
            ["Hook:Missing1", "Hook:Missing2"],
            "Hook:PostToolUse",
        ],
    )

    assert metrics["id_hit_at_8"] == 1.0
    assert metrics["id_recall_at_8"] == 0.5
    assert metrics["cited_expected_hit"] == 1.0


def test_gold_question_from_mapping_normalizes_or_groups():
    from ragas_evals.run_ragas_eval import GoldQuestion

    sample = GoldQuestion.from_mapping(
        {
            "id": "qx",
            "category": "permissions",
            "question": "anything sufficiently long",
            "reference_answer": "x" * 60,
            "expected_context_ids": [
                "plain-string",
                ["alt-a", "alt-b"],
            ],
        }
    )

    assert sample.expected_context_ids == [["plain-string"], ["alt-a", "alt-b"]]


def test_compute_id_metrics_handles_empty_retrieval_without_division_error():
    plan = Plan(sub_queries=[SubQuery(text="Nothing?", target_labels=["Section"])])
    response = QAResponse.empty(plan)

    metrics = compute_id_metrics(response, expected_context_ids=["expected"])

    assert metrics["id_hit_at_8"] == 0.0
    assert metrics["id_recall_at_8"] == 0.0
    assert metrics["cited_expected_hit"] == 0.0
    assert metrics["no_results"] == 1.0


def test_openai_judge_defaults_use_small_compatible_openai_models():
    assert DEFAULT_JUDGE_MODEL == "gpt-4.1"
    assert DEFAULT_EMBEDDING_MODEL == "text-embedding-3-large"


def test_ensure_openai_api_key_reads_openai_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OTHER_LLM_API_KEY", "other-test-key")

    try:
        ensure_openai_api_key()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OPENAI_API_KEY should fail")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    assert ensure_openai_api_key() == "sk-test-openai"
