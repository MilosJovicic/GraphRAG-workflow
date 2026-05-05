from qa_agent.activities.retrieve import _merge_legs
from qa_agent.schemas import Candidate


def _candidate(
    node_id: str,
    *,
    bm25: float | None = None,
    vector: float | None = None,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_label="Section",
        indexed_text="x",
        raw_text="y",
        bm25_score=bm25,
        vector_score=vector,
    )


def test_merge_legs_dedupes_and_combines_scores():
    bm25 = [_candidate("a", bm25=10.0), _candidate("b", bm25=5.0)]
    vector = [_candidate("a", vector=0.9), _candidate("c", vector=0.5)]

    out = _merge_legs(bm25, vector)

    by_id = {candidate.node_id: candidate for candidate in out}
    assert by_id["a"].bm25_score == 10.0
    assert by_id["a"].vector_score == 0.9
    assert by_id["b"].bm25_score == 5.0
    assert by_id["b"].vector_score is None
    assert by_id["c"].bm25_score is None
    assert by_id["c"].vector_score == 0.5
    assert len(out) == 3
