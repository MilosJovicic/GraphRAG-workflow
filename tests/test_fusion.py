import pytest

from qa_agent.retrieval.fusion import rrf_fuse
from qa_agent.schemas import Candidate


def _c(nid: str, bm25: float | None = None, vec: float | None = None) -> Candidate:
    return Candidate(
        node_id=nid,
        node_label="Section",
        indexed_text=f"text-{nid}",
        raw_text=f"raw-{nid}",
        bm25_score=bm25,
        vector_score=vec,
    )


def test_rrf_empty_input_returns_empty():
    assert rrf_fuse([], k=60, top_n=10) == []


def test_rrf_single_subquery_single_leg():
    cs = [_c("a", bm25=10.0), _c("b", bm25=8.0), _c("c", bm25=5.0)]
    out = rrf_fuse([cs], k=60, top_n=10)
    assert [x.node_id for x in out] == ["a", "b", "c"]
    assert all(x.rrf_score is not None for x in out)
    assert out[0].rrf_score == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_single_subquery_both_legs_dedup():
    cs = [_c("a", bm25=10.0, vec=0.5), _c("b", bm25=8.0), _c("c", vec=0.9)]
    out = rrf_fuse([cs], k=60, top_n=10)
    by_id = {x.node_id: x for x in out}
    assert by_id["a"].rrf_score == pytest.approx(1 / 61 + 1 / 62, rel=1e-6)
    assert by_id["c"].rrf_score == pytest.approx(1 / 61, rel=1e-6)
    assert by_id["b"].rrf_score == pytest.approx(1 / 62, rel=1e-6)
    assert [x.node_id for x in out] == ["a", "c", "b"]


def test_rrf_top_n_truncates():
    cs = [_c(f"n{i}", bm25=100 - i) for i in range(10)]
    out = rrf_fuse([cs], k=60, top_n=3)
    assert len(out) == 3
    assert [x.node_id for x in out] == ["n0", "n1", "n2"]


def test_rrf_deterministic_across_runs():
    cs = [_c("z", bm25=5.0), _c("a", bm25=5.0), _c("m", bm25=5.0)]
    runs = [tuple(x.node_id for x in rrf_fuse([cs], k=60, top_n=10)) for _ in range(50)]
    assert len(set(runs)) == 1
    assert runs[0] == ("a", "m", "z")


def test_rrf_multiple_subqueries_accumulate():
    sq1 = [_c("a", bm25=10.0), _c("b", bm25=5.0)]
    sq2 = [_c("a", vec=0.9), _c("b", vec=0.5)]
    out = rrf_fuse([sq1, sq2], k=60, top_n=10)
    by_id = {x.node_id: x for x in out}
    assert by_id["a"].rrf_score == pytest.approx(2 / 61, rel=1e-6)
    assert by_id["b"].rrf_score == pytest.approx(2 / 62, rel=1e-6)
    assert [x.node_id for x in out] == ["a", "b"]


def test_rrf_preserves_payload():
    cs = [_c("a", bm25=10.0)]
    out = rrf_fuse([cs], k=60, top_n=10)
    assert out[0].indexed_text == "text-a"
    assert out[0].raw_text == "raw-a"
    assert out[0].bm25_score == 10.0
