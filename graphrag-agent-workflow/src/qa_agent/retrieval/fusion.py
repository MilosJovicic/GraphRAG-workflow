"""Reciprocal Rank Fusion. Pure, deterministic, and safe in a Temporal workflow."""

from __future__ import annotations

from qa_agent.schemas import Candidate


def _rank_by(cands: list[Candidate], score_attr: str) -> list[Candidate]:
    scored = [c for c in cands if getattr(c, score_attr) is not None]
    return sorted(scored, key=lambda c: (-getattr(c, score_attr), c.node_id))


def rrf_fuse(
    per_subquery_results: list[list[Candidate]],
    k: int = 60,
    top_n: int = 40,
) -> list[Candidate]:
    """Fuse candidates across sub-query BM25/vector ranked lists."""
    by_key: dict[tuple[str, str], Candidate] = {}
    scores: dict[tuple[str, str], float] = {}

    for sq_results in per_subquery_results:
        for ranked in (_rank_by(sq_results, "bm25_score"), _rank_by(sq_results, "vector_score")):
            for rank, cand in enumerate(ranked, start=1):
                key = (cand.node_id, cand.node_label)
                by_key.setdefault(key, cand)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(scores.keys(), key=lambda key: (-scores[key], key[0]))
    return [
        by_key[key].model_copy(update={"rrf_score": scores[key]})
        for key in sorted_keys[:top_n]
    ]
