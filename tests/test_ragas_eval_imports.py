from __future__ import annotations


def test_ragas_eval_import_surface_is_available():
    import pandas  # noqa: F401
    import ragas  # noqa: F401
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
        SemanticSimilarity,
    )

    assert callable(llm_factory)
    assert OpenAIEmbeddings is not None
    assert Faithfulness is not None
    assert ContextPrecision is not None
    assert ContextRecall is not None
    assert AnswerRelevancy is not None
    assert FactualCorrectness is not None
    assert SemanticSimilarity is not None
