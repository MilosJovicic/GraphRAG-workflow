# RAGAS Evaluation Plan

## Purpose

This folder is the source of truth for evaluating the Claude Code documentation
GraphRAG Q&A agent. The initial gold set has seven hand-curated questions that
cover the dataset's most important retrieval shapes:

- permissions and permission modes
- hook lifecycle behavior
- tool and code example lookup
- session/model configuration
- provider setup

The eval checks both retrieval quality and answer quality. Deterministic ID
metrics are the first gate because this dataset has stable Neo4j node IDs for
expected contexts. RAGAS scores are used to inspect answer grounding, relevance,
and agreement with reference answers.

The default RAGAS judge model is OpenAI `gpt-4.1`. The smaller `gpt-4.1-mini`
under-scores `factual_correctness` and `answer_relevancy` on paraphrased
answers (observed 0.3x averages on grounded responses), so the default is the
full model. Override with `--judge-model gpt-4.1-mini` for cheaper smoke runs
where claim-decomposition fidelity is not critical. Semantic similarity uses
`text-embedding-3-large`.

## Required Services

Run these before launching the eval:

- Temporal server on `TEMPORAL_HOST`
- Q&A worker on `QA_TASK_QUEUE`
- Neo4j with the Claude Code docs graph
- Ollama embeddings service for Q&A retrieval
- OpenAI API key in `.env` or the environment as `OPENAI_API_KEY`
- Optional Cohere key for production reranking; without it the Q&A workflow may
  use its RRF fallback

## Metrics

RAGAS metrics:

- `faithfulness`: whether answer claims are supported by retrieved contexts
- `context_precision`: whether useful contexts are ranked above less useful ones
- `context_recall`: whether retrieved contexts cover the reference answer
- `answer_relevancy`: whether the answer addresses the question directly
- `factual_correctness`: claim overlap between response and reference answer
- `semantic_similarity`: embedding similarity between response and reference

Deterministic metrics:

- `id_hit_at_8`: at least one expected context ID appears in retrieved top 8
- `id_recall_at_8`: fraction of expected context IDs found in retrieved top 8
- `cited_expected_hit`: at least one cited node ID is expected
- `citation_count`: number of returned citations
- `no_results`: whether the Q&A workflow returned its empty-result path
- `fallback_used`: whether the workflow used a retrieval or rerank fallback
- `latency_*_ms`: workflow latency fields copied from `QAResponse.latency_ms`

## Retrieval patterns relevant to this eval

- `code_examples` (added 2026-05-07): graph-traversal expansion that surfaces
  CodeBlocks for entity seeds via `Entity ← DEFINES|MENTIONS ← Section →
  HAS_SUBSECTION*0.. → Section → CONTAINS_CODE → CodeBlock`. The planner
  emits this when the question asks for an example. Q3 (Edit Python example)
  depends on this pattern; without it, retrieval falls back to BM25/vector
  over CodeBlocks and the gold quickstart codeblocks do not surface.

## Thresholds

The default hard gate is deterministic:

- `id_hit_at_8` average must be at least `1.0`
- every row should have `no_results = 0`

RAGAS scores are report-first while the gold answers stabilize. Treat these as
initial review targets, not strict CI gates:

- `faithfulness` target: `>= 0.85`
- `context_precision` target: `>= 0.70`
- `context_recall` target: `>= 0.70`
- `answer_relevancy` target: `>= 0.80`
- `factual_correctness` target: `>= 0.70`
- `semantic_similarity` target: `>= 0.75`

## Run Commands

Install the eval dependencies:

```powershell
pip install -e ".[dev,eval]"
```

Run unit checks for the eval harness:

```powershell
pytest tests/test_ragas_eval_dataset.py tests/test_ragas_eval_helpers.py tests/test_ragas_eval_imports.py -q
```

Run the live eval:

```powershell
$env:QA_RUN_INTEGRATION = "1"
python ragas_evals/run_ragas_eval.py --enforce-id-thresholds
```

Use `--limit 2` for a small smoke run, `--skip-ragas` for deterministic
retrieval-only debugging, or `--metric-timeout-seconds 120` when you want to
give slower RAGAS judge calls more room.

Default OpenAI models:

```powershell
python ragas_evals/run_ragas_eval.py --judge-model gpt-4.1 --embedding-model text-embedding-3-large --enforce-id-thresholds
```

## Outputs

Each run writes timestamped files to `ragas_evals/results/`:

- `ragas_eval_<timestamp>.jsonl`: raw sample, response, contexts, and scores
- `ragas_eval_<timestamp>.csv`: row-level tabular scores
- `ragas_eval_<timestamp>.md`: summary averages, category breakdowns, and ID
  failures
