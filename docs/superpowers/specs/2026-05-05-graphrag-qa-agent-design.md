# GraphRAG Q&A Agent â€” Design Spec

**Date:** 2026-05-05
**Status:** Ready for step-by-step implementation after doc consistency fixes
**Owner:** milos.jovicic@gmail.com

## 1. Goal

Build a question-answering agent over the existing Claude Code documentation knowledge graph in Neo4j. Retrieval combines dense (Qwen 3 0.6B embeddings) and sparse (BM25 fulltext) search via Reciprocal Rank Fusion, augments seed candidates with graph expansion over the existing relationship structure, reranks with Cohere, and generates a cited answer using OpenAI gpt-5.4-mini via PydanticAI. Orchestration is Temporal; the entry point is a Flask `POST /ask` endpoint that starts a workflow per question and returns when the workflow completes.

## 2. Non-goals (v1)

- Streaming responses (token-by-token).
- Multi-turn / conversational state.
- Targeted per-pattern Cypher tools (e.g. `find_tool_definition(name)`); the planner picks from a fixed expansion menu only.
- Authentication, rate limiting, deployment scripts (Docker/k8s).
- RAGAS / DeepEval scoring; v1 ships with a small fixture eval.
- Modifications to the upstream graph schema or contextual enrichment outputs.

## 3. Stack

- **Orchestration:** Temporal (Python SDK, `temporalio==1.20.0`). Workflow per question.
- **Agent framework:** PydanticAI (>=1.61.0) for both the planner and the answerer agents. Pydantic output schemas, no freeform parsing.
- **LLM (planner + answerer):** OpenAI gpt-5.4-mini via the OpenAI API. Read `OPENAI_API_KEY` from `.env`.
- **Embeddings:** Qwen 3 0.6B served by Ollama, OpenAI-compatible endpoint at `http://localhost:11434/v1`. Dimension 1024 (must match the existing `*_embedding` vector indexes).
- **Reranker:** Cohere `rerank-v3.5` (multilingual). Read optional `COHERE_API_KEY` from `.env`; when absent, use the RRF fallback path.
- **Graph DB:** existing Neo4j at `neo4j://127.0.0.1:7687`. No schema changes.
- **HTTP layer:** Flask 3.0.
- **Observability:** Logfire (already pinned).

## 4. Architecture

Two long-lived processes share `.env`:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  POST /ask     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  start-and-wait  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Caller      â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶â”‚  Flask app      â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¶â”‚ Temporal serverâ”‚
â”‚  (curl/UI)   â”‚â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤  (api.py)       â”‚â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤ (localhost:7233)â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  AnswerEnvelopeâ””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  WorkflowResult  â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                                                                              â”‚
                                                                              â–¼
                                                                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                                                  â”‚ Worker process      â”‚
                                                                  â”‚  qa_agent.worker    â”‚
                                                                  â”‚   â€¢ workflows/      â”‚
                                                                  â”‚   â€¢ activities/     â”‚
                                                                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                                             â”‚
                                                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                                â–¼                            â–¼                     â–¼
                                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                           â”‚ Neo4j   â”‚               â”‚ Ollama Qwen  â”‚      â”‚ Cohere rerank  â”‚
                                           â”‚ 7687    â”‚               â”‚ 11434/v1     â”‚      â”‚ + OpenAI gpt-5.4-mini â”‚
                                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

The worker registers one workflow class (`QAWorkflow`) and the activities listed in Â§6, listening on task queue `qa-agent-tq` (separate from any upstream pipelines).

### 4.1 Workflow body

```python
@workflow.defn
class QAWorkflow:
    @workflow.run
    async def run(self, req: QARequest) -> QAResponse:
        # 1. Plan â€” non-retryable planner failure falls back to a synthesized
        #    single-subquery plan and uses the naive fallback path below.
        plan: Plan
        planner_failed = False
        try:
            plan = await workflow.execute_activity(
                plan_query, req.question,
                start_to_close_timeout=timedelta(seconds=20))
        except ApplicationError as e:
            if not e.non_retryable:
                raise
            planner_failed = True
            plan = Plan.fallback_for(req.question)   # synthetic plan, used for response shaping only

        if planner_failed:
            top: list[Candidate] = await workflow.execute_activity(
                naive_hybrid_fallback, req.question,
                start_to_close_timeout=timedelta(seconds=15))
            if not top:
                return QAResponse.empty(plan, fallback_used=True)
            return await workflow.execute_activity(
                generate_answer,
                GenerateRequest(question=req.question, evidence=top, plan=plan),
                start_to_close_timeout=timedelta(seconds=45))

        # 2. Hybrid retrieval, parallel per sub-query
        per_subquery: list[list[Candidate]] = await asyncio.gather(*[
            workflow.execute_activity(hybrid_search, sub_q,
                start_to_close_timeout=timedelta(seconds=15))
            for sub_q in plan.sub_queries
        ])

        # 3. RRF fusion (pure, deterministic)
        fused: list[Candidate] = rrf_fuse(per_subquery, k=60, top_n=40)

        # 4. Graph expansion
        expanded: list[Candidate] = await workflow.execute_activity(
            expand_graph,
            ExpandRequest(seeds=fused, patterns=plan.expansion_patterns),
            start_to_close_timeout=timedelta(seconds=10))

        # 5. Cohere rerank â†’ top 8
        top = await workflow.execute_activity(
            rerank,
            RerankRequest(question=req.question, candidates=expanded, top_k=8),
            start_to_close_timeout=timedelta(seconds=15))

        # 6. Fallback if empty
        if not top:
            top = await workflow.execute_activity(
                naive_hybrid_fallback, req.question,
                start_to_close_timeout=timedelta(seconds=15))

        # 7. Generate (or graceful empty)
        if not top:
            return QAResponse.empty(plan)
        return await workflow.execute_activity(
            generate_answer,
            GenerateRequest(question=req.question, evidence=top, plan=plan),
            start_to_close_timeout=timedelta(seconds=45))
```

`Plan.fallback_for(question)` is a classmethod returning a synthetic plan: a single `SubQuery(text=question, target_labels=["Section","CodeBlock","TableRow","Callout"])` with no expansion, used purely so the response can include a populated `plan` field for debug. The actual retrieval in the planner-failed branch is the naive fallback path (Â§8.5).

### 4.2 Invariants

- The workflow is **deterministic**: no clocks, no `random`, no I/O outside activity calls. RRF fusion and response shaping are pure.
- Every activity has a `start_to_close_timeout` and a default retry policy (3 attempts, exponential backoff with `initial_interval=1s`, `backoff_coefficient=2.0`, `maximum_interval=30s`). Validation failures raise `ApplicationError(non_retryable=True)`.
- The workflow is **idempotent** by construction: zero DB writes anywhere in the read path. Re-running a workflow on the same question is safe.
- Latency budget: ~3â€“6s for typical questions (plan â‰ˆ 500ms, retrieval â‰ˆ 500ms, expansion â‰ˆ 200ms, rerank â‰ˆ 500ms, generation â‰ˆ 1â€“3s). Per-stage timings logged via Logfire and returned in `QAResponse.latency_ms`.

## 5. Project layout

```
Q&A Agent/
â”œâ”€â”€ .env                          # NEO4J_*, TEMPORAL_*, OPENAI_API_KEY,
â”‚                                 # OLLAMA_BASE_URL, EMBEDDING_MODEL,
â”‚                                 # COHERE_API_KEY, COHERE_RERANK_MODEL,
â”‚                                 # OPENAI_MODEL, QA_TASK_QUEUE
â”œâ”€â”€ .env.example
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ README.md
â”œâ”€â”€ src/qa_agent/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ config.py                 # pydantic-settings BaseSettings
â”‚   â”œâ”€â”€ schemas.py                # wire types (see Â§6)
â”‚   â”œâ”€â”€ neo4j_client.py           # extends existing file
â”‚   â”œâ”€â”€ cypher/
â”‚   â”‚   â”œâ”€â”€ vector_section.cypher
â”‚   â”‚   â”œâ”€â”€ vector_chunk.cypher
â”‚   â”‚   â”œâ”€â”€ vector_codeblock.cypher
â”‚   â”‚   â”œâ”€â”€ vector_tablerow.cypher
â”‚   â”‚   â”œâ”€â”€ vector_callout.cypher
â”‚   â”‚   â”œâ”€â”€ fulltext_section.cypher
â”‚   â”‚   â”œâ”€â”€ fulltext_chunk.cypher
â”‚   â”‚   â”œâ”€â”€ fulltext_codeblock.cypher
â”‚   â”‚   â”œâ”€â”€ fulltext_tablerow.cypher
â”‚   â”‚   â”œâ”€â”€ fulltext_callout.cypher
â”‚   â”‚   â”œâ”€â”€ expand_siblings.cypher
â”‚   â”‚   â”œâ”€â”€ expand_parent_page.cypher
â”‚   â”‚   â”œâ”€â”€ expand_links.cypher
â”‚   â”‚   â”œâ”€â”€ expand_defines.cypher
â”‚   â”‚   â”œâ”€â”€ expand_navigates_to.cypher
â”‚   â”‚   â””â”€â”€ hydrate_nodes.cypher
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”‚   â”œâ”€â”€ bm25.py
â”‚   â”‚   â”œâ”€â”€ vector.py
â”‚   â”‚   â”œâ”€â”€ fusion.py             # rrf_fuse â€” pure, deterministic
â”‚   â”‚   â”œâ”€â”€ expansion.py
â”‚   â”‚   â””â”€â”€ rerank.py
â”‚   â”œâ”€â”€ embeddings.py             # Ollama OpenAI-compatible client
â”‚   â”œâ”€â”€ agents/
â”‚   â”‚   â”œâ”€â”€ planner.py
â”‚   â”‚   â””â”€â”€ answerer.py
â”‚   â”œâ”€â”€ prompts/
â”‚   â”‚   â”œâ”€â”€ planner.txt
â”‚   â”‚   â””â”€â”€ answerer.txt
â”‚   â”œâ”€â”€ activities/
â”‚   â”‚   â”œâ”€â”€ plan.py
â”‚   â”‚   â”œâ”€â”€ retrieve.py
â”‚   â”‚   â”œâ”€â”€ expand.py
â”‚   â”‚   â”œâ”€â”€ rerank.py
â”‚   â”‚   â””â”€â”€ generate.py
â”‚   â”œâ”€â”€ workflows/
â”‚   â”‚   â””â”€â”€ qa.py
â”‚   â”œâ”€â”€ api.py                    # Flask: POST /ask, GET /health
â”‚   â”œâ”€â”€ worker.py                 # Temporal worker entrypoint
â”‚   â””â”€â”€ starter.py                # CLI: python -m qa_agent.starter "question"
â””â”€â”€ tests/
    â”œâ”€â”€ test_fusion.py
    â”œâ”€â”€ test_planner.py
    â”œâ”€â”€ test_cypher_safety.py
    â”œâ”€â”€ test_workflow_replay.py
    â”œâ”€â”€ test_retrieval_smoke.py
    â””â”€â”€ fixtures/
        â””â”€â”€ eval_questions.yaml
```

The existing `neo4j_client.py` is preserved and extended with a `run_cypher(filename, params)` helper that combines `load_cypher` with `run_query_async`. A `config.py` is added so the existing `from .config import get_settings` import becomes valid.

## 6. Wire types (`schemas.py`)

```python
from typing import Literal
from pydantic import BaseModel, Field

# ---- inputs / outputs ----
class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    debug: bool = False

class Citation(BaseModel):
    id: int                          # 1-indexed, matches inline [n] markers
    node_id: str
    node_label: str                  # may include Page from graph expansion
    url: str | None = None
    anchor: str | None = None
    title: str | None = None         # breadcrumb tail or entity name
    snippet: str                     # truncated indexed_text for display
    score: float                     # final rerank score (or rrf fallback)

class QAResponse(BaseModel):
    answer: str                                 # contains inline [1][2] markers
    citations: list[Citation]
    plan: "Plan"
    retrieved: list["Candidate"] | None = None  # only when debug=True
    latency_ms: dict[str, int]                  # plan, retrieve, expand, rerank, generate, total
    fallback_used: bool = False                 # true if naive fallback fired
    no_results: bool = False                    # true on graceful empty response

# ---- planner output ----
NODE_LABELS = ["Section","Chunk","CodeBlock","TableRow","Callout",
               "Tool","Hook","SettingKey","PermissionMode",
               "MessageType","Provider"]

class SubQuery(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    target_labels: list[str] = Field(min_length=1)
    filters: dict[str, str] = {}                # allow-listed keys only
    bm25_keywords: list[str] = []

class ExpansionPattern(BaseModel):
    name: Literal["siblings","parent_page","links","defines","navigates_to"]
    max_per_seed: int = Field(default=3, ge=1, le=5)

class Plan(BaseModel):
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=4)
    expansion_patterns: list[ExpansionPattern] = []
    notes: str = ""

# ---- retrieval primitives ----
class Candidate(BaseModel):
    node_id: str
    node_label: str
    indexed_text: str                            # what was searched/embedded
    raw_text: str                                # for the LLM context
    url: str | None = None
    anchor: str | None = None
    breadcrumb: str | None = None
    title: str | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    expansion_origin: str | None = None          # "<pattern>:<seed_node_id>"

# ---- activity request envelopes ----
class ExpandRequest(BaseModel):
    seeds: list[Candidate]
    patterns: list[ExpansionPattern]

class RerankRequest(BaseModel):
    question: str
    candidates: list[Candidate]
    top_k: int = 8

class GenerateRequest(BaseModel):
    question: str
    evidence: list[Candidate]
    plan: Plan
```

The filter allow-list (`filters` in `SubQuery`) is enforced at the activity boundary â€” only `language` and `page_path_prefix` are accepted in v1; unknown keys are silently dropped (logged as a warning). This prevents the LLM from inventing filter keys that don't exist in the graph.

## 7. Activity contracts

| Activity | Input | Output | Non-retryable on |
|---|---|---|---|
| `plan_query` | `str` (question) | `Plan` | Pydantic validation failure after 3 retries |
| `hybrid_search` | `SubQuery` | `list[Candidate]` (â‰¤100 per sub-query) | Cypher syntax / schema error |
| `expand_graph` | `ExpandRequest` | `list[Candidate]` (seeds âˆª expansions, deduped) | Cypher syntax / schema error |
| `rerank` | `RerankRequest` | `list[Candidate]` (top_k, with `rerank_score`) | Cohere 4xx (auth/quota) |
| `naive_hybrid_fallback` | `str` (question) | `list[Candidate]` (top 8) | same as above |
| `generate_answer` | `GenerateRequest` | `QAResponse` with `plan`; workflow/API stitch final flags and latency | Pydantic validation after 3 retries |

Every activity is a typed, idempotent operation from input to output. Activities may read packaged prompt/Cypher templates, but they do not write files or use mutable session state. All env access goes through `get_settings()` (cached).

## 8. Retrieval mechanics

### 8.1 Per-label hybrid search

For each label in `subquery.target_labels`, the `hybrid_search` activity runs the BM25 leg and the vector leg in parallel via `asyncio.gather`:

**BM25 leg** (`fulltext_<label>.cypher`):
```cypher
CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
WHERE ($filters_match)
RETURN node, score
ORDER BY score DESC LIMIT $limit
```

The query string is `subquery.text` augmented with `bm25_keywords` (joined by spaces). `$filters_match` is composed from the allow-listed filter keys, never via f-string interpolation. `$limit` defaults to 50.

**Vector leg** (`vector_<label>.cypher`):
```cypher
CALL db.index.vector.queryNodes($index_name, $limit, $query_vector) YIELD node, score
WHERE ($filters_match)
RETURN node, score
```

`$query_vector` is a single Ollama embedding call on `subquery.text`. The activity caches the embedding by `text` for the activity's lifetime only â€” no cross-activity caching (would risk Temporal determinism guarantees).

The activity returns `list[Candidate]` deduped by `(node_id, node_label)`, with whichever of `bm25_score` / `vector_score` were populated. Capped at 100 per sub-query (50 BM25 âˆª 50 vector, deduped).

### 8.2 RRF fusion (pure)

Runs **inside the workflow**, not an activity. Fully deterministic.

```python
def rrf_fuse(per_subquery_results: list[list[Candidate]],
             k: int = 60, top_n: int = 40) -> list[Candidate]:
    """
    Reciprocal Rank Fusion across sub-queries Ã— BM25/vector legs.

    For each sub-query result-set, we fuse two lists:
      - candidates ordered by bm25_score desc
      - candidates ordered by vector_score desc

    A candidate ranked #r in any list contributes 1/(k+r) to its rrf_score.
    Candidates without a score in a leg contribute nothing from that leg.

    Output: top_n candidates by rrf_score, with rrf_score populated.
    Tie-break: stable ordering by node_id (deterministic across retries).
    """
```

Numeric example: with k=60, a candidate ranked #1 in BM25 and #3 in vector for the same sub-query gets `1/61 + 1/63 â‰ˆ 0.0322`. A candidate ranked #1 in only one leg gets `1/61 â‰ˆ 0.0164`. The constant k=60 is the canonical Cormack et al. value; we are not tuning it in v1.

### 8.3 Graph expansion

Inputs: `seeds` (top 40 from RRF) and `patterns` (planner's pick).

Fixed pattern menu, each backed by a parameterized `.cypher` file:

| Pattern | Source labels | Direction |
|---|---|---|
| `siblings` | Chunk, CodeBlock, Callout, TableRow | parent Section â†’ other children |
| `parent_page` | any Section descendant | back up to Page |
| `links` | Section | `:LINKS_TO` / `:LINKS_TO_PAGE` |
| `defines` | Tool, Hook, SettingKey, PermissionMode, MessageType, Provider | `:DEFINES` (and inverse `:MENTIONS`) |
| `navigates_to` | Section | `:NAVIGATES_TO` |

Activity behavior:
1. Group seeds by label, run each applicable pattern's Cypher once per group.
2. Hydrate each expanded node into a `Candidate` (with `indexed_text`, `raw_text`, breadcrumb, etc.).
3. Set `expansion_origin = "<pattern>:<seed_node_id>"`.
4. Cap `max_per_seed` per pattern (planner-supplied, server-side ceiling 5).
5. Cap total expansion at **20** items across all patterns (`MAX_EXPANSION_TOTAL`).
6. Return seeds âˆª expansions, deduped by `(node_id, node_label)`, seeds first.

### 8.4 Cohere rerank

Cohere `rerank-v3.5` (multilingual, 4096-token doc limit). We send `indexed_text` (which carries the upstream-generated `context` prefix, the disambiguation signal we want at rerank time), truncated to 1500 chars per doc.

```http
POST https://api.cohere.com/v2/rerank
{
  "model": "rerank-v3.5",
  "query": "<question>",
  "documents": [<indexed_text truncated>...],
  "top_n": 8,
  "return_documents": false
}
```

Response gives indices + relevance scores; we rebuild `list[Candidate]` in rerank-score order with `rerank_score` populated.

Edge cases:
- Empty candidate list â†’ return `[]` immediately, do not call Cohere.
- Missing Cohere key or Cohere 4xx â†’ non-retryable. Workflow catches and uses RRF top-8 as fallback (sets `fallback_used=True`, logs warning).
- Cohere 5xx / network â†’ retryable, 3 attempts.

### 8.5 Naive hybrid fallback

When the main path returns 0 reranked candidates:

```python
@activity.defn
async def naive_hybrid_fallback(question: str) -> list[Candidate]:
    # Single hybrid sweep across [Section, CodeBlock, TableRow, Callout],
    # no filters, no expansion. RRF then rerank to top 8.
```

If the fallback also returns empty, the workflow returns `QAResponse.empty(plan)` with `no_results=True`.

### 8.6 Numeric knobs

| Param | Default | Env var |
|---|---|---|
| BM25 top per (sub-query Ã— label Ã— leg) | 50 | `BM25_TOP_K` |
| Vector top per (sub-query Ã— label Ã— leg) | 50 | `VECTOR_TOP_K` |
| RRF constant `k` | 60 | hard-coded |
| RRF top_n into expansion | 40 | `RRF_TOP_N` |
| Expansion `max_per_seed` per pattern | 3 | planner-provided, ceiling 5 |
| Expansion total cap | 20 | `MAX_EXPANSION_TOTAL` |
| Cohere top_k into generator | 8 | `RERANK_TOP_K` |
| Indexed-text truncation for rerank input | 1500 chars | `RERANK_DOC_CHARS` |

All knobs live in `config.py`; no magic numbers in retrieval code.

## 9. Generation

### 9.1 Planner agent

PydanticAI agent with `Plan` as the output schema, OpenAI gpt-5.4-mini model. Prompt (`prompts/planner.txt`) provides:
- The user's question.
- A short label catalog (one line per node label).
- The five expansion patterns with one-line descriptions.
- Few-shot examples covering common shapes (factual lookup, configuration questions, cross-tool comparisons).

PydanticAI handles structured output enforcement and one internal validation retry. The activity wraps the agent call with three Temporal-level retries; persistent validation failures are non-retryable.

The planner can only target labels that exist in the graph and can only request expansion patterns from the fixed menu â€” both enforced by the pydantic schema. Off-menu values are validation errors.

### 9.2 Answerer agent

Same OpenAI gpt-5.4-mini model, different prompt. Input rendering â€” for each top-8 candidate:

```
[1] {label} {title or breadcrumb tail}
    url: {url}#{anchor}
    {indexed_text truncated to 800 chars}
```

Numbers are 1-indexed and stable through the response.

Prompt rules:
- Answer using **only** the evidence. If the evidence does not support an answer, say so explicitly.
- Cite every factual claim with `[n]` markers matching the evidence numbers.
- One claim may cite multiple sources: `[1][3]`.
- No outside knowledge, no hedging, no preamble.

Output schema:

```python
class AnswerWithCitations(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    used_citation_ids: list[int]
```

Post-hoc validation in `generate_answer`:
1. Regex out all `[(\d+)]` from `answer`.
2. Check every extracted id is in `1..len(evidence)`.
3. Compare against `used_citation_ids` â€” log a warning on mismatch but do not fail the request (PydanticAI sometimes drops one).
4. On failure of (2), retry the agent call once with feedback. Then accept whatever comes back.

`QAResponse.citations` is built from the evidence by intersecting `used_citation_ids` with `extracted_ids` â€” only candidates the model actually cited appear in the response.

### 9.3 Empty-result response

When `no_results=True`:

```python
QAResponse(
    answer="I couldn't find anything relevant in the knowledge base for that question. "
           "Try rephrasing, or ask about a more specific topic.",
    citations=[],
    plan=plan,
    latency_ms={...},
    no_results=True,
)
```

No LLM call.

## 10. Error handling

| Failure | Where | Behavior |
|---|---|---|
| Planner LLM 5xx / timeout | `plan_query` | 3Ã— exp backoff. On final failure: workflow falls back to running `naive_hybrid_fallback` directly with the raw question, skipping the plan path. |
| Planner pydantic validation | inside agent | PydanticAI internal retry once with feedback, then surfaces; activity retries once more, then non-retryable. |
| Ollama embeddings 5xx / connection refused | `hybrid_search` | 3Ã— exp backoff. On final failure the sub-query's vector leg returns empty; BM25 leg alone produces results. RRF still functions. |
| Neo4j connection error | any Cypher activity | 3Ã— exp backoff. Schema/syntax errors are non-retryable â†’ 500. |
| Missing Cohere key / Cohere 4xx | `rerank` | Non-retryable. Workflow catches and uses RRF top-8 (sets `fallback_used=True`). |
| Cohere 5xx / network | `rerank` | 3Ã— exp backoff. |
| OpenAI 5xx / timeout | `generate_answer` | 3Ã— exp backoff. On final failure: workflow returns 500. |
| OpenAI quota (429) | `generate_answer` | Retryable with longer backoff (60s, 120s, 240s) to give the quota window time to reset. |
| All hybrid empty + naive fallback empty | workflow logic | `QAResponse.empty(plan)` with `no_results=True`. HTTP 200, valid response. |

The Flask layer maps Temporal errors:
- Workflow-level failure â†’ HTTP 500 with `{error, workflow_id, run_id}`.
- Non-retryable validation errors caused by user input â†’ HTTP 400.

## 11. Observability

- One `logfire.span` per workflow run (tagged with truncated `question`, `workflow_id`, `run_id`).
- Nested span per activity, tagged with key fields (`sub_query.text`, `len(candidates)`, `top_k`).
- Per-stage timings stitched into `QAResponse.latency_ms`: `plan`, `retrieve`, `expand`, `rerank`, `generate`, `total`.
- Structured warning logs on every fallback (`fallback=naive_hybrid`, `fallback=rrf_no_rerank`) with the reason code.
- Temporal UI provides per-question workflow history natively.

## 12. Testing strategy

Five tiers:

1. **Pure-function unit tests** (`test_fusion.py`):
   - RRF determinism over 100 runs of the same input.
   - RRF tie-break stability by `node_id`.
   - RRF empty input â†’ `[]`.
   - RRF single-leg behavior (vector list empty).

2. **Cypher safety** (`test_cypher_safety.py`):
   - Recursively grep `src/qa_agent/` for `f"MATCH"`, `f"CALL "`, `f"RETURN "`, `f"WHERE "`, `+ "MATCH"`, etc. Fails CI if any match.

3. **Planner schema tests** (`test_planner.py`):
   - Mock the LLM client with canned outputs. Assert parsed `Plan` has valid labels, valid expansion patterns, â‰¥1 sub-query.
   - Feed broken JSON. Assert validation retry kicks in.

4. **Workflow replay test** (`test_workflow_replay.py`):
   - Run one happy-path workflow execution, capture the history in memory, replay via `temporalio.testing.WorkflowEnvironment`. Catches accidental non-determinism in workflow code.

5. **End-to-end smoke + small eval** (`test_retrieval_smoke.py`, `fixtures/eval_questions.yaml`):
   - 5â€“10 hand-curated questions paired with expected node IDs that should appear in the reranked top-8.
   - Assertion: for each question, â‰¥1 expected ID is in the top-8 (recall@8 floor).
   - Real local Neo4j + Ollama required; gated behind `pytest -m integration` so CI without those services skips.

## 13. Configuration

`.env` keys (all required unless marked optional):

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
QA_TASK_QUEUE=qa-agent-tq

OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B:Q8_0   # confirm exact tag with user before first run

COHERE_API_KEY=...             # optional for development; missing key triggers RRF fallback
COHERE_RERANK_MODEL=rerank-v3.5

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini

# Numeric knobs (optional; defaults in Â§8.6)
BM25_TOP_K=50
VECTOR_TOP_K=50
RRF_TOP_N=40
MAX_EXPANSION_TOTAL=20
RERANK_TOP_K=8
RERANK_DOC_CHARS=1500
```

The variable name is standardized to `OPENAI_API_KEY` (all caps).

## 14. Open items / risks

- **Embedding model tag.** The exact Ollama model tag for Qwen 3 0.6B Embedding needs confirmation with the user before the first run. The vector index dimension (1024) is authoritative; whichever Ollama model is used must produce 1024-dim vectors.
- **Cohere API key.** User said they will obtain it. Code scaffolds around `COHERE_API_KEY` and falls back gracefully (RRF top-8) if the key is missing, so development can proceed without it.
- **OpenAI quota.** Free-tier limits change; if 429s become frequent the longer-backoff retry policy in Â§10 is the first line of defense. Beyond that, paid-tier upgrade is the natural escape hatch (out of scope for v1).
- **Planner robustness.** The planner is the highest-leverage step for retrieval quality. Few-shot prompt content will need iteration once we see real questions. The fallback path (Â§8.5) ensures a bad plan never blocks an answer â€” it just produces a worse one.
- **Embedding cache.** Currently per-activity only. If we observe the same query getting embedded redundantly across activities (it shouldn't in v1, but if v2 adds re-planning), revisit with an external cache (Redis) â€” never an in-process cache that persists across activity retries.

## 15. Deliverable order (for the implementation plan)

1. `pyproject.toml`, `.env.example`, `README.md`, verify local `.env` uses `OPENAI_API_KEY` and has the required Ollama keys.
2. `config.py`, `schemas.py`, `embeddings.py` + minimum extension to `neo4j_client.py`.
3. `cypher/` files (vector, fulltext, expansion, hydrate) + `test_cypher_safety.py`.
4. `retrieval/` modules (bm25, vector, fusion, expansion, rerank) + `test_fusion.py`.
5. `agents/planner.py` + `prompts/planner.txt` + `test_planner.py`.
6. `agents/answerer.py` + `prompts/answerer.txt`.
7. Activities (`plan`, `retrieve`, `expand`, `rerank`, `generate`).
8. `workflows/qa.py` + `test_workflow_replay.py`.
9. `worker.py`, `starter.py`.
10. `api.py` (Flask).
11. `fixtures/eval_questions.yaml` + `test_retrieval_smoke.py` (gated integration test).
12. Pilot run on 3 representative questions â†’ review with user â†’ iterate planner prompt.

