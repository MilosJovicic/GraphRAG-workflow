# GraphRAG Q&A Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a question-answering agent over the existing Claude Code documentation knowledge graph in Neo4j, using Temporal orchestration, PydanticAI agents (planner + answerer), Qwen 3 0.6B embeddings via Ollama, BM25 + vector hybrid retrieval fused via Reciprocal Rank Fusion, graph expansion over the existing schema, Cohere reranking, and Gemini Flash for generation, exposed via a Flask `POST /ask` endpoint.

**Architecture:** Two long-lived processes (Flask app + Temporal worker) sharing `.env`. Flask kicks off one Temporal workflow per question; the workflow runs plan → parallel hybrid retrieval (per sub-query) → RRF fusion (pure, in-workflow) → graph expansion → Cohere rerank → Gemini Flash generation, with a naive-hybrid fallback path when the planner or rerank fails. Every I/O step is a Temporal activity with its own retries; the workflow body is deterministic.

**Tech Stack:** Python 3.11+, Temporal Python SDK (`temporalio>=1.20`), PydanticAI (>=1.61, including its Google provider support), Pydantic v2, neo4j Python driver, openai SDK (pointed at Ollama), Cohere SDK, Flask 3, Logfire, pytest.

**Reference spec:** [`docs/superpowers/specs/2026-05-05-graphrag-qa-agent-design.md`](../specs/2026-05-05-graphrag-qa-agent-design.md)

---

## File Structure

Files are listed with their single responsibility. Tasks below produce them in this order; tests live alongside as the spec dictates.

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata + pinned deps; source of truth for dependencies |
| `requirements.txt` | Compatibility shim that installs the editable project with dev extras |
| `.env.example` | Template for `.env`; commits no secrets |
| `README.md` | How to install, run worker, hit the API |
| `src/qa_agent/__init__.py` | Package marker |
| `src/qa_agent/config.py` | `Settings` (pydantic-settings) and `get_settings()` cache |
| `src/qa_agent/schemas.py` | All wire types: `QARequest`, `QAResponse`, `Plan`, `SubQuery`, `Candidate`, `Citation`, request envelopes |
| `src/qa_agent/neo4j_client.py` | Driver factories + `run_cypher(filename, params)` helper. Existing file relocated and extended. |
| `src/qa_agent/embeddings.py` | Ollama OpenAI-compatible embedding client; `embed_one`, `embed_batch` |
| `src/qa_agent/cypher/vector_<label>.cypher` | One file per searchable label: vector index lookup |
| `src/qa_agent/cypher/fulltext_<label>.cypher` | One file per searchable label: fulltext index lookup |
| `src/qa_agent/cypher/expand_<pattern>.cypher` | One file per expansion pattern (siblings, parent_page, links, defines, navigates_to) |
| `src/qa_agent/cypher/hydrate_nodes.cypher` | Fetch full node records by id list, regardless of label |
| `src/qa_agent/retrieval/bm25.py` | BM25 leg: composes fulltext call + filter clause from a `SubQuery` |
| `src/qa_agent/retrieval/vector.py` | Vector leg: embeds query, calls vector index per label |
| `src/qa_agent/retrieval/fusion.py` | `rrf_fuse` — pure deterministic RRF over BM25/vector lists |
| `src/qa_agent/retrieval/expansion.py` | Pattern dispatch + per-pattern Cypher invocation, dedup, caps |
| `src/qa_agent/retrieval/rerank.py` | Cohere SDK wrapper; truncates docs, returns top_k |
| `src/qa_agent/agents/planner.py` | PydanticAI `Agent` returning `Plan` |
| `src/qa_agent/agents/answerer.py` | PydanticAI `Agent` returning `AnswerWithCitations`; post-hoc citation regex validation |
| `src/qa_agent/prompts/planner.txt` | Planner system prompt + few-shot examples |
| `src/qa_agent/prompts/answerer.txt` | Answerer system prompt with citation rules |
| `src/qa_agent/activities/plan.py` | `plan_query` activity: thin call into planner agent |
| `src/qa_agent/activities/retrieve.py` | `hybrid_search` (per SubQuery, parallel BM25+vector across labels) and `naive_hybrid_fallback` |
| `src/qa_agent/activities/expand.py` | `expand_graph` activity |
| `src/qa_agent/activities/rerank.py` | `rerank` activity |
| `src/qa_agent/activities/generate.py` | `generate_answer` activity: render evidence, call answerer, validate citations, build `QAResponse` |
| `src/qa_agent/workflows/qa.py` | `QAWorkflow` (the workflow body shown in spec §4.1) |
| `src/qa_agent/worker.py` | Temporal worker entrypoint registering workflow + activities |
| `src/qa_agent/starter.py` | CLI: `python -m qa_agent.starter "question"` |
| `src/qa_agent/api.py` | Flask app: `POST /ask`, `GET /health` |
| `tests/test_schemas.py` | Pydantic validation tests for `Plan`, `SubQuery`, etc. |
| `tests/test_fusion.py` | RRF determinism + edge cases |
| `tests/test_cypher_safety.py` | Greps `src/qa_agent/` for f-string Cypher |
| `tests/test_planner.py` | Planner agent schema validation against canned LLM outputs |
| `tests/test_answerer_citations.py` | Citation regex extraction + validation |
| `tests/test_workflow_replay.py` | Temporal replay test against a recorded happy-path history |
| `tests/test_retrieval_smoke.py` | End-to-end against a real local Neo4j+Ollama, gated `-m integration` |
| `tests/fixtures/eval_questions.yaml` | 5–10 hand-curated questions for v1 eval |

---

## Phase 0: Bootstrap

### Task 0.1: Create pyproject.toml and requirements shim

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`

- [x] **Step 1: Write pyproject.toml and requirements shim**

```toml
[project]
name = "qa-agent"
version = "0.1.0"
description = "GraphRAG Q&A Agent over Claude Code documentation"
requires-python = ">=3.11"
readme = "README.md"
dependencies = [
    "temporalio>=1.20.0",
    "pydantic>=2.7.0",
    "pydantic-ai>=1.61.0",
    "pydantic-settings>=2.4.0",
    "neo4j>=5.20.0",
    "openai>=1.40.0",
    "cohere>=5.10.0",
    "Flask>=3.0.0",
    "logfire>=4.24.0",
    "python-dotenv>=1.0.1",
    "tenacity>=9.0.0",
    "PyYAML>=6.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.6.0",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: integration tests requiring local Neo4j + Ollama (deselect with '-m \"not integration\"')",
]
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100
```

Replace `requirements.txt` with:

```text
# Dependency source of truth is pyproject.toml.
-e .[dev]
```

- [x] **Step 2: Create the empty src base directory**

Run: `mkdir src`
Expected: creates `src/`. This directory is populated with `qa_agent/` in Task 0.4, but it must exist before editable install because `pyproject.toml` uses `where = ["src"]`.

- [x] **Step 3: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: installs all deps, exits 0.

- [x] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add pyproject.toml with project deps"
```

---

### Task 0.2: Create .env.example

**Files:**
- Create: `.env.example`

- [x] **Step 1: Write .env.example**

```
# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme

# Temporal
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
QA_TASK_QUEUE=qa-agent-tq

# Ollama embeddings (Qwen 3 0.6B)
OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B:Q8_0

# Cohere reranker
COHERE_API_KEY=
COHERE_RERANK_MODEL=rerank-v3.5

# Gemini Flash
GEMINI_API_KEY=changeme
GEMINI_MODEL=gemini-2.5-flash

# Retrieval knobs (optional; defaults applied if unset)
BM25_TOP_K=50
VECTOR_TOP_K=50
RRF_TOP_N=40
MAX_EXPANSION_TOTAL=20
RERANK_TOP_K=8
RERANK_DOC_CHARS=1500
```

- [x] **Step 2: Verify .env.example does not contain real secrets**

Run: `grep -E "^(GEMINI_API_KEY|COHERE_API_KEY|NEO4J_PASSWORD)=" .env.example`
Expected: `GEMINI_API_KEY` and `NEO4J_PASSWORD` have the value `changeme`; `COHERE_API_KEY` is blank. No real keys.

- [x] **Step 3: Verify local .env has required key names**

Run this without printing secret values:

```bash
python -c "from dotenv import dotenv_values; required=['NEO4J_URI','NEO4J_USER','NEO4J_PASSWORD','TEMPORAL_HOST','TEMPORAL_NAMESPACE','QA_TASK_QUEUE','OLLAMA_BASE_URL','EMBEDDING_MODEL','GEMINI_API_KEY','GEMINI_MODEL']; env=dotenv_values('.env'); missing=[k for k in required if k not in env]; print('missing=' + (','.join(missing) if missing else 'none'))"
```

Expected: `missing=none`. `COHERE_API_KEY` may be absent or blank during development; rerank will fall back to RRF until it is set.

- [x] **Step 4: Commit**

```bash
git add .env.example
git commit -m "chore: add .env.example template"
```

---

### Task 0.3: Create README.md

**Files:**
- Create: `README.md`

- [x] **Step 1: Write README.md**

```markdown
# Q&A Agent

GraphRAG question-answering over the Claude Code documentation graph in Neo4j.

## Prerequisites

- Python 3.11+
- Local Neo4j with the Claude Code docs graph already loaded (see `../ETL/`) and contextual enrichment populated (see `../Contextual enrichement/`).
- Local Ollama with a Qwen 3 0.6B embedding model that produces 1024-dim vectors.
- A local Temporal server (`temporal server start-dev`).
- A Cohere API key for reranking. This is optional during development; without it, the workflow uses RRF top-8 fallback.
- A Gemini API key (Google AI Studio free tier is fine).

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
# Then edit .env with your credentials.
```

## Run

In one terminal, start the worker:

```bash
python -m qa_agent.worker
```

In another, start the API:

```bash
python -m qa_agent.api    # serves on http://localhost:5000
```

Or hit the workflow directly via CLI:

```bash
python -m qa_agent.starter "How do I configure the Bash tool's permission mode?"
```

## API

`POST /ask`

```json
{ "question": "How do I configure the Bash tool's permission mode?", "debug": false }
```

Returns:

```json
{
  "answer": "Set the permissionMode setting in .claude/settings.json [1] ...",
  "citations": [{"id": 1, "node_id": "...", "url": "...", ...}],
  "plan": {...},
  "latency_ms": {...},
  "fallback_used": false,
  "no_results": false
}
```

## Tests

```bash
pytest                    # unit tests only
pytest -m integration     # integration tests (requires running Neo4j + Ollama)
```

## Design

See [`docs/superpowers/specs/2026-05-05-graphrag-qa-agent-design.md`](docs/superpowers/specs/2026-05-05-graphrag-qa-agent-design.md).
```

- [x] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and run instructions"
```

---

### Task 0.4: Move neo4j_client.py into src/qa_agent/

**Files:**
- Create: `src/qa_agent/__init__.py`
- Move: `neo4j_client.py` → `src/qa_agent/neo4j_client.py`

- [x] **Step 1: Create the package directory and __init__.py**

```bash
mkdir -p src/qa_agent
```

Create `src/qa_agent/__init__.py`:

```python
"""GraphRAG Q&A Agent over Claude Code documentation."""
__version__ = "0.1.0"
```

- [x] **Step 2: Move neo4j_client.py via git**

```bash
git mv neo4j_client.py src/qa_agent/neo4j_client.py
```

- [x] **Step 3: Create empty package subdirs**

```bash
mkdir -p src/qa_agent/cypher src/qa_agent/retrieval src/qa_agent/agents \
         src/qa_agent/prompts src/qa_agent/activities src/qa_agent/workflows
mkdir -p tests/fixtures
```

Create `src/qa_agent/retrieval/__init__.py`, `src/qa_agent/agents/__init__.py`, `src/qa_agent/activities/__init__.py`, `src/qa_agent/workflows/__init__.py`, `tests/__init__.py`, each containing a single line:

```python
```

(Empty file. Use `New-Item -ItemType File <path>` or `touch <path>`.)

- [x] **Step 4: Verify the package imports**

Run: `python -c "import qa_agent; print(qa_agent.__version__)"`
Expected: prints `0.1.0`. Do not import `qa_agent.neo4j_client` yet; that waits for `config.py` in the next task.

- [x] **Step 5: Commit the move**

```bash
git add src/ tests/
git commit -m "refactor: move neo4j_client.py into src/qa_agent/ package layout"
```

---

## Phase 1: Foundation modules

### Task 1.1: config.py — settings via pydantic-settings

**Files:**
- Create: `src/qa_agent/config.py`
- Test: `tests/test_config.py`

- [x] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
import os
from qa_agent.config import Settings, get_settings


def test_settings_reads_required_fields(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("TEMPORAL_HOST", "localhost:7233")
    monkeypatch.setenv("TEMPORAL_NAMESPACE", "default")
    monkeypatch.setenv("QA_TASK_QUEUE", "qa-agent-tq")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding")
    monkeypatch.setenv("COHERE_API_KEY", "ck")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    get_settings.cache_clear()
    s = Settings()
    assert s.neo4j_uri == "neo4j://localhost:7687"
    assert s.qa_task_queue == "qa-agent-tq"
    assert s.cohere_api_key == "ck"
    assert s.bm25_top_k == 50  # default
    assert s.rrf_top_n == 40   # default


def test_get_settings_caches(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
```

- [x] **Step 2: Run test, confirm it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qa_agent.config'` or similar.

- [x] **Step 3: Implement config.py**

`src/qa_agent/config.py`:

```python
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Temporal
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    qa_task_queue: str = "qa-agent-tq"

    # Ollama embeddings
    ollama_base_url: str
    embedding_model: str

    # Cohere
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Retrieval knobs
    bm25_top_k: int = Field(default=50, ge=1, le=500)
    vector_top_k: int = Field(default=50, ge=1, le=500)
    rrf_top_n: int = Field(default=40, ge=1, le=200)
    max_expansion_total: int = Field(default=20, ge=0, le=100)
    rerank_top_k: int = Field(default=8, ge=1, le=50)
    rerank_doc_chars: int = Field(default=1500, ge=200, le=8000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [x] **Step 4: Run test, confirm it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/config.py tests/test_config.py
git commit -m "feat(config): add Settings via pydantic-settings"
```

---

### Task 1.2: schemas.py — wire types

**Files:**
- Create: `src/qa_agent/schemas.py`
- Test: `tests/test_schemas.py`

- [x] **Step 1: Write the failing tests**

`tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from qa_agent.schemas import (
    QARequest, Plan, SubQuery, ExpansionPattern, Candidate, NODE_LABELS,
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
    ExpansionPattern(name="siblings", max_per_seed=5)  # ceiling OK


def test_expansion_pattern_unknown_name_rejected():
    with pytest.raises(ValidationError):
        ExpansionPattern(name="bogus")  # type: ignore


def test_qa_request_min_length():
    with pytest.raises(ValidationError):
        QARequest(question="")


def test_node_labels_constant_complete():
    expected = {
        "Section", "Chunk", "CodeBlock", "TableRow", "Callout",
        "Tool", "Hook", "SettingKey", "PermissionMode",
        "MessageType", "Provider",
    }
    assert set(NODE_LABELS) == expected


def test_candidate_default_scores_none():
    c = Candidate(node_id="n1", node_label="Section", indexed_text="x", raw_text="y")
    assert c.bm25_score is None
    assert c.vector_score is None
    assert c.rrf_score is None
```

- [x] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement schemas.py**

`src/qa_agent/schemas.py`:

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

NODE_LABELS = [
    "Section", "Chunk", "CodeBlock", "TableRow", "Callout",
    "Tool", "Hook", "SettingKey", "PermissionMode",
    "MessageType", "Provider",
]
NodeLabel = Literal[
    "Section", "Chunk", "CodeBlock", "TableRow", "Callout",
    "Tool", "Hook", "SettingKey", "PermissionMode",
    "MessageType", "Provider",
]
ExpansionName = Literal["siblings", "parent_page", "links", "defines", "navigates_to"]


# ---- inputs / outputs ----
class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    debug: bool = False


class Citation(BaseModel):
    id: int
    node_id: str
    node_label: str  # may include Page from graph expansion
    url: str | None = None
    anchor: str | None = None
    title: str | None = None
    snippet: str
    score: float


# ---- planner ----
class SubQuery(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    target_labels: list[NodeLabel] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    bm25_keywords: list[str] = Field(default_factory=list)


class ExpansionPattern(BaseModel):
    name: ExpansionName
    max_per_seed: int = Field(default=3, ge=1, le=5)


class Plan(BaseModel):
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=4)
    expansion_patterns: list[ExpansionPattern] = Field(default_factory=list)
    notes: str = ""

    @classmethod
    def fallback_for(cls, question: str) -> "Plan":
        return cls(
            sub_queries=[SubQuery(
                text=question,
                target_labels=["Section", "CodeBlock", "TableRow", "Callout"],
            )],
            expansion_patterns=[],
            notes="synthetic-fallback",
        )


# ---- retrieval primitives ----
class Candidate(BaseModel):
    node_id: str
    node_label: str
    indexed_text: str
    raw_text: str
    url: str | None = None
    anchor: str | None = None
    breadcrumb: str | None = None
    title: str | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    expansion_origin: str | None = None


# ---- activity envelopes ----
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


# ---- responses ----
class QAResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    plan: Plan
    retrieved: list[Candidate] | None = None
    latency_ms: dict[str, int] = Field(default_factory=dict)
    fallback_used: bool = False
    no_results: bool = False

    @classmethod
    def empty(cls, plan: Plan, fallback_used: bool = False) -> "QAResponse":
        return cls(
            answer=(
                "I couldn't find anything relevant in the knowledge base for that question. "
                "Try rephrasing, or ask about a more specific topic."
            ),
            citations=[],
            plan=plan,
            fallback_used=fallback_used,
            no_results=True,
        )


# ---- agent output (internal) ----
class AnswerWithCitations(BaseModel):
    """Output schema of the answerer agent. Used internally by generate_answer."""
    answer: str = Field(min_length=1, max_length=4000)
    used_citation_ids: list[int] = Field(default_factory=list)
```

- [x] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add wire types for retrieval, planning, response"
```

---

## Phase 2: Pure utilities (TDD)

### Task 2.1: rrf_fuse — Reciprocal Rank Fusion

**Files:**
- Create: `src/qa_agent/retrieval/fusion.py`
- Test: `tests/test_fusion.py`

- [x] **Step 1: Write the failing tests**

`tests/test_fusion.py`:

```python
import pytest
from qa_agent.schemas import Candidate
from qa_agent.retrieval.fusion import rrf_fuse


def _c(nid: str, bm25: float | None = None, vec: float | None = None) -> Candidate:
    return Candidate(
        node_id=nid, node_label="Section", indexed_text=f"text-{nid}", raw_text=f"raw-{nid}",
        bm25_score=bm25, vector_score=vec,
    )


def test_rrf_empty_input_returns_empty():
    assert rrf_fuse([], k=60, top_n=10) == []


def test_rrf_single_subquery_single_leg():
    # only BM25 scores
    cs = [_c("a", bm25=10.0), _c("b", bm25=8.0), _c("c", bm25=5.0)]
    out = rrf_fuse([cs], k=60, top_n=10)
    assert [x.node_id for x in out] == ["a", "b", "c"]
    # rrf_score must be set for all
    assert all(x.rrf_score is not None for x in out)
    # ranked #1 contributes 1/61
    assert out[0].rrf_score == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_single_subquery_both_legs_dedup():
    # candidate "a" appears in both legs ranked differently
    cs = [_c("a", bm25=10.0, vec=0.5), _c("b", bm25=8.0), _c("c", vec=0.9)]
    # bm25 desc: a (#1), b (#2). vector desc: c (#1), a (#2).
    out = rrf_fuse([cs], k=60, top_n=10)
    by_id = {x.node_id: x for x in out}
    assert by_id["a"].rrf_score == pytest.approx(1/61 + 1/62, rel=1e-6)
    assert by_id["c"].rrf_score == pytest.approx(1/61, rel=1e-6)
    assert by_id["b"].rrf_score == pytest.approx(1/62, rel=1e-6)
    # order: a (highest), c, b
    assert [x.node_id for x in out] == ["a", "c", "b"]


def test_rrf_top_n_truncates():
    cs = [_c(f"n{i}", bm25=100 - i) for i in range(10)]
    out = rrf_fuse([cs], k=60, top_n=3)
    assert len(out) == 3
    assert [x.node_id for x in out] == ["n0", "n1", "n2"]


def test_rrf_deterministic_across_runs():
    # ties broken by node_id ascending
    cs = [_c("z", bm25=5.0), _c("a", bm25=5.0), _c("m", bm25=5.0)]
    runs = [tuple(x.node_id for x in rrf_fuse([cs], k=60, top_n=10)) for _ in range(50)]
    assert len(set(runs)) == 1, "rrf_fuse must be deterministic"
    # tie-break: lexicographic ascending
    assert runs[0] == ("a", "m", "z")


def test_rrf_multiple_subqueries_accumulate():
    sq1 = [_c("a", bm25=10.0), _c("b", bm25=5.0)]
    sq2 = [_c("a", vec=0.9), _c("b", vec=0.5)]
    # a is rank #1 in both, b is rank #2 in both
    out = rrf_fuse([sq1, sq2], k=60, top_n=10)
    by_id = {x.node_id: x for x in out}
    assert by_id["a"].rrf_score == pytest.approx(2/61, rel=1e-6)
    assert by_id["b"].rrf_score == pytest.approx(2/62, rel=1e-6)
    assert [x.node_id for x in out] == ["a", "b"]


def test_rrf_preserves_payload():
    cs = [_c("a", bm25=10.0)]
    out = rrf_fuse([cs], k=60, top_n=10)
    assert out[0].indexed_text == "text-a"
    assert out[0].raw_text == "raw-a"
    assert out[0].bm25_score == 10.0
```

- [x] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement fusion.py**

`src/qa_agent/retrieval/fusion.py`:

```python
"""Reciprocal Rank Fusion. Pure, deterministic. Safe to call from a Temporal workflow."""
from __future__ import annotations
from qa_agent.schemas import Candidate


def _rank_by(cands: list[Candidate], score_attr: str) -> list[Candidate]:
    scored = [c for c in cands if getattr(c, score_attr) is not None]
    # Stable sort: by score desc, then node_id asc for deterministic tie-break.
    return sorted(scored, key=lambda c: (-getattr(c, score_attr), c.node_id))


def rrf_fuse(
    per_subquery_results: list[list[Candidate]],
    k: int = 60,
    top_n: int = 40,
) -> list[Candidate]:
    """
    Fuse candidates across (sub_query × {bm25, vector}) ranked lists.

    Each candidate accrues 1/(k+rank) per list it appears in. Final ranking is
    by accumulated rrf_score desc, with stable tie-break by node_id ascending.

    Args:
        per_subquery_results: One list of Candidates per sub-query.
        k: RRF constant (canonical value 60).
        top_n: Max candidates to return.

    Returns:
        At most top_n Candidates with rrf_score populated, deterministic order.
    """
    by_key: dict[tuple[str, str], Candidate] = {}
    scores: dict[tuple[str, str], float] = {}

    for sq_results in per_subquery_results:
        bm25_ranked = _rank_by(sq_results, "bm25_score")
        vec_ranked = _rank_by(sq_results, "vector_score")
        for ranked in (bm25_ranked, vec_ranked):
            for rank, cand in enumerate(ranked, start=1):
                key = (cand.node_id, cand.node_label)
                by_key.setdefault(key, cand)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # Build sorted output. Tie-break: rrf_score desc, then node_id asc.
    sorted_keys = sorted(scores.keys(), key=lambda k_: (-scores[k_], k_[0]))
    out: list[Candidate] = []
    for key in sorted_keys[:top_n]:
        c = by_key[key].model_copy(update={"rrf_score": scores[key]})
        out.append(c)
    return out
```

- [x] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/test_fusion.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/fusion.py src/qa_agent/retrieval/__init__.py tests/test_fusion.py
git commit -m "feat(retrieval): add deterministic RRF fusion"
```

---

## Phase 3: Cypher safety + templates

### Task 3.1: Cypher safety test (TDD red)

**Files:**
- Test: `tests/test_cypher_safety.py`

- [x] **Step 1: Write the test (will pass trivially today, but locks the contract)**

`tests/test_cypher_safety.py`:

```python
"""
Lock the contract: every Cypher in the codebase is a parameterized template
loaded from a .cypher file. No f-string Cypher in Python sources.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qa_agent"

# Patterns that suggest f-string or concat-built Cypher.
FORBIDDEN = [
    re.compile(r'f"\s*MATCH\b', re.IGNORECASE),
    re.compile(r'f"\s*MERGE\b', re.IGNORECASE),
    re.compile(r'f"\s*CREATE\b', re.IGNORECASE),
    re.compile(r'f"\s*CALL\s+db\.', re.IGNORECASE),
    re.compile(r'f"\s*RETURN\b', re.IGNORECASE),
    re.compile(r'f"\s*WHERE\b', re.IGNORECASE),
    re.compile(r'f"\s*UNWIND\b', re.IGNORECASE),
    re.compile(r'f"\s*WITH\b', re.IGNORECASE),
    re.compile(r'\+\s*"MATCH\b', re.IGNORECASE),
    re.compile(r'\+\s*"MERGE\b', re.IGNORECASE),
    re.compile(r'\+\s*"CALL\s+db\.', re.IGNORECASE),
]


def test_no_f_string_cypher_in_src():
    offenders: list[tuple[Path, int, str]] = []
    for py in SRC.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            for pat in FORBIDDEN:
                if pat.search(line):
                    offenders.append((py.relative_to(SRC.parent.parent), i, line.strip()))
    assert offenders == [], "Found f-string Cypher in source:\n" + "\n".join(
        f"  {p}:{i}  {line}" for p, i, line in offenders
    )
```

- [x] **Step 2: Run test, confirm it passes (no Cypher yet)**

Run: `pytest tests/test_cypher_safety.py -v`
Expected: PASS (vacuously — there's no Cypher in src yet).

- [x] **Step 3: Commit**

```bash
git add tests/test_cypher_safety.py
git commit -m "test(cypher): lock contract that all Cypher is loaded from .cypher files"
```

---

### Task 3.2: Vector index Cypher templates

**Files:**
- Create: `src/qa_agent/cypher/vector_section.cypher`
- Create: `src/qa_agent/cypher/vector_chunk.cypher`
- Create: `src/qa_agent/cypher/vector_codeblock.cypher`
- Create: `src/qa_agent/cypher/vector_tablerow.cypher`
- Create: `src/qa_agent/cypher/vector_callout.cypher`

- [x] **Step 1: Write vector_section.cypher**

```cypher
CALL db.index.vector.queryNodes('section_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS s, score
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  s.id          AS node_id,
  'Section'     AS node_label,
  s.indexed_text AS indexed_text,
  s.text        AS raw_text,
  p.url         AS url,
  s.anchor      AS anchor,
  s.breadcrumb  AS breadcrumb,
  s.breadcrumb  AS title,
  score         AS vector_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 2: Write vector_chunk.cypher**

```cypher
CALL db.index.vector.queryNodes('chunk_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section {id: c.section_id})
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  c.id            AS node_id,
  'Chunk'         AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 3: Write vector_codeblock.cypher**

```cypher
CALL db.index.vector.queryNodes('codeblock_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS b, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(b)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($language IS NULL OR b.language = $language)
RETURN
  b.id            AS node_id,
  'CodeBlock'     AS node_label,
  b.indexed_text  AS indexed_text,
  b.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  COALESCE('[' + b.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 4: Write vector_tablerow.cypher**

```cypher
CALL db.index.vector.queryNodes('tablerow_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS r, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE_ROW]->(r)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  r.id            AS node_id,
  'TableRow'      AS node_label,
  r.indexed_text  AS indexed_text,
  r.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 5: Write vector_callout.cypher**

```cypher
CALL db.index.vector.queryNodes('callout_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section)-[:HAS_CALLOUT]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  c.id            AS node_id,
  'Callout'       AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 6: Verify cypher safety test still passes**

Run: `pytest tests/test_cypher_safety.py -v`
Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/qa_agent/cypher/vector_*.cypher
git commit -m "feat(cypher): add vector-index lookup templates per searchable label"
```

---

### Task 3.3: Fulltext index Cypher templates

**Files:**
- Create: `src/qa_agent/cypher/fulltext_section.cypher`
- Create: `src/qa_agent/cypher/fulltext_chunk.cypher`
- Create: `src/qa_agent/cypher/fulltext_codeblock.cypher`
- Create: `src/qa_agent/cypher/fulltext_tablerow.cypher`
- Create: `src/qa_agent/cypher/fulltext_callout.cypher`

- [x] **Step 1: Write fulltext_section.cypher**

```cypher
CALL db.index.fulltext.queryNodes('section_fulltext', $query) YIELD node, score
WITH node AS s, score
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  s.id          AS node_id,
  'Section'     AS node_label,
  s.indexed_text AS indexed_text,
  s.text        AS raw_text,
  p.url         AS url,
  s.anchor      AS anchor,
  s.breadcrumb  AS breadcrumb,
  s.breadcrumb  AS title,
  score         AS bm25_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 2: Write fulltext_chunk.cypher**

```cypher
CALL db.index.fulltext.queryNodes('chunk_fulltext', $query) YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section {id: c.section_id})
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  c.id            AS node_id,
  'Chunk'         AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 3: Write fulltext_codeblock.cypher**

```cypher
CALL db.index.fulltext.queryNodes('codeblock_fulltext', $query) YIELD node, score
WITH node AS b, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(b)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($language IS NULL OR b.language = $language)
  AND ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  b.id            AS node_id,
  'CodeBlock'     AS node_label,
  b.indexed_text  AS indexed_text,
  b.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  COALESCE('[' + b.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 4: Write fulltext_tablerow.cypher**

```cypher
CALL db.index.fulltext.queryNodes('tablerow_fulltext', $query) YIELD node, score
WITH node AS r, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE_ROW]->(r)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  r.id            AS node_id,
  'TableRow'      AS node_label,
  r.indexed_text  AS indexed_text,
  r.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 5: Write fulltext_callout.cypher**

```cypher
CALL db.index.fulltext.queryNodes('callout_fulltext', $query) YIELD node, score
WITH node AS c, score
OPTIONAL MATCH (s:Section)-[:HAS_CALLOUT]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WHERE ($page_path_prefix IS NULL OR p.path STARTS WITH $page_path_prefix)
RETURN
  c.id            AS node_id,
  'Callout'       AS node_label,
  c.indexed_text  AS indexed_text,
  c.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS bm25_score
ORDER BY score DESC
LIMIT $limit
```

- [x] **Step 6: Verify cypher safety test still passes**

Run: `pytest tests/test_cypher_safety.py -v`
Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/qa_agent/cypher/fulltext_*.cypher
git commit -m "feat(cypher): add fulltext (BM25) lookup templates per searchable label"
```

---

### Task 3.4: Graph expansion Cypher templates

**Files:**
- Create: `src/qa_agent/cypher/expand_siblings.cypher`
- Create: `src/qa_agent/cypher/expand_parent_page.cypher`
- Create: `src/qa_agent/cypher/expand_links.cypher`
- Create: `src/qa_agent/cypher/expand_defines.cypher`
- Create: `src/qa_agent/cypher/expand_navigates_to.cypher`
- Create: `src/qa_agent/cypher/hydrate_nodes.cypher`

- [x] **Step 1: Write expand_siblings.cypher**

For Chunk/CodeBlock/Callout/TableRow seeds: find siblings via the parent Section.

```cypher
UNWIND $seeds AS seed
MATCH (parent:Section)-[r]->(seed_node)
WHERE seed_node.id = seed.node_id
  AND any(lbl IN labels(seed_node) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow'])
WITH parent, seed.node_id AS seed_id
MATCH (parent)-[]->(sibling)
WHERE sibling.id IS NOT NULL
  AND sibling.id <> seed_id
  AND any(lbl IN labels(sibling) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow'])
WITH seed_id, sibling, head([lbl IN labels(sibling) WHERE lbl IN ['Chunk','CodeBlock','Callout','TableRow']]) AS sibling_label
RETURN
  sibling.id           AS node_id,
  sibling_label        AS node_label,
  sibling.indexed_text AS indexed_text,
  sibling.text         AS raw_text,
  null                 AS url,
  null                 AS anchor,
  null                 AS breadcrumb,
  null                 AS title,
  seed_id              AS seed_id
LIMIT $cap
```

- [x] **Step 2: Write expand_parent_page.cypher**

```cypher
UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})
MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN DISTINCT
  p.url               AS node_id,
  'Page'              AS node_label,    -- Page nodes have no indexed_text/embedding
  COALESCE(p.title, p.path) AS indexed_text,
  COALESCE(p.description, p.title, p.path) AS raw_text,
  p.url               AS url,
  null                AS anchor,
  null                AS breadcrumb,
  p.title             AS title,
  seed.node_id        AS seed_id
LIMIT $cap
```

Note: Page nodes are skipped by the upstream pipeline so they have no `indexed_text` / `embedding`. We synthesize fallback values from `title`/`description` so they're still usable as evidence.

- [x] **Step 3: Write expand_links.cypher**

```cypher
UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})-[:LINKS_TO|LINKS_TO_PAGE]->(t)
WHERE t.id IS NOT NULL OR t.url IS NOT NULL
WITH seed.node_id AS seed_id, t,
     CASE WHEN 'Section' IN labels(t) THEN 'Section' ELSE 'Page' END AS tlabel
RETURN
  COALESCE(t.id, t.url)         AS node_id,
  tlabel                        AS node_label,
  COALESCE(t.indexed_text, t.title, t.path) AS indexed_text,
  COALESCE(t.text, t.description, t.title)  AS raw_text,
  CASE WHEN tlabel = 'Page' THEN t.url ELSE null END AS url,
  t.anchor                      AS anchor,
  t.breadcrumb                  AS breadcrumb,
  COALESCE(t.title, t.breadcrumb) AS title,
  seed_id                       AS seed_id
LIMIT $cap
```

- [x] **Step 4: Write expand_defines.cypher**

For entity seeds (Tool/Hook/SettingKey/etc.): pull Sections that DEFINE or MENTION them. For Section seeds: pull entities they define/mention.

```cypher
UNWIND $seeds AS seed
OPTIONAL MATCH (s:Section)-[r:DEFINES|MENTIONS]->(e)
  WHERE e.name = seed.node_id OR e.key = seed.node_id
OPTIONAL MATCH (src:Section {id: seed.node_id})-[r2:DEFINES|MENTIONS]->(e2)
WITH seed.node_id AS seed_id,
     collect(DISTINCT s) + collect(DISTINCT e2) AS hits
UNWIND hits AS hit
WITH seed_id, hit
WHERE hit IS NOT NULL
WITH seed_id, hit, head(labels(hit)) AS hlabel
RETURN
  COALESCE(hit.id, hit.name, hit.key, hit.url) AS node_id,
  hlabel                  AS node_label,
  COALESCE(hit.indexed_text, hit.text, hit.description, hit.name) AS indexed_text,
  COALESCE(hit.text, hit.description, hit.name) AS raw_text,
  null                    AS url,
  hit.anchor              AS anchor,
  hit.breadcrumb          AS breadcrumb,
  COALESCE(hit.title, hit.name, hit.breadcrumb) AS title,
  seed_id                 AS seed_id
LIMIT $cap
```

- [x] **Step 5: Write expand_navigates_to.cypher**

```cypher
UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})-[:NAVIGATES_TO]->(t:Section)
WITH seed.node_id AS seed_id, t
RETURN
  t.id                AS node_id,
  'Section'           AS node_label,
  t.indexed_text      AS indexed_text,
  t.text              AS raw_text,
  null                AS url,
  t.anchor            AS anchor,
  t.breadcrumb        AS breadcrumb,
  t.breadcrumb        AS title,
  seed_id             AS seed_id
LIMIT $cap
```

- [x] **Step 6: Write hydrate_nodes.cypher**

Used by tests / debug paths to fetch full node records by id (regardless of label):

```cypher
UNWIND $ids AS id
OPTIONAL MATCH (n) WHERE n.id = id OR n.url = id OR n.name = id OR n.key = id
WITH id, n
WHERE n IS NOT NULL
RETURN
  id                   AS node_id,
  head(labels(n))      AS node_label,
  n.indexed_text       AS indexed_text,
  COALESCE(n.text, n.description, n.title, n.name) AS raw_text
```

- [x] **Step 7: Verify cypher safety test still passes**

Run: `pytest tests/test_cypher_safety.py -v`
Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add src/qa_agent/cypher/expand_*.cypher src/qa_agent/cypher/hydrate_nodes.cypher
git commit -m "feat(cypher): add graph expansion templates and hydrate query"
```

---

## Phase 4: I/O wrappers

### Task 4.1: Extend neo4j_client.py with run_cypher helper

**Files:**
- Modify: `src/qa_agent/neo4j_client.py`
- Test: `tests/test_neo4j_client.py`

- [x] **Step 1: Write a unit-test of the cypher loader (no DB call)**

`tests/test_neo4j_client.py`:

```python
import pytest
from qa_agent.neo4j_client import load_cypher


def test_load_cypher_existing_file():
    txt = load_cypher("vector_section.cypher")
    assert "db.index.vector.queryNodes" in txt
    assert "section_embedding" in txt


def test_load_cypher_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_cypher("does_not_exist.cypher")
```

- [x] **Step 2: Run, expect failure if loader is broken**

Run: `pytest tests/test_neo4j_client.py -v`

If FAIL: implement step 3. If PASS: continue (loader is already correct from Phase 0; only `run_cypher` helper is missing — add it anyway).

- [x] **Step 3: Replace neo4j_client.py with the extended version**

`src/qa_agent/neo4j_client.py`:

```python
from pathlib import Path
from functools import lru_cache
from neo4j import GraphDatabase, AsyncGraphDatabase
from qa_agent.config import get_settings

_CYPHER_DIR = Path(__file__).parent / "cypher"


def load_cypher(relative_path: str) -> str:
    path = _CYPHER_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Cypher template not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache
def get_driver():
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


@lru_cache
def get_async_driver():
    s = get_settings()
    return AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def run_query(cypher: str, parameters: dict | None = None) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, parameters or {})
        return [dict(record) for record in result]


async def run_query_async(cypher: str, parameters: dict | None = None) -> list[dict]:
    driver = get_async_driver()
    async with driver.session() as session:
        result = await session.run(cypher, parameters or {})
        return [dict(record) async for record in result]


async def run_cypher(filename: str, parameters: dict | None = None) -> list[dict]:
    """Load a parameterized .cypher template and execute it against Neo4j."""
    return await run_query_async(load_cypher(filename), parameters)
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_neo4j_client.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/neo4j_client.py tests/test_neo4j_client.py
git commit -m "feat(neo4j): add run_cypher helper and FileNotFoundError on missing template"
```

---

### Task 4.2: embeddings.py — Ollama OpenAI-compatible client

**Files:**
- Create: `src/qa_agent/embeddings.py`
- Test: `tests/test_embeddings.py`

- [x] **Step 1: Write the failing test (mock the OpenAI client)**

`tests/test_embeddings.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qa_agent.embeddings import embed_one, embed_batch


@pytest.mark.asyncio
async def test_embed_one_returns_vector(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding")
    monkeypatch.setenv("NEO4J_URI", "x"); monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x"); monkeypatch.setenv("COHERE_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1024)]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    with patch("qa_agent.embeddings._get_client", return_value=fake_client):
        from qa_agent.config import get_settings
        get_settings.cache_clear()
        v = await embed_one("hello world")

    assert len(v) == 1024
    assert v[0] == 0.1
    fake_client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_input(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding")
    monkeypatch.setenv("NEO4J_URI", "x"); monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x"); monkeypatch.setenv("COHERE_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[float(i)] * 4) for i in range(3)]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    with patch("qa_agent.embeddings._get_client", return_value=fake_client):
        from qa_agent.config import get_settings
        get_settings.cache_clear()
        vs = await embed_batch(["a", "b", "c"])

    assert len(vs) == 3
    assert vs[0] == [0.0, 0.0, 0.0, 0.0]
    assert vs[2] == [2.0, 2.0, 2.0, 2.0]
```

- [x] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL.

- [x] **Step 3: Implement embeddings.py**

`src/qa_agent/embeddings.py`:

```python
"""Ollama OpenAI-compatible embedding client for Qwen 3 0.6B."""
from __future__ import annotations
from functools import lru_cache
from openai import AsyncOpenAI
from qa_agent.config import get_settings


@lru_cache
def _get_client() -> AsyncOpenAI:
    s = get_settings()
    # Ollama doesn't require an API key; SDK insists on a non-empty value.
    return AsyncOpenAI(base_url=s.ollama_base_url, api_key="ollama")


async def embed_one(text: str) -> list[float]:
    s = get_settings()
    client = _get_client()
    resp = await client.embeddings.create(model=s.embedding_model, input=text)
    return list(resp.data[0].embedding)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = _get_client()
    resp = await client.embeddings.create(model=s.embedding_model, input=texts)
    # Preserve input order
    by_index = {d.index: list(d.embedding) for d in resp.data}
    return [by_index[i] for i in range(len(texts))]
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): add Ollama OpenAI-compatible embed_one/embed_batch"
```

---

### Task 4.3: retrieval/bm25.py

**Files:**
- Create: `src/qa_agent/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

- [x] **Step 1: Write the failing test (mocks run_cypher)**

`tests/test_bm25.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from qa_agent.schemas import SubQuery
from qa_agent.retrieval.bm25 import bm25_search


@pytest.mark.asyncio
async def test_bm25_search_section_label_calls_correct_template():
    sq = SubQuery(text="how to configure", target_labels=["Section"], bm25_keywords=["bash"])
    fake_records = [{
        "node_id": "s1", "node_label": "Section", "indexed_text": "x", "raw_text": "y",
        "url": "https://...", "anchor": "h1", "breadcrumb": "Home > Bash",
        "title": "Bash", "bm25_score": 12.0,
    }]
    with patch("qa_agent.retrieval.bm25.run_cypher", new=AsyncMock(return_value=fake_records)) as m:
        out = await bm25_search(sq, label="Section", limit=50)

    assert len(out) == 1
    assert out[0].node_id == "s1"
    assert out[0].bm25_score == 12.0
    # Filename matches the label
    args, kwargs = m.call_args
    assert args[0] == "fulltext_section.cypher"
    # Query string includes both subquery text and bm25_keywords
    assert "how to configure" in args[1]["query"]
    assert "bash" in args[1]["query"]
    assert args[1]["limit"] == 50


@pytest.mark.asyncio
async def test_bm25_search_unknown_label_raises():
    sq = SubQuery(text="x", target_labels=["Section"])
    with pytest.raises(ValueError):
        await bm25_search(sq, label="Page", limit=50)


@pytest.mark.asyncio
async def test_bm25_search_drops_unknown_filter_keys():
    sq = SubQuery(text="x", target_labels=["Section"], filters={"language": "python", "bogus": "v"})
    with patch("qa_agent.retrieval.bm25.run_cypher", new=AsyncMock(return_value=[])) as m:
        await bm25_search(sq, label="CodeBlock", limit=50)
    args, _ = m.call_args
    params = args[1]
    assert params.get("language") == "python"
    assert "bogus" not in params
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_bm25.py -v`
Expected: FAIL.

- [x] **Step 3: Implement bm25.py**

`src/qa_agent/retrieval/bm25.py`:

```python
"""BM25 (fulltext) retrieval leg. One Cypher template per searchable label."""
from __future__ import annotations
from qa_agent.schemas import SubQuery, Candidate
from qa_agent.neo4j_client import run_cypher

SEARCHABLE_LABELS = {"Section", "Chunk", "CodeBlock", "TableRow", "Callout"}
ALLOWED_FILTERS = {"language", "page_path_prefix"}


def _template_for(label: str) -> str:
    return f"fulltext_{label.lower()}.cypher"


def _build_query_string(sq: SubQuery) -> str:
    parts = [sq.text.strip()]
    if sq.bm25_keywords:
        parts.append(" ".join(sq.bm25_keywords))
    return " ".join(p for p in parts if p)


def _build_filter_params(filters: dict[str, str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {k: None for k in ALLOWED_FILTERS}
    for k, v in filters.items():
        if k in ALLOWED_FILTERS:
            out[k] = v
    return out


async def bm25_search(sq: SubQuery, label: str, limit: int) -> list[Candidate]:
    if label not in SEARCHABLE_LABELS:
        raise ValueError(f"BM25 search not supported for label {label!r}")
    params = {
        "query": _build_query_string(sq),
        "limit": limit,
        **_build_filter_params(sq.filters),
    }
    rows = await run_cypher(_template_for(label), params)
    return [Candidate(**row) for row in rows]
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_bm25.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/bm25.py tests/test_bm25.py
git commit -m "feat(retrieval): add BM25 leg with allow-listed filters"
```

---

### Task 4.4: retrieval/vector.py

**Files:**
- Create: `src/qa_agent/retrieval/vector.py`
- Test: `tests/test_vector.py`

- [x] **Step 1: Write the failing test**

`tests/test_vector.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from qa_agent.schemas import SubQuery
from qa_agent.retrieval.vector import vector_search


@pytest.mark.asyncio
async def test_vector_search_embeds_then_queries():
    sq = SubQuery(text="permission mode", target_labels=["Section"])
    fake_records = [{
        "node_id": "s1", "node_label": "Section", "indexed_text": "x", "raw_text": "y",
        "url": "u", "anchor": "a", "breadcrumb": "b", "title": "t",
        "vector_score": 0.91,
    }]

    with patch("qa_agent.retrieval.vector.embed_one",
               new=AsyncMock(return_value=[0.1] * 1024)) as emb, \
         patch("qa_agent.retrieval.vector.run_cypher",
               new=AsyncMock(return_value=fake_records)) as cy:
        out = await vector_search(sq, label="Section", limit=50)

    emb.assert_awaited_once_with("permission mode")
    args, _ = cy.call_args
    assert args[0] == "vector_section.cypher"
    assert args[1]["limit"] == 50
    assert len(args[1]["query_vector"]) == 1024
    assert out[0].vector_score == 0.91


@pytest.mark.asyncio
async def test_vector_search_unknown_label_raises():
    sq = SubQuery(text="x", target_labels=["Section"])
    with pytest.raises(ValueError):
        await vector_search(sq, label="Page", limit=50)
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_vector.py -v`
Expected: FAIL.

- [x] **Step 3: Implement vector.py**

`src/qa_agent/retrieval/vector.py`:

```python
"""Vector retrieval leg. Embeds the query, calls the matching vector index."""
from __future__ import annotations
from qa_agent.schemas import SubQuery, Candidate
from qa_agent.embeddings import embed_one
from qa_agent.neo4j_client import run_cypher
from qa_agent.retrieval.bm25 import SEARCHABLE_LABELS, ALLOWED_FILTERS, _build_filter_params


def _template_for(label: str) -> str:
    return f"vector_{label.lower()}.cypher"


async def vector_search(sq: SubQuery, label: str, limit: int) -> list[Candidate]:
    if label not in SEARCHABLE_LABELS:
        raise ValueError(f"Vector search not supported for label {label!r}")
    qvec = await embed_one(sq.text)
    params = {
        "query_vector": qvec,
        "limit": limit,
        **_build_filter_params(sq.filters),
    }
    rows = await run_cypher(_template_for(label), params)
    return [Candidate(**row) for row in rows]
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_vector.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/vector.py tests/test_vector.py
git commit -m "feat(retrieval): add vector leg using embed_one + per-label template"
```

---

### Task 4.5: retrieval/expansion.py

**Files:**
- Create: `src/qa_agent/retrieval/expansion.py`
- Test: `tests/test_expansion.py`

- [x] **Step 1: Write the failing test**

`tests/test_expansion.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from qa_agent.schemas import Candidate, ExpansionPattern
from qa_agent.retrieval.expansion import expand


def _seed(nid: str, label: str = "Section") -> Candidate:
    return Candidate(node_id=nid, node_label=label, indexed_text="x", raw_text="y")


@pytest.mark.asyncio
async def test_expand_dispatches_per_pattern_and_dedupes():
    seeds = [_seed("s1"), _seed("s2")]
    patterns = [ExpansionPattern(name="links", max_per_seed=2)]

    fake_links_rows = [
        {"node_id": "t1", "node_label": "Section", "indexed_text": "x", "raw_text": "y",
         "url": None, "anchor": None, "breadcrumb": None, "title": "T1", "seed_id": "s1"},
        # duplicate node_id should be deduped
        {"node_id": "t1", "node_label": "Section", "indexed_text": "x", "raw_text": "y",
         "url": None, "anchor": None, "breadcrumb": None, "title": "T1", "seed_id": "s2"},
    ]
    with patch("qa_agent.retrieval.expansion.run_cypher",
               new=AsyncMock(return_value=fake_links_rows)) as m:
        out = await expand(seeds, patterns, total_cap=20)

    # Seeds first
    assert [c.node_id for c in out[:2]] == ["s1", "s2"]
    # Single t1 (deduped)
    expanded = [c for c in out if c.node_id == "t1"]
    assert len(expanded) == 1
    assert expanded[0].expansion_origin and expanded[0].expansion_origin.startswith("links:")
    # Cypher called with pattern file
    args, _ = m.call_args
    assert args[0] == "expand_links.cypher"


@pytest.mark.asyncio
async def test_expand_total_cap_truncates_expansions_but_not_seeds():
    seeds = [_seed(f"s{i}") for i in range(3)]
    patterns = [ExpansionPattern(name="siblings", max_per_seed=5)]

    rows = [
        {"node_id": f"e{i}", "node_label": "Chunk",
         "indexed_text": "x", "raw_text": "y",
         "url": None, "anchor": None, "breadcrumb": None, "title": None, "seed_id": "s0"}
        for i in range(50)
    ]
    with patch("qa_agent.retrieval.expansion.run_cypher",
               new=AsyncMock(return_value=rows)):
        out = await expand(seeds, patterns, total_cap=4)

    # 3 seeds + 4 expansions = 7
    assert len(out) == 3 + 4
    # seeds preserved
    assert {c.node_id for c in out[:3]} == {"s0", "s1", "s2"}


@pytest.mark.asyncio
async def test_expand_no_patterns_returns_seeds_only():
    seeds = [_seed("s1")]
    out = await expand(seeds, [], total_cap=20)
    assert [c.node_id for c in out] == ["s1"]
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_expansion.py -v`
Expected: FAIL.

- [x] **Step 3: Implement expansion.py**

`src/qa_agent/retrieval/expansion.py`:

```python
"""Graph expansion: fan out from RRF seeds via fixed Cypher patterns."""
from __future__ import annotations
from qa_agent.schemas import Candidate, ExpansionPattern
from qa_agent.neo4j_client import run_cypher


_PATTERN_TEMPLATE = {
    "siblings":      "expand_siblings.cypher",
    "parent_page":   "expand_parent_page.cypher",
    "links":         "expand_links.cypher",
    "defines":       "expand_defines.cypher",
    "navigates_to":  "expand_navigates_to.cypher",
}


def _row_to_candidate(row: dict, pattern: str) -> Candidate:
    return Candidate(
        node_id=row["node_id"],
        node_label=row["node_label"],
        indexed_text=row.get("indexed_text") or "",
        raw_text=row.get("raw_text") or "",
        url=row.get("url"),
        anchor=row.get("anchor"),
        breadcrumb=row.get("breadcrumb"),
        title=row.get("title"),
        expansion_origin=f"{pattern}:{row.get('seed_id', '')}",
    )


async def expand(
    seeds: list[Candidate],
    patterns: list[ExpansionPattern],
    total_cap: int,
) -> list[Candidate]:
    """Run each pattern's Cypher, dedupe expansions vs seeds, cap totals.

    Output ordering: seeds first (in input order), then expansions in
    pattern-order then row-order, deduped by (node_id, node_label).
    """
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for s in seeds:
        key = (s.node_id, s.node_label)
        if key not in seen:
            seen.add(key)
            out.append(s)

    expansions: list[Candidate] = []
    for p in patterns:
        template = _PATTERN_TEMPLATE.get(p.name)
        if not template:
            continue
        seed_payload = [{"node_id": s.node_id, "node_label": s.node_label} for s in seeds]
        rows = await run_cypher(template, {
            "seeds": seed_payload,
            "cap": p.max_per_seed * max(len(seeds), 1),
        })
        for row in rows:
            cand = _row_to_candidate(row, p.name)
            key = (cand.node_id, cand.node_label)
            if key in seen:
                continue
            seen.add(key)
            expansions.append(cand)
            if len(expansions) >= total_cap:
                break
        if len(expansions) >= total_cap:
            break

    out.extend(expansions[:total_cap])
    return out
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_expansion.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/expansion.py tests/test_expansion.py
git commit -m "feat(retrieval): add graph expansion with dedup and total_cap"
```

---

### Task 4.6: retrieval/rerank.py

**Files:**
- Create: `src/qa_agent/retrieval/rerank.py`
- Test: `tests/test_rerank.py`

- [x] **Step 1: Write the failing test**

`tests/test_rerank.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qa_agent.schemas import Candidate
from qa_agent.retrieval.rerank import cohere_rerank


def _c(nid: str, text: str) -> Candidate:
    return Candidate(node_id=nid, node_label="Section", indexed_text=text, raw_text=text)


@pytest.mark.asyncio
async def test_cohere_rerank_empty_input_returns_empty():
    out = await cohere_rerank("hello", [], top_k=8, doc_chars=1500)
    assert out == []


@pytest.mark.asyncio
async def test_cohere_rerank_truncates_documents_and_reorders(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake")
    monkeypatch.setenv("NEO4J_URI", "x"); monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x"); monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x"); monkeypatch.setenv("GEMINI_API_KEY", "x")

    cands = [_c("a", "A" * 3000), _c("b", "B" * 100), _c("c", "C" * 100)]

    fake_resp = MagicMock(results=[
        MagicMock(index=2, relevance_score=0.9),
        MagicMock(index=0, relevance_score=0.6),
    ])
    fake_client = MagicMock()
    fake_client.rerank = AsyncMock(return_value=fake_resp)

    with patch("qa_agent.retrieval.rerank._get_client", return_value=fake_client):
        from qa_agent.config import get_settings
        get_settings.cache_clear()
        out = await cohere_rerank("q", cands, top_k=2, doc_chars=1500)

    # Reordered by relevance
    assert [c.node_id for c in out] == ["c", "a"]
    assert out[0].rerank_score == 0.9
    assert out[1].rerank_score == 0.6

    # Documents passed to Cohere were truncated
    args, kwargs = fake_client.rerank.call_args
    docs = kwargs["documents"]
    assert all(len(d) <= 1500 for d in docs)
    assert kwargs["top_n"] == 2
    assert kwargs["query"] == "q"
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_rerank.py -v`
Expected: FAIL.

- [x] **Step 3: Implement rerank.py**

`src/qa_agent/retrieval/rerank.py`:

```python
"""Cohere rerank wrapper. Returns top_k candidates with rerank_score set."""
from __future__ import annotations
from functools import lru_cache
import cohere
from qa_agent.schemas import Candidate
from qa_agent.config import get_settings


@lru_cache
def _get_client() -> cohere.AsyncClientV2:
    s = get_settings()
    if not s.cohere_api_key or s.cohere_api_key == "changeme":
        raise RuntimeError("COHERE_API_KEY is not configured")
    return cohere.AsyncClientV2(api_key=s.cohere_api_key)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]


async def cohere_rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int,
    doc_chars: int,
) -> list[Candidate]:
    if not candidates:
        return []
    s = get_settings()
    client = _get_client()
    docs = [_truncate(c.indexed_text or c.raw_text or "", doc_chars) for c in candidates]
    resp = await client.rerank(
        model=s.cohere_rerank_model,
        query=query,
        documents=docs,
        top_n=min(top_k, len(candidates)),
    )
    out: list[Candidate] = []
    for r in resp.results:
        c = candidates[r.index].model_copy(update={"rerank_score": float(r.relevance_score)})
        out.append(c)
    return out
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_rerank.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/rerank.py tests/test_rerank.py
git commit -m "feat(retrieval): add Cohere rerank with doc truncation"
```

---

## Phase 5: Agents

### Task 5.1: prompts/planner.txt

**Files:**
- Create: `src/qa_agent/prompts/planner.txt`

- [x] **Step 1: Write the planner prompt**

`src/qa_agent/prompts/planner.txt`:

```
You plan retrieval over a knowledge graph of Claude Code documentation. Your job
is to decompose a user question into a small Plan that drives a hybrid (BM25 +
vector) search and optional graph expansion.

# Knowledge graph labels you can target

- Section: a section of a documentation page (most general; carries text + breadcrumb).
- Chunk: a paragraph-sized slice of a Section (for fine-grained retrieval inside long sections).
- CodeBlock: a code example. Has a `language` field — use the `language` filter to constrain.
- TableRow: a row from a documentation table (e.g. settings tables, command flag tables).
- Callout: an inline note/warning/tip block.
- Tool: a Claude Code tool definition (Bash, Read, Edit, etc.).
- Hook: a Claude Code hook event (PreToolUse, PostToolUse, etc.).
- SettingKey: a settings.json key (model, env, permissions, hooks, etc.).
- PermissionMode: one of acceptEdits, bypassPermissions, default, plan.
- MessageType: an SDK message type (assistant, user, tool_result, etc.).
- Provider: an LLM provider (anthropic, vertex, bedrock).

# Allow-listed filters

- language: language tag for CodeBlock (e.g. "python", "bash", "typescript").
- page_path_prefix: restrict to a path prefix on Page (e.g. "settings/").

Other filter keys are silently dropped — only emit `language` and `page_path_prefix`.

# Graph expansion patterns

After hybrid retrieval finds seed candidates, you may request expansion patterns:

- siblings: from a Chunk/CodeBlock/Callout/TableRow seed, fetch other children of the same Section. Useful when context around the seed matters.
- parent_page: from a Section descendant, fetch the parent Page. Useful when the user's question is about a whole topic, not a specific subsection.
- links: from a Section seed, follow `:LINKS_TO` and `:LINKS_TO_PAGE`. Useful for "related" content.
- defines: from an entity seed (Tool/Hook/SettingKey/etc.), fetch Sections that DEFINE or MENTION it (or vice versa).
- navigates_to: from a Section seed, follow `:NAVIGATES_TO` (cross-page navigation tables).

# Output schema (you MUST match this exactly)

{
  "sub_queries": [
    {
      "text": "<cleaned/expanded search query>",
      "target_labels": ["<NodeLabel>", ...],
      "filters": {"language": "...", "page_path_prefix": "..."},   // optional
      "bm25_keywords": ["<keyword>", ...]                            // optional
    }
  ],
  "expansion_patterns": [
    {"name": "<pattern>", "max_per_seed": 3}
  ],
  "notes": "<one-line rationale, optional>"
}

Rules:
- 1 to 4 sub_queries. Use multiple only when the question genuinely covers
  multiple distinct topics or asks for a comparison.
- target_labels: pick the labels most likely to carry the answer. Prefer
  Section + Chunk for prose questions; CodeBlock for "show me the code";
  TableRow for "what flags / fields"; Tool/SettingKey/etc. for "what is X".
- bm25_keywords: extract 2-5 dense keywords/phrases the user explicitly used or
  obvious synonyms. Skip stopwords. These help BM25 catch what vectors miss.
- expansion_patterns: pick zero or one or two patterns that match the shape of
  the question. Skip when the question is very narrow.

# Examples

User: "How do I configure the Bash tool's permission mode?"

{
  "sub_queries": [
    {"text": "configure Bash tool permission mode", "target_labels": ["Section","SettingKey","Tool"], "bm25_keywords": ["Bash","permission","permissionMode"]}
  ],
  "expansion_patterns": [{"name": "defines", "max_per_seed": 3}],
  "notes": "tool+settings; defines pulls Sections that define the SettingKey"
}

User: "Show me a Python example of using the Edit tool."

{
  "sub_queries": [
    {"text": "Edit tool Python example", "target_labels": ["CodeBlock","Section"], "filters": {"language": "python"}, "bm25_keywords": ["Edit","python"]}
  ],
  "expansion_patterns": [{"name": "siblings", "max_per_seed": 2}],
  "notes": "want a CodeBlock; siblings pull surrounding context"
}

User: "What are all the hook events and when do they fire?"

{
  "sub_queries": [
    {"text": "Claude Code hook events list", "target_labels": ["Hook","Section","TableRow"], "bm25_keywords": ["hook","event"]}
  ],
  "expansion_patterns": [{"name": "defines", "max_per_seed": 3}, {"name": "siblings", "max_per_seed": 2}],
  "notes": "entity-list question; defines pulls Sections that document each hook"
}

User: "Compare the bypassPermissions and acceptEdits permission modes."

{
  "sub_queries": [
    {"text": "bypassPermissions permission mode", "target_labels": ["PermissionMode","Section","TableRow"], "bm25_keywords": ["bypassPermissions"]},
    {"text": "acceptEdits permission mode", "target_labels": ["PermissionMode","Section","TableRow"], "bm25_keywords": ["acceptEdits"]}
  ],
  "expansion_patterns": [{"name": "defines", "max_per_seed": 3}],
  "notes": "comparison: one sub_query per side"
}

Now produce the Plan for the user's question.
```

- [x] **Step 2: Commit**

```bash
git add src/qa_agent/prompts/planner.txt
git commit -m "feat(prompts): add planner prompt with label catalog and few-shots"
```

---

### Task 5.2: agents/planner.py

**Files:**
- Create: `src/qa_agent/agents/planner.py`
- Test: `tests/test_planner.py`

- [x] **Step 1: Write the failing test (mocks PydanticAI)**

`tests/test_planner.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from pydantic_ai.models.test import TestModel
from qa_agent.agents.planner import build_planner_agent, plan_question
from qa_agent.schemas import Plan


@pytest.mark.asyncio
async def test_planner_returns_valid_plan_with_test_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x"); monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("NEO4J_URI", "x"); monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x"); monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x"); monkeypatch.setenv("COHERE_API_KEY", "x")

    from qa_agent.config import get_settings
    get_settings.cache_clear()
    agent = build_planner_agent(model=TestModel())
    plan = await plan_question("How do I configure Bash?", agent=agent)
    assert isinstance(plan, Plan)
    assert len(plan.sub_queries) >= 1
```

(Note: PydanticAI's `TestModel` returns a deterministic stub matching the output schema. This test only verifies the wiring.)

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL.

- [x] **Step 3: Implement planner.py**

`src/qa_agent/agents/planner.py`:

```python
"""Planner agent: turns a user question into a Plan."""
from __future__ import annotations
from pathlib import Path
from pydantic_ai import Agent
from qa_agent.schemas import Plan
from qa_agent.config import get_settings

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_planner_agent(model: object | None = None) -> Agent[None, Plan]:
    """Build the PydanticAI planner agent.

    Args:
        model: optional override (e.g. TestModel for tests). When None, uses
        Gemini Flash via the GEMINI_API_KEY env.
    """
    if model is None:
        s = get_settings()
        # PydanticAI accepts "google-gla:<model_id>" with GEMINI_API_KEY in env.
        model = f"google-gla:{s.gemini_model}"
    return Agent(
        model=model,
        output_type=Plan,
        system_prompt=_load_prompt(),
    )


async def plan_question(question: str, agent: Agent[None, Plan] | None = None) -> Plan:
    a = agent if agent is not None else build_planner_agent()
    result = await a.run(question)
    return result.output
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_planner.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/agents/planner.py tests/test_planner.py
git commit -m "feat(agents): add planner agent producing Plan via PydanticAI"
```

---

### Task 5.3: prompts/answerer.txt

**Files:**
- Create: `src/qa_agent/prompts/answerer.txt`

- [x] **Step 1: Write the answerer prompt**

`src/qa_agent/prompts/answerer.txt`:

```
You answer the user's question using ONLY the numbered evidence below. Each piece
of evidence has an integer id; you must cite the evidence you use with bracketed
markers like [1] or [2][5].

Rules:
1. Use only the provided evidence. If the evidence does not support an answer,
   reply with exactly: "The provided documentation does not contain a clear
   answer to this question." (no citations).
2. Cite every factual claim. Multiple sources for one claim: [1][3].
3. Do not restate the question. Do not add a preamble or sign-off.
4. Be precise. Prefer the documentation's exact terminology over paraphrase.
5. When code or commands are part of the answer, include them verbatim from the
   evidence and cite them.
6. Do not invent ids. Cite only ids you actually see below.
7. After the answer, list the ids you used in `used_citation_ids` (a JSON array).

# Question

{question}

# Evidence

{evidence_block}

# Output

Return JSON matching this schema:

{
  "answer": "<your answer with [n] markers>",
  "used_citation_ids": [<int>, ...]
}
```

- [x] **Step 2: Commit**

```bash
git add src/qa_agent/prompts/answerer.txt
git commit -m "feat(prompts): add answerer prompt with strict citation rules"
```

---

### Task 5.4: agents/answerer.py + citation validation

**Files:**
- Create: `src/qa_agent/agents/answerer.py`
- Test: `tests/test_answerer_citations.py`

- [x] **Step 1: Write the failing test for citation validation**

`tests/test_answerer_citations.py`:

```python
import pytest
from qa_agent.agents.answerer import extract_citation_ids, validate_citations


def test_extract_citation_ids_simple():
    assert extract_citation_ids("Use foo [1] and bar [2][3].") == [1, 2, 3]


def test_extract_citation_ids_dedup_preserves_first_seen_order():
    assert extract_citation_ids("[2] x [1] y [2]") == [2, 1]


def test_extract_citation_ids_empty():
    assert extract_citation_ids("no citations here.") == []


def test_validate_citations_passes_when_in_range():
    ok, missing, out_of_range = validate_citations("Foo [1][2].", evidence_count=3, used_ids=[1,2])
    assert ok is True
    assert missing == [] and out_of_range == []


def test_validate_citations_flags_out_of_range():
    ok, missing, oor = validate_citations("Foo [9].", evidence_count=3, used_ids=[9])
    assert ok is False
    assert oor == [9]


def test_validate_citations_flags_used_but_not_in_text():
    ok, missing, oor = validate_citations("Foo [1].", evidence_count=3, used_ids=[1, 2])
    assert ok is False
    assert missing == [2]
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_answerer_citations.py -v`
Expected: FAIL.

- [x] **Step 3: Implement answerer.py**

`src/qa_agent/agents/answerer.py`:

```python
"""Answerer agent: cited answer from numbered evidence."""
from __future__ import annotations
import re
from pathlib import Path
from pydantic_ai import Agent
from qa_agent.schemas import AnswerWithCitations, Candidate
from qa_agent.config import get_settings

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "answerer.txt"
_CITE_RE = re.compile(r"\[(\d+)\]")
_EVIDENCE_TEXT_CHARS = 800


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_evidence(candidates: list[Candidate]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        title = c.title or c.breadcrumb or c.node_id
        url_part = f"\n    url: {c.url}{('#' + c.anchor) if c.anchor else ''}" if c.url else ""
        body = (c.indexed_text or c.raw_text or "")[:_EVIDENCE_TEXT_CHARS]
        lines.append(f"[{i}] {c.node_label} {title}{url_part}\n    {body}")
    return "\n\n".join(lines)


def extract_citation_ids(answer: str) -> list[int]:
    """Return citation ids in first-seen order, deduplicated."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate_citations(
    answer: str,
    evidence_count: int,
    used_ids: list[int],
) -> tuple[bool, list[int], list[int]]:
    """Return (ok, missing_in_text, out_of_range).

    missing_in_text: ids in used_ids that don't appear as [n] markers in answer.
    out_of_range: any [n] in answer where n < 1 or n > evidence_count.
    """
    extracted = extract_citation_ids(answer)
    out_of_range = [n for n in extracted if n < 1 or n > evidence_count]
    extracted_set = set(extracted)
    missing = [u for u in used_ids if u not in extracted_set]
    return (not out_of_range and not missing, missing, out_of_range)


def build_answerer_agent(model: object | None = None) -> Agent[None, AnswerWithCitations]:
    if model is None:
        s = get_settings()
        model = f"google-gla:{s.gemini_model}"
    return Agent(
        model=model,
        output_type=AnswerWithCitations,
        system_prompt="(replaced per call)",  # we render the full prompt at run-time
    )


async def answer_with_citations(
    question: str,
    evidence: list[Candidate],
    agent: Agent[None, AnswerWithCitations] | None = None,
) -> AnswerWithCitations:
    a = agent if agent is not None else build_answerer_agent()
    template = _load_prompt_template()
    prompt = template.replace("{question}", question).replace(
        "{evidence_block}", render_evidence(evidence)
    )
    result = await a.run(prompt)
    return result.output
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_answerer_citations.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/agents/answerer.py tests/test_answerer_citations.py
git commit -m "feat(agents): add answerer agent with citation extraction and validation"
```

---

## Phase 6: Activities

### Task 6.1: activities/plan.py — plan_query

**Files:**
- Create: `src/qa_agent/activities/plan.py`

- [x] **Step 1: Implement the activity**

`src/qa_agent/activities/plan.py`:

```python
"""Temporal activity: plan_query."""
from __future__ import annotations
from temporalio import activity
from temporalio.exceptions import ActivityError, ApplicationError
from pydantic import ValidationError
import logfire
from qa_agent.schemas import Plan
from qa_agent.agents.planner import plan_question


@activity.defn(name="plan_query")
async def plan_query(question: str) -> Plan:
    with logfire.span("plan_query", question=question[:200]):
        try:
            return await plan_question(question)
        except ValidationError as e:
            # Schema validation failures aren't worth retrying.
            raise ApplicationError(
                f"planner produced invalid Plan: {e}",
                non_retryable=True,
                type="PlannerValidationError",
            ) from e
```

- [x] **Step 2: Smoke import**

Run: `python -c "from qa_agent.activities.plan import plan_query; print(plan_query.__name__)"`
Expected: prints `plan_query`.

- [x] **Step 3: Commit**

```bash
git add src/qa_agent/activities/plan.py
git commit -m "feat(activities): add plan_query activity wrapping planner agent"
```

---

### Task 6.2: activities/retrieve.py — hybrid_search + naive_hybrid_fallback

**Files:**
- Create: `src/qa_agent/activities/retrieve.py`
- Test: `tests/test_activity_retrieve.py`

- [x] **Step 1: Write a unit test for the de-dup merge logic**

`tests/test_activity_retrieve.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from qa_agent.schemas import SubQuery, Candidate
from qa_agent.activities.retrieve import _merge_legs


def _c(nid: str, *, bm: float | None = None, vec: float | None = None) -> Candidate:
    return Candidate(
        node_id=nid, node_label="Section", indexed_text="x", raw_text="y",
        bm25_score=bm, vector_score=vec,
    )


def test_merge_legs_dedupes_and_combines_scores():
    bm = [_c("a", bm=10.0), _c("b", bm=5.0)]
    vec = [_c("a", vec=0.9), _c("c", vec=0.5)]
    out = _merge_legs(bm, vec)
    by_id = {c.node_id: c for c in out}
    assert by_id["a"].bm25_score == 10.0 and by_id["a"].vector_score == 0.9
    assert by_id["b"].bm25_score == 5.0 and by_id["b"].vector_score is None
    assert by_id["c"].bm25_score is None and by_id["c"].vector_score == 0.5
    assert len(out) == 3
```

- [x] **Step 2: Run, expect failure**

Run: `pytest tests/test_activity_retrieve.py -v`
Expected: FAIL.

- [x] **Step 3: Implement retrieve.py**

`src/qa_agent/activities/retrieve.py`:

```python
"""Temporal activities: hybrid_search and naive_hybrid_fallback."""
from __future__ import annotations
import asyncio
from temporalio import activity
from temporalio.exceptions import ApplicationError
import logfire

from qa_agent.config import get_settings
from qa_agent.schemas import SubQuery, Candidate
from qa_agent.retrieval.bm25 import bm25_search, SEARCHABLE_LABELS
from qa_agent.retrieval.vector import vector_search
from qa_agent.retrieval.fusion import rrf_fuse
from qa_agent.retrieval.rerank import cohere_rerank


def _merge_legs(bm: list[Candidate], vec: list[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[str, str], Candidate] = {}
    for src in (bm, vec):
        for c in src:
            key = (c.node_id, c.node_label)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = c
            else:
                # Combine: take whichever scores are non-None from each side
                by_key[key] = existing.model_copy(update={
                    "bm25_score": existing.bm25_score if existing.bm25_score is not None else c.bm25_score,
                    "vector_score": existing.vector_score if existing.vector_score is not None else c.vector_score,
                })
    return list(by_key.values())


async def _hybrid_for_label(sq: SubQuery, label: str) -> list[Candidate]:
    s = get_settings()
    bm_task = bm25_search(sq, label=label, limit=s.bm25_top_k)
    vec_task = vector_search(sq, label=label, limit=s.vector_top_k)
    try:
        bm, vec = await asyncio.gather(bm_task, vec_task, return_exceptions=False)
    except Exception:
        # Re-raise so Temporal applies retry policy
        raise
    return _merge_legs(bm, vec)


@activity.defn(name="hybrid_search")
async def hybrid_search(sub_query: SubQuery) -> list[Candidate]:
    """Hybrid (BM25 + vector) retrieval for a SubQuery, across all target_labels.

    Returns at most ~100 deduped candidates per target_label, unioned.
    """
    with logfire.span("hybrid_search", sub_query=sub_query.text[:200]):
        labels = [lbl for lbl in sub_query.target_labels if lbl in SEARCHABLE_LABELS]
        if not labels:
            return []
        per_label = await asyncio.gather(*[_hybrid_for_label(sub_query, lbl) for lbl in labels])
        merged: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for clist in per_label:
            for c in clist:
                key = (c.node_id, c.node_label)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(c)
        logfire.info("hybrid_search.done", labels=labels, count=len(merged))
        return merged


@activity.defn(name="naive_hybrid_fallback")
async def naive_hybrid_fallback(question: str) -> list[Candidate]:
    """No planner, no expansion: hybrid sweep across [Section, CodeBlock, TableRow, Callout],
    RRF, then rerank to top 8. Used when the main path produces zero results."""
    with logfire.span("naive_hybrid_fallback", question=question[:200]):
        s = get_settings()
        sq = SubQuery(text=question, target_labels=["Section", "CodeBlock", "TableRow", "Callout"])
        cands = await hybrid_search(sq)
        if not cands:
            return []
        fused = rrf_fuse([cands], k=60, top_n=s.rrf_top_n)
        try:
            return await cohere_rerank(question, fused, top_k=s.rerank_top_k, doc_chars=s.rerank_doc_chars)
        except Exception as e:
            logfire.warn("fallback rerank failed; returning RRF top-K", error=str(e))
            return fused[: s.rerank_top_k]
```

- [x] **Step 4: Run tests**

Run: `pytest tests/test_activity_retrieve.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/qa_agent/activities/retrieve.py tests/test_activity_retrieve.py
git commit -m "feat(activities): add hybrid_search and naive_hybrid_fallback"
```

---

### Task 6.3: activities/expand.py

**Files:**
- Create: `src/qa_agent/activities/expand.py`

- [x] **Step 1: Implement the activity**

`src/qa_agent/activities/expand.py`:

```python
"""Temporal activity: expand_graph."""
from __future__ import annotations
from temporalio import activity
import logfire
from qa_agent.config import get_settings
from qa_agent.schemas import ExpandRequest, Candidate
from qa_agent.retrieval.expansion import expand


@activity.defn(name="expand_graph")
async def expand_graph(req: ExpandRequest) -> list[Candidate]:
    with logfire.span("expand_graph",
                      seed_count=len(req.seeds),
                      patterns=[p.name for p in req.patterns]):
        s = get_settings()
        result = await expand(
            seeds=req.seeds,
            patterns=req.patterns,
            total_cap=s.max_expansion_total,
        )
        logfire.info("expand_graph.done", count=len(result))
        return result
```

- [x] **Step 2: Smoke import**

Run: `python -c "from qa_agent.activities.expand import expand_graph; print(expand_graph.__name__)"`
Expected: prints `expand_graph`.

- [x] **Step 3: Commit**

```bash
git add src/qa_agent/activities/expand.py
git commit -m "feat(activities): add expand_graph activity"
```

---

### Task 6.4: activities/rerank.py

**Files:**
- Create: `src/qa_agent/activities/rerank.py`

- [x] **Step 1: Implement the activity**

`src/qa_agent/activities/rerank.py`:

```python
"""Temporal activity: rerank."""
from __future__ import annotations
from temporalio import activity
from temporalio.exceptions import ApplicationError
import logfire
from qa_agent.config import get_settings
from qa_agent.schemas import RerankRequest, Candidate
from qa_agent.retrieval.rerank import cohere_rerank


@activity.defn(name="rerank")
async def rerank(req: RerankRequest) -> list[Candidate]:
    with logfire.span("rerank", count=len(req.candidates), top_k=req.top_k):
        if not req.candidates:
            return []
        s = get_settings()
        if not s.cohere_api_key or s.cohere_api_key == "changeme":
            logfire.warn("Cohere API key missing; workflow will use RRF fallback")
            raise ApplicationError(
                "Cohere API key is not configured",
                non_retryable=True,
                type="CohereMissingKey",
            )
        try:
            return await cohere_rerank(
                query=req.question,
                candidates=req.candidates,
                top_k=req.top_k,
                doc_chars=s.rerank_doc_chars,
            )
        except Exception as e:
            msg = str(e)
            # Treat 4xx (auth/quota) as non-retryable so the workflow can fall back.
            if any(token in msg for token in ("401", "403", "429", "InvalidApiKey", "Unauthorized")):
                logfire.warn("Cohere 4xx — non-retryable", error=msg)
                raise ApplicationError(
                    f"Cohere rerank refused: {msg}",
                    non_retryable=True,
                    type="CohereClientError",
                ) from e
            raise
```

- [x] **Step 2: Smoke import**

Run: `python -c "from qa_agent.activities.rerank import rerank; print(rerank.__name__)"`
Expected: prints `rerank`.

- [x] **Step 3: Commit**

```bash
git add src/qa_agent/activities/rerank.py
git commit -m "feat(activities): add rerank activity with 4xx non-retryable mapping"
```

---

### Task 6.5: activities/generate.py

**Files:**
- Create: `src/qa_agent/activities/generate.py`

- [x] **Step 1: Implement the activity**

`src/qa_agent/activities/generate.py`:

```python
"""Temporal activity: generate_answer."""
from __future__ import annotations
from temporalio import activity
import logfire
from qa_agent.schemas import GenerateRequest, QAResponse, Citation, Candidate
from qa_agent.agents.answerer import (
    answer_with_citations, extract_citation_ids, validate_citations,
)


_SNIPPET_CHARS = 280


def _candidate_to_citation(idx: int, c: Candidate) -> Citation:
    return Citation(
        id=idx,
        node_id=c.node_id,
        node_label=c.node_label,  # validated against literal set in pydantic
        url=c.url,
        anchor=c.anchor,
        title=c.title or c.breadcrumb or c.node_id,
        snippet=(c.indexed_text or c.raw_text or "")[:_SNIPPET_CHARS],
        score=float(c.rerank_score if c.rerank_score is not None
                    else c.rrf_score if c.rrf_score is not None
                    else 0.0),
    )


@activity.defn(name="generate_answer")
async def generate_answer(req: GenerateRequest) -> QAResponse:
    with logfire.span("generate_answer",
                      evidence_count=len(req.evidence),
                      question=req.question[:200]):
        result = await answer_with_citations(req.question, req.evidence)
        ok, missing, out_of_range = validate_citations(
            result.answer, evidence_count=len(req.evidence), used_ids=result.used_citation_ids,
        )
        if not ok:
            logfire.warn("answer citation mismatch — accepting and trimming",
                         missing=missing, out_of_range=out_of_range)

        used_in_text = extract_citation_ids(result.answer)
        valid_ids = [n for n in used_in_text if 1 <= n <= len(req.evidence)]
        citations = [_candidate_to_citation(n, req.evidence[n - 1]) for n in valid_ids]

        return QAResponse(
            answer=result.answer,
            citations=citations,
            plan=req.plan,
            latency_ms={},   # filled in by workflow / api layer
            fallback_used=False,
            no_results=False,
        )
```

- [x] **Step 2: Smoke import**

Run: `python -c "from qa_agent.activities.generate import generate_answer; print(generate_answer.__name__)"`
Expected: prints `generate_answer`.

- [x] **Step 3: Commit**

```bash
git add src/qa_agent/activities/generate.py
git commit -m "feat(activities): add generate_answer with citation validation"
```

---

## Phase 7: Workflow + replay test

### Task 7.1: workflows/qa.py

**Files:**
- Create: `src/qa_agent/workflows/qa.py`

- [x] **Step 1: Implement the workflow**

`src/qa_agent/workflows/qa.py`:

```python
"""QAWorkflow — orchestrates plan → hybrid retrieval → fusion → expansion →
rerank → generate, with planner-failure and empty-results fallbacks."""
from __future__ import annotations
import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# Workflow code must be deterministic. Imports below are pure / pydantic only.
with workflow.unsafe.imports_passed_through():
    from qa_agent.schemas import (
        QARequest, QAResponse, Plan, ExpandRequest, RerankRequest,
        GenerateRequest, Candidate,
    )
    from qa_agent.retrieval.fusion import rrf_fuse


_DEFAULT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


def _is_non_retryable_application_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, ActivityError)
        and isinstance(exc.cause, ApplicationError)
        and exc.cause.non_retryable
    )


@workflow.defn
class QAWorkflow:
    @workflow.run
    async def run(self, req: QARequest) -> QAResponse:
        # 1. Plan (or fallback)
        plan: Plan
        planner_failed = False
        try:
            plan = await workflow.execute_activity(
                "plan_query",
                req.question,
                result_type=Plan,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
        except ActivityError as e:
            if not _is_non_retryable_application_error(e):
                raise
            planner_failed = True
            plan = Plan.fallback_for(req.question)

        # If planner failed: skip the planned retrieval, run naive fallback.
        if planner_failed:
            top: list[Candidate] = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )
            if not top:
                return QAResponse.empty(plan, fallback_used=True)
            resp = await workflow.execute_activity(
                "generate_answer",
                GenerateRequest(question=req.question, evidence=top, plan=plan),
                result_type=QAResponse,
                start_to_close_timeout=timedelta(seconds=45),
                retry_policy=_DEFAULT_RETRY,
            )
            return resp.model_copy(update={"fallback_used": True})

        # 2. Parallel hybrid retrieval per sub-query
        per_subquery: list[list[Candidate]] = await asyncio.gather(*[
            workflow.execute_activity(
                "hybrid_search",
                sq,
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=_DEFAULT_RETRY,
            )
            for sq in plan.sub_queries
        ])

        # 3. RRF fusion (pure, in-workflow)
        fused = rrf_fuse(per_subquery, k=60, top_n=40)

        # 4. Graph expansion
        expanded = await workflow.execute_activity(
            "expand_graph",
            ExpandRequest(seeds=fused, patterns=plan.expansion_patterns),
            result_type=list[Candidate],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=_DEFAULT_RETRY,
        )

        # 5. Cohere rerank → top 8 (with non-retryable graceful degradation)
        rerank_failed = False
        try:
            top = await workflow.execute_activity(
                "rerank",
                RerankRequest(question=req.question, candidates=expanded, top_k=8),
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=_DEFAULT_RETRY,
            )
        except ActivityError as e:
            if not _is_non_retryable_application_error(e):
                raise
            rerank_failed = True
            # Fall back to RRF top-K
            top = expanded[:8]

        # 6. Empty-result fallback
        fallback_used = rerank_failed
        if not top:
            fallback_used = True
            top = await workflow.execute_activity(
                "naive_hybrid_fallback",
                req.question,
                result_type=list[Candidate],
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=_DEFAULT_RETRY,
            )

        # 7. Generate (or graceful empty)
        if not top:
            return QAResponse.empty(plan, fallback_used=fallback_used)
        resp = await workflow.execute_activity(
            "generate_answer",
            GenerateRequest(question=req.question, evidence=top, plan=plan),
            result_type=QAResponse,
            start_to_close_timeout=timedelta(seconds=45),
            retry_policy=_DEFAULT_RETRY,
        )
        return resp.model_copy(update={"fallback_used": fallback_used})
```

- [x] **Step 2: Smoke import**

Run: `python -c "from qa_agent.workflows.qa import QAWorkflow; print(QAWorkflow.__name__)"`
Expected: prints `QAWorkflow`.

- [x] **Step 3: Commit**

```bash
git add src/qa_agent/workflows/qa.py
git commit -m "feat(workflows): add QAWorkflow with planner-failure and rerank fallbacks"
```

---

### Task 7.2: test_workflow_replay.py

**Files:**
- Test: `tests/test_workflow_replay.py`

- [ ] **Step 1: Add a determinism replay test**

For v1, we use Temporal's `Worker` in a `WorkflowEnvironment` to run the workflow once with mock activities, capture the history, then replay it. The "replay" is what catches non-determinism if someone later edits the workflow body.

`tests/test_workflow_replay.py`:

```python
"""Determinism replay test for QAWorkflow.

We register stub activities, run the workflow once, fetch its history, then
ask Temporal to replay it. Replay reuses the activity *results* recorded in
history; if the workflow code makes any non-deterministic decision (e.g. uses
random/clock outside an activity), replay will fail.
"""
import pytest
from datetime import timedelta
from temporalio import activity
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker, Replayer

from qa_agent.schemas import (
    QARequest, Plan, SubQuery, ExpandRequest, RerankRequest, GenerateRequest,
    Candidate, QAResponse, ExpansionPattern,
)
from qa_agent.workflows.qa import QAWorkflow


def _cand(nid: str, score: float = 1.0) -> Candidate:
    return Candidate(
        node_id=nid, node_label="Section", indexed_text="x", raw_text="y",
        bm25_score=score, vector_score=score, rerank_score=score,
    )


@activity.defn(name="plan_query")
async def stub_plan(question: str) -> Plan:
    return Plan(sub_queries=[SubQuery(text=question, target_labels=["Section"])],
                expansion_patterns=[ExpansionPattern(name="parent_page")])


@activity.defn(name="hybrid_search")
async def stub_hybrid(sq: SubQuery) -> list[Candidate]:
    return [_cand("a", 10.0), _cand("b", 8.0)]


@activity.defn(name="expand_graph")
async def stub_expand(req: ExpandRequest) -> list[Candidate]:
    return req.seeds + [_cand("c", 1.0)]


@activity.defn(name="rerank")
async def stub_rerank(req: RerankRequest) -> list[Candidate]:
    return req.candidates[: req.top_k]


@activity.defn(name="naive_hybrid_fallback")
async def stub_fallback(question: str) -> list[Candidate]:
    return [_cand("z", 0.5)]


@activity.defn(name="generate_answer")
async def stub_generate(req: GenerateRequest) -> QAResponse:
    return QAResponse(
        answer="ok [1].",
        citations=[],
        plan=req.plan,
        latency_ms={},
    )


@pytest.mark.asyncio
async def test_qaworkflow_replays_cleanly():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        async with Worker(
            client,
            task_queue="test-tq",
            workflows=[QAWorkflow],
            activities=[stub_plan, stub_hybrid, stub_expand, stub_rerank,
                        stub_fallback, stub_generate],
        ):
            handle = await client.start_workflow(
                QAWorkflow.run,
                QARequest(question="hello"),
                id="qa-replay-test",
                task_queue="test-tq",
                execution_timeout=timedelta(seconds=30),
            )
            result: QAResponse = await handle.result()
            assert result.answer.startswith("ok")
            history = await handle.fetch_history()

    # Replay against the same workflow definition.
    replayer = Replayer(workflows=[QAWorkflow])
    await replayer.replay_workflow(history)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_workflow_replay.py -v`
Expected: PASS. (If it complains about Temporal time-skipping needing the test server binary, let the SDK auto-fetch it or install the Temporal test server binary using the SDK's documented workflow.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_workflow_replay.py
git commit -m "test(workflow): add replay determinism test for QAWorkflow"
```

---

## Phase 8: Worker & starter

### Task 8.1: worker.py

**Files:**
- Create: `src/qa_agent/worker.py`

- [ ] **Step 1: Implement the worker**

`src/qa_agent/worker.py`:

```python
"""Temporal worker entrypoint for the Q&A agent."""
from __future__ import annotations
import asyncio
import logfire
from temporalio.client import Client
from temporalio.worker import Worker

from qa_agent.config import get_settings
from qa_agent.workflows.qa import QAWorkflow
from qa_agent.activities.plan import plan_query
from qa_agent.activities.retrieve import hybrid_search, naive_hybrid_fallback
from qa_agent.activities.expand import expand_graph
from qa_agent.activities.rerank import rerank
from qa_agent.activities.generate import generate_answer


async def main() -> None:
    s = get_settings()
    logfire.configure(service_name="qa-agent-worker", send_to_logfire=False)

    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    worker = Worker(
        client,
        task_queue=s.qa_task_queue,
        workflows=[QAWorkflow],
        activities=[
            plan_query, hybrid_search, naive_hybrid_fallback,
            expand_graph, rerank, generate_answer,
        ],
    )
    print(f"qa-agent worker listening on task queue {s.qa_task_queue!r} "
          f"({s.temporal_host}, namespace={s.temporal_namespace})")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Smoke check that the worker imports cleanly**

Run: `python -c "import qa_agent.worker"`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/qa_agent/worker.py
git commit -m "feat(worker): add Temporal worker entrypoint"
```

---

### Task 8.2: starter.py — CLI

**Files:**
- Create: `src/qa_agent/starter.py`

- [ ] **Step 1: Implement the CLI**

`src/qa_agent/starter.py`:

```python
"""CLI starter: run a single question through the QAWorkflow."""
from __future__ import annotations
import asyncio
import json
import sys
import uuid
from temporalio.client import Client

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow


async def ask(question: str, debug: bool = False) -> QAResponse:
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    return await client.execute_workflow(
        QAWorkflow.run,
        QARequest(question=question, debug=debug),
        id=f"qa-{uuid.uuid4().hex[:12]}",
        task_queue=s.qa_task_queue,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m qa_agent.starter "<question>"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    resp = asyncio.run(ask(question))
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke check**

Run: `python -c "import qa_agent.starter"`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/qa_agent/starter.py
git commit -m "feat(cli): add starter for one-shot questions via Temporal"
```

---

## Phase 9: Flask API

### Task 9.1: api.py

**Files:**
- Create: `src/qa_agent/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

`tests/test_api.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from qa_agent.api import create_app
from qa_agent.schemas import QAResponse, Plan, SubQuery


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "x"); monkeypatch.setenv("NEO4J_USER", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x"); monkeypatch.setenv("OLLAMA_BASE_URL", "x")
    monkeypatch.setenv("EMBEDDING_MODEL", "x"); monkeypatch.setenv("COHERE_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    from qa_agent.config import get_settings
    get_settings.cache_clear()
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_ask_validates_payload(client):
    r = client.post("/ask", json={})
    assert r.status_code == 400


def test_ask_invokes_workflow(client):
    fake_resp = QAResponse(
        answer="hi [1].",
        citations=[],
        plan=Plan(sub_queries=[SubQuery(text="hi", target_labels=["Section"])]),
    )
    with patch("qa_agent.api.run_workflow", new=AsyncMock(return_value=fake_resp)):
        r = client.post("/ask", json={"question": "hello?"})
    assert r.status_code == 200
    assert r.get_json()["answer"] == "hi [1]."
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement api.py**

`src/qa_agent/api.py`:

```python
"""Flask API: POST /ask, GET /health."""
from __future__ import annotations
import asyncio
import time
import uuid
from flask import Flask, jsonify, request
from pydantic import ValidationError
from temporalio.client import Client

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow


async def run_workflow(req: QARequest) -> QAResponse:
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    return await client.execute_workflow(
        QAWorkflow.run,
        req,
        id=f"qa-{uuid.uuid4().hex[:12]}",
        task_queue=s.qa_task_queue,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        try:
            req = QARequest(**payload)
        except ValidationError as e:
            return jsonify({"error": "invalid request", "details": e.errors()}), 400

        started = time.monotonic()
        try:
            resp = asyncio.run(run_workflow(req))
        except Exception as e:
            return jsonify({"error": "workflow_failure", "detail": str(e)}), 500

        # Stitch the request total into latency_ms (worker fills per-stage)
        total_ms = int((time.monotonic() - started) * 1000)
        latency = dict(resp.latency_ms or {})
        latency["total"] = total_ms
        return jsonify(resp.model_copy(update={"latency_ms": latency}).model_dump())

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/api.py tests/test_api.py
git commit -m "feat(api): add Flask /ask and /health endpoints"
```

---

## Phase 10: Eval harness + pilot

### Task 10.1: fixtures/eval_questions.yaml

**Files:**
- Create: `tests/fixtures/eval_questions.yaml`

- [ ] **Step 1: Add ~7 hand-curated questions tied to expected node IDs**

The exact node IDs must be confirmed against the local graph. Use the placeholder format below; the engineer running this task fills in real IDs by querying the graph (one Cypher query per question). For example:

```cypher
MATCH (s:Section) WHERE s.text CONTAINS 'permission mode' RETURN s.id LIMIT 5;
```

`tests/fixtures/eval_questions.yaml`:

```yaml
# Eval set for the Q&A Agent. Each question lists node ids that SHOULD appear
# in the reranked top-8. Recall@8 ≥ 1 is the assertion.
#
# Replace `expected_ids` placeholders with real ids from your local graph.
# Find them with simple Cypher like:
#   MATCH (s:Section) WHERE s.text CONTAINS 'permission' RETURN s.id LIMIT 10;

- id: q1
  question: "How do I configure the Bash tool's permission mode?"
  expected_ids:
    - "<section-id-for-bash-permissions>"
    - "<settingkey-id-permissionMode>"

- id: q2
  question: "What hook events are available and when do they fire?"
  expected_ids:
    - "<section-id-hook-events>"

- id: q3
  question: "Show me a Python example of using the Edit tool."
  expected_ids:
    - "<codeblock-id-edit-python>"

- id: q4
  question: "Compare bypassPermissions and acceptEdits permission modes."
  expected_ids:
    - "<section-id-permission-modes-table>"

- id: q5
  question: "What's the difference between PreToolUse and PostToolUse hooks?"
  expected_ids:
    - "<section-id-pretooluse>"
    - "<section-id-posttooluse>"

- id: q6
  question: "How do I set the model for a specific session?"
  expected_ids:
    - "<settingkey-id-model>"

- id: q7
  question: "What providers does Claude Code support?"
  expected_ids:
    - "<provider-id-anthropic>"
    - "<provider-id-vertex>"
    - "<provider-id-bedrock>"
```

- [ ] **Step 2: Resolve placeholder IDs against the local graph**

For each entry, run a small Cypher query to find a real node id and replace the placeholder. (The exact ids depend on the upstream pipeline state.)

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/eval_questions.yaml
git commit -m "test(eval): add hand-curated eval question set"
```

---

### Task 10.2: test_retrieval_smoke.py

**Files:**
- Test: `tests/test_retrieval_smoke.py`

- [ ] **Step 1: Add an integration-marked smoke test**

`tests/test_retrieval_smoke.py`:

```python
"""End-to-end smoke test against a real local Neo4j + Ollama + Cohere + Gemini.

Skipped unless `pytest -m integration` is used. Reads tests/fixtures/eval_questions.yaml.
"""
import pytest
import yaml
import uuid
from pathlib import Path
from temporalio.client import Client

from qa_agent.config import get_settings
from qa_agent.schemas import QARequest, QAResponse
from qa_agent.workflows.qa import QAWorkflow


pytestmark = pytest.mark.integration


def _load_eval():
    path = Path(__file__).parent / "fixtures" / "eval_questions.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", _load_eval(), ids=lambda e: e["id"])
async def test_recall_at_8(entry):
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    resp: QAResponse = await client.execute_workflow(
        QAWorkflow.run,
        QARequest(question=entry["question"], debug=True),
        id=f"qa-eval-{entry['id']}-{uuid.uuid4().hex[:6]}",
        task_queue=s.qa_task_queue,
    )

    cited_ids = {c.node_id for c in resp.citations}
    retrieved_ids = {c.node_id for c in (resp.retrieved or [])}
    seen = cited_ids | retrieved_ids
    expected = set(entry["expected_ids"])
    overlap = seen & expected
    assert overlap, (
        f"Q[{entry['id']}] {entry['question']!r}: "
        f"none of expected={expected} found in cited={cited_ids} or retrieved={retrieved_ids}"
    )
```

- [ ] **Step 2: Run with the integration marker (only when worker + services are up)**

Run: `pytest -m integration tests/test_retrieval_smoke.py -v`

Expected when not running services: deselected (PASS, no tests run because marker filter).
Expected when worker is up: most cases PASS; investigate any failures by inspecting `resp.plan` and `resp.retrieved`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_smoke.py
git commit -m "test(smoke): add integration-marked recall@8 smoke against eval set"
```

---

### Task 10.3: Pilot run + report

**Files:** none new — this is operational verification.

- [ ] **Step 1: Start dependencies**

In separate terminals:

```bash
# Terminal 1
temporal server start-dev

# Terminal 2
ollama serve
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0   # or whichever tag matches 1024-dim

# Terminal 3
neo4j console      # or your existing Neo4j launcher
```

- [ ] **Step 2: Verify env**

```bash
python -c "from qa_agent.config import get_settings; s = get_settings(); print('ok', s.embedding_model, s.cohere_rerank_model, s.gemini_model)"
```

Expected: prints `ok` plus the three model names. If it raises, fix the missing env var.

- [ ] **Step 3: Run a query end-to-end via CLI**

```bash
python -m qa_agent.worker          # Terminal 4
python -m qa_agent.starter "How do I configure the Bash tool's permission mode?"   # Terminal 5
```

Expected: JSON response with `answer`, `citations` (≥1), and `latency_ms` populated. `fallback_used: false`, `no_results: false`.

- [ ] **Step 4: Run a question that should hit the empty path**

```bash
python -m qa_agent.starter "What is the airspeed velocity of an unladen swallow?"
```

Expected: `no_results: true`, friendly message, `citations: []`.

- [ ] **Step 5: Run a question that should exercise CodeBlock retrieval**

```bash
python -m qa_agent.starter "Show me a Python example of using the Edit tool."
```

Expected: at least one citation whose `node_label` is `CodeBlock`.

- [ ] **Step 6: Inspect Temporal UI for the three runs**

Open `http://localhost:8233`. Confirm:
- Each `QAWorkflow` execution completed.
- Activity timings are reasonable (no single activity > 10s under normal load).
- No retries on the happy path; retries on transient Cohere/Gemini failures resolved within attempts.

- [ ] **Step 7: Commit any tuning changes from observation**

If the pilot reveals an obvious tuning gap (e.g. `RERANK_TOP_K=8` is too small / too large for the kind of answers Gemini produces), adjust `.env` and/or `config.py` defaults and commit. If not, no commit needed.

```bash
# only if changes made
git add .env.example src/qa_agent/config.py
git commit -m "tune: adjust retrieval knobs based on pilot results"
```

---

## End of plan

The agent is now end-to-end runnable. Next steps beyond v1 are tracked in spec §2 (non-goals) and §14 (open items / risks).
