# Contextual Embeddings + BM25 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `context`, `indexed_text`, and `embedding` on every non-Page node in the Neo4j graph using a Temporal + PydanticAI (Claude) + Ollama pipeline.

**Architecture:** A Temporal parent workflow fans out one child `PerTypeWorkflow` per batch-of-50-IDs per node label. Each child fetches payloads from Neo4j, routes them (skip / template / LLM), generates context via PydanticAI + Claude Sonnet, batch-embeds with Ollama `qwen3-embedding:0.6b`, and writes back via parameterized Cypher.

**Tech Stack:** Python 3.11, temporalio, pydantic-ai[anthropic], neo4j, openai (Ollama-compat client), tenacity, pydantic-settings, pytest, pytest-asyncio

---

## Confirmed Configuration

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | already in .env (fix typo — see Task 1) |
| `LLM_MODEL` | `claude-sonnet-4-6` |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` (dim=1024) |
| `TEMPORAL_HOST` | `localhost:7233` |
| `TEMPORAL_NAMESPACE` | `default` |
| Pilot scope | 50 nodes per label before full run |
| Existing context/embedding | None — full overwrite safe |

---

## File Map

```
contextual_pipeline/
├── pyproject.toml
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings BaseSettings
│   ├── schemas.py                   # NodePayload + ContextResult
│   ├── routing.py                   # decide_route, build_raw_text_repr, templaters
│   ├── neo4j_client.py              # driver singleton, load_cypher(), run_query()
│   ├── cypher/
│   │   ├── fetch_ids/
│   │   │   ├── sections.cypher
│   │   │   ├── codeblocks.cypher
│   │   │   ├── tablerows.cypher
│   │   │   ├── callouts.cypher
│   │   │   └── entities.cypher      # Tool/Hook/SettingKey/PermissionMode/MessageType/Provider
│   │   ├── fetch_nodes/
│   │   │   ├── sections.cypher
│   │   │   ├── codeblocks.cypher
│   │   │   ├── tablerows.cypher
│   │   │   ├── callouts.cypher
│   │   │   └── entities.cypher
│   │   └── write_context.cypher
│   ├── prompts/
│   │   └── context_generation.txt
│   ├── agents/
│   │   └── context_agent.py         # PydanticAI Agent with Claude
│   ├── activities/
│   │   ├── __init__.py
│   │   ├── fetch.py                 # fetch_node_ids, fetch_nodes_batch
│   │   ├── generate_context.py      # generate_context (single LLM call)
│   │   ├── embed.py                 # embed_batch
│   │   └── write.py                 # write_results
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── per_type.py              # PerTypeWorkflow
│   │   └── parent.py                # ContextualizationWorkflow
│   ├── worker.py
│   └── starter.py
└── tests/
    ├── conftest.py
    ├── test_routing.py              # 25+ parametrized cases
    ├── test_templater.py            # determinism + content checks
    ├── test_cypher_safety.py        # no f-string Cypher
    ├── test_pydantic_validation.py  # ContextResult boundary cases
    └── test_indexed_text_format.py  # exact indexed_text format
```

**Primary key per label** (used in fetch IDs and write_context routing):
- `Section`, `CodeBlock`, `TableRow`, `Callout` → `n.id`
- `Tool`, `Hook`, `PermissionMode`, `MessageType`, `Provider` → `n.name`
- `SettingKey` → `n.key`
- `Page` → `n.url` (skipped — not processed)

---

## Task 1: Fix .env and verify services

**Files:**
- Modify: `.env`
- Create: `.env.example`

- [ ] **Step 1: Fix the Anthropic API key variable name typo and add missing vars**

Edit `.env` to contain exactly:
```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=milos1234
ANTHROPIC_API_KEY=sk-ant-api03-...   # keep existing key value
LLM_MODEL=claude-sonnet-4-6
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=qwen3-embedding:0.6b
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
CONTEXT_VERSION=1
```
(The old key `Antropic_API_KEY` was missing the `h` — pydantic-settings maps env var names case-insensitively but `ANTHROPIC_API_KEY` is the standard name that pydantic-ai expects.)

- [ ] **Step 2: Create .env.example**

Create `contextual_pipeline/.env.example`:
```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
ANTHROPIC_API_KEY=
LLM_MODEL=claude-sonnet-4-6
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=qwen3-embedding:0.6b
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
CONTEXT_VERSION=1
```

- [ ] **Step 3: Verify Ollama is running and model is available**

```bash
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","input":["hello world"]}'
```
Expected: JSON with `data[0].embedding` array of length 1024.

If model not downloaded: `ollama pull qwen3-embedding:0.6b`

- [ ] **Step 4: Verify Neo4j index dimension**

In Neo4j Browser or cypher-shell, run:
```cypher
SHOW INDEXES YIELD name, type, options
WHERE type = 'VECTOR'
RETURN name, options.indexConfig.`vector.dimensions` AS dim
```
Expected: at least one index with `dim = 1024`. If dim is different, stop and ask the user — the embedding model must match the index dimension.

- [ ] **Step 5: Verify Temporal is running**

```bash
temporal workflow list --namespace default
```
Expected: empty list or existing workflows — no connection error.

---

## Task 2: Project skeleton and dependencies

**Files:**
- Create: `contextual_pipeline/pyproject.toml`
- Create: `contextual_pipeline/src/__init__.py`
- Create: `contextual_pipeline/src/activities/__init__.py`
- Create: `contextual_pipeline/src/workflows/__init__.py`
- Create: `contextual_pipeline/tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

Create `contextual_pipeline/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "contextual-pipeline"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "temporalio>=1.7.0",
    "pydantic-ai[anthropic]>=0.0.14",
    "neo4j>=5.20.0",
    "openai>=1.30.0",
    "tenacity>=8.3.0",
    "pydantic-settings>=2.3.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Install dependencies**

```bash
cd contextual_pipeline
pip install -e ".[dev]"
```

Expected output: `Successfully installed contextual-pipeline-0.1.0 ...`

- [ ] **Step 3: Create empty __init__ files**

Create `contextual_pipeline/src/__init__.py` (empty).
Create `contextual_pipeline/src/activities/__init__.py` (empty).
Create `contextual_pipeline/src/workflows/__init__.py` (empty).

- [ ] **Step 4: Create tests/conftest.py**

Create `contextual_pipeline/tests/conftest.py`:
```python
import pytest
from src.schemas import NodePayload


def make_section(
    raw_text: str = "Some section text",
    breadcrumb: str = "Page > Section",
    synthesized: str = "false",
    node_id: str = "s1",
) -> NodePayload:
    return NodePayload(
        node_id=node_id,
        label="Section",
        raw_text=raw_text,
        breadcrumb=breadcrumb,
        extras={"synthesized": synthesized},
    )


def make_codeblock(
    raw_text: str = "x = 1",
    preceding: str = "",
    language: str = "python",
    node_id: str = "c1",
) -> NodePayload:
    return NodePayload(
        node_id=node_id,
        label="CodeBlock",
        raw_text=raw_text,
        breadcrumb="Page > Section",
        extras={"preceding": preceding, "language": language},
    )
```

- [ ] **Step 5: Commit skeleton**

```bash
cd contextual_pipeline
git init
git add pyproject.toml .env.example src/ tests/
git commit -m "chore: project skeleton, pyproject.toml, empty inits"
```

---

## Task 3: Pydantic schemas (TDD)

**Files:**
- Create: `contextual_pipeline/tests/test_pydantic_validation.py`
- Create: `contextual_pipeline/src/schemas.py`

- [ ] **Step 1: Write the failing tests**

Create `contextual_pipeline/tests/test_pydantic_validation.py`:
```python
import pytest
from pydantic import ValidationError
from src.schemas import ContextResult, NodePayload


def test_context_result_valid():
    r = ContextResult(context="This section explains tool permissions in Claude Code.")
    assert r.context.startswith("This section")


def test_context_result_too_short():
    with pytest.raises(ValidationError):
        ContextResult(context="Too short")


def test_context_result_empty():
    with pytest.raises(ValidationError):
        ContextResult(context="")


def test_context_result_too_long():
    with pytest.raises(ValidationError):
        ContextResult(context="A" * 601)


def test_context_result_exactly_600():
    r = ContextResult(context="A" * 600)
    assert len(r.context) == 600


def test_context_result_markdown_header_is_valid():
    # Markdown content is NOT blocked by schema — routing handles quality
    r = ContextResult(context="## This is a context\nWith some detail here.")
    assert r.context.startswith("##")


def test_node_payload_defaults():
    p = NodePayload(node_id="x", label="Section", raw_text="hello")
    assert p.tags == []
    assert p.related_hints == []
    assert p.extras == {}


def test_node_payload_extras_stored():
    p = NodePayload(
        node_id="x", label="CodeBlock", raw_text="code",
        extras={"language": "python", "preceding": "Some text"}
    )
    assert p.extras["language"] == "python"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd contextual_pipeline
pytest tests/test_pydantic_validation.py -v
```
Expected: `ERROR` — `src.schemas` module not found.

- [ ] **Step 3: Implement schemas.py**

Create `contextual_pipeline/src/schemas.py`:
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional


class NodePayload(BaseModel):
    node_id: str
    label: Literal[
        "Section", "CodeBlock", "TableRow", "Callout",
        "Tool", "Hook", "SettingKey", "PermissionMode",
        "MessageType", "Provider"
    ]
    raw_text: str
    page_title: Optional[str] = None
    page_description: Optional[str] = None
    breadcrumb: Optional[str] = None
    parent_text: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    related_hints: list[str] = Field(default_factory=list)
    extras: dict[str, str] = Field(default_factory=dict)


class ContextResult(BaseModel):
    context: str = Field(
        min_length=10,
        max_length=600,
        description="1-3 sentence retrieval context",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pydantic_validation.py -v
```
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/schemas.py tests/test_pydantic_validation.py
git commit -m "feat: NodePayload and ContextResult schemas with validation tests"
```

---

## Task 4: Config

**Files:**
- Create: `contextual_pipeline/src/config.py`

- [ ] **Step 1: Create config.py**

Create `contextual_pipeline/src/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",   # .env lives one level above contextual_pipeline/
        extra="ignore",
    )

    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-6"

    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_batch_size: int = 32

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"

    context_version: int = 1
    llm_batch_size: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2: Verify config loads**

```bash
cd contextual_pipeline
python -c "from src.config import get_settings; s = get_settings(); print(s.llm_model, s.embedding_model)"
```
Expected: `claude-sonnet-4-6 qwen3-embedding:0.6b`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: pydantic-settings config with Anthropic + Ollama vars"
```

---

## Task 5: Routing logic — decide_route (TDD)

**Files:**
- Create: `contextual_pipeline/tests/test_routing.py`
- Create: `contextual_pipeline/src/routing.py` (partial — decide_route only)

- [ ] **Step 1: Write failing tests for decide_route**

Create `contextual_pipeline/tests/test_routing.py`:
```python
import pytest
from src.routing import decide_route
from src.schemas import NodePayload


def section(raw_text, breadcrumb="Page > Sec", synthesized="false", node_id="s1"):
    return NodePayload(
        node_id=node_id, label="Section", raw_text=raw_text,
        breadcrumb=breadcrumb, extras={"synthesized": synthesized},
    )


def codeblock(raw_text, preceding="", node_id="c1"):
    return NodePayload(
        node_id=node_id, label="CodeBlock", raw_text=raw_text,
        breadcrumb="Page > Sec", extras={"preceding": preceding, "language": "python"},
    )


def ref_entity(label, node_id="e1"):
    return NodePayload(node_id=node_id, label=label, raw_text="desc",
                       extras={"name": node_id})


@pytest.mark.parametrize("payload,expected", [
    # --- Section: skip cases ---
    (section(""), "skip"),
    (section("   "), "skip"),
    (section("---"), "skip"),
    (section(None or ""), "skip"),                          # None-ish
    # --- Section: template cases ---
    (section("x", synthesized="true"), "template"),
    (section("A" * 201, "Page > A > B"), "template"),       # deep breadcrumb + long
    (section("A" * 201, synthesized="true"), "template"),   # synthesized wins even if long+deep
    # --- Section: llm cases ---
    (section("short text"), "llm"),                         # short, no synthesized
    (section("A" * 201, "Page > Sec"), "llm"),              # long but shallow breadcrumb (depth=2)
    (section("A" * 200, "Page > A > B"), "llm"),            # exactly 200 chars → not > 200
    (section("A" * 201, "Page > A"), "llm"),                # long, depth=2 → no template
    # --- CodeBlock: template cases ---
    (codeblock("X" * 1001, "P" * 201), "template"),         # long code + long preceding
    # --- CodeBlock: llm cases (trivial labels treated as empty preceding) ---
    (codeblock("X" * 1001, "Input:"), "llm"),
    (codeblock("X" * 1001, "Output:"), "llm"),
    (codeblock("X" * 1001, "Example:"), "llm"),
    (codeblock("X" * 1001, "Migration:"), "llm"),
    (codeblock("X" * 1001, "Windows PowerShell"), "llm"),
    (codeblock("X" * 1001, "Windows CMD"), "llm"),
    (codeblock("X" * 1001, "macOS"), "llm"),
    (codeblock("X" * 1001, "macOS, Linux, WSL"), "llm"),
    (codeblock("X" * 1001, "In code"), "llm"),
    (codeblock("X" * 1001, ".mcp.json"), "llm"),
    (codeblock("X" * 1001, "P" * 200), "llm"),              # preceding exactly 200 → not > 200
    (codeblock("X" * 1000, "P" * 201), "llm"),              # code exactly 1000 → not > 1000
    (codeblock("X" * 1001, ""), "llm"),                     # no preceding
    # --- Reference entities: always llm ---
    (ref_entity("TableRow"), "llm"),
    (ref_entity("Callout"), "llm"),
    (ref_entity("Tool"), "llm"),
    (ref_entity("Hook"), "llm"),
    (ref_entity("SettingKey"), "llm"),
    (ref_entity("PermissionMode"), "llm"),
    (ref_entity("MessageType"), "llm"),
    (ref_entity("Provider"), "llm"),
])
def test_decide_route(payload, expected):
    assert decide_route(payload) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_routing.py -v
```
Expected: `ERROR` — `src.routing` not found.

- [ ] **Step 3: Implement decide_route in routing.py**

Create `contextual_pipeline/src/routing.py`:
```python
from typing import Literal
from .schemas import NodePayload

TRIVIAL_LABELS = {
    "Input:", "Output:", "Example:", "Migration:",
    "Windows PowerShell", "Windows CMD", "macOS",
    "macOS, Linux, WSL", "In code", ".mcp.json",
}


def decide_route(payload: NodePayload) -> Literal["skip", "template", "llm"]:
    if payload.label == "Section":
        if not payload.raw_text or payload.raw_text.strip() in {"", "---"}:
            return "skip"
        if payload.extras.get("synthesized") == "true":
            return "template"
        depth = len((payload.breadcrumb or "").split(" > "))
        if len(payload.raw_text) > 200 and depth >= 3:
            return "template"
        return "llm"

    if payload.label == "CodeBlock":
        pp = payload.extras.get("preceding", "") or ""
        if pp in TRIVIAL_LABELS:
            pp = ""
        if len(payload.raw_text) > 1000 and len(pp) > 200:
            return "template"
        return "llm"

    if payload.label in {
        "TableRow", "Callout", "Tool", "Hook",
        "SettingKey", "PermissionMode", "MessageType", "Provider",
    }:
        return "llm"

    return "skip"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_routing.py -v
```
Expected: `31 passed` (or however many parametrize cases there are).

- [ ] **Step 5: Commit**

```bash
git add src/routing.py tests/test_routing.py
git commit -m "feat: decide_route with 30+ test cases covering all routing rules"
```

---

## Task 6: Routing — templaters and raw_text_repr (TDD)

**Files:**
- Create: `contextual_pipeline/tests/test_templater.py`
- Create: `contextual_pipeline/tests/test_indexed_text_format.py`
- Modify: `contextual_pipeline/src/routing.py` (add templater + repr functions)

- [ ] **Step 1: Write failing tests for templaters**

Create `contextual_pipeline/tests/test_templater.py`:
```python
from src.routing import template_section, template_codeblock, build_raw_text_repr
from src.schemas import NodePayload


def _section(extras=None, **kw):
    return NodePayload(
        node_id="s1", label="Section",
        raw_text=kw.get("raw_text", "Some text"),
        page_title=kw.get("page_title", "My Page"),
        breadcrumb=kw.get("breadcrumb", "My Page > Intro > Detail"),
        parent_text=kw.get("parent_text", "Parent paragraph."),
        extras=extras or {},
    )


def test_template_section_synthesized_contains_changelog():
    p = _section(extras={"synthesized": "true"},
                 breadcrumb="Changelog > 2.1.122")
    result = template_section(p)
    assert "changelog entry" in result
    assert "2.1.122" in result


def test_template_section_deterministic():
    p = _section(breadcrumb="My Page > Intro > Detail", raw_text="A" * 250)
    assert template_section(p) == template_section(p)


def test_template_section_contains_breadcrumb_tail():
    p = _section(breadcrumb="My Page > Intro > Detail")
    result = template_section(p)
    assert "Detail" in result


def test_template_section_contains_parent_breadcrumb():
    p = _section(breadcrumb="My Page > Intro > Detail")
    result = template_section(p)
    assert "My Page > Intro" in result


def test_template_codeblock_contains_language():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": "Install dependencies first."},
    )
    result = template_codeblock(p)
    assert "python" in result


def test_template_codeblock_deterministic():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": "Some preceding text."},
    )
    assert template_codeblock(p) == template_codeblock(p)


def test_template_codeblock_contains_breadcrumb():
    p = NodePayload(
        node_id="c1", label="CodeBlock", raw_text="x=1",
        breadcrumb="Page > Sec",
        extras={"language": "python", "preceding": ""},
    )
    result = template_codeblock(p)
    assert "Page > Sec" in result


# --- build_raw_text_repr ---

def test_repr_section():
    p = NodePayload(node_id="s1", label="Section", raw_text="Hello world")
    assert build_raw_text_repr(p) == "Hello world"


def test_repr_codeblock():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="x=1",
                    extras={"language": "python"})
    assert build_raw_text_repr(p) == "[python]\nx=1"


def test_repr_codeblock_no_language_defaults_code():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="x=1", extras={})
    assert build_raw_text_repr(p).startswith("[code]")


def test_repr_tablerow():
    p = NodePayload(node_id="t1", label="TableRow", raw_text="",
                    extras={"headers": "Name|Value", "cells": "foo|bar"})
    result = build_raw_text_repr(p)
    assert "Name | Value" in result
    assert "foo | bar" in result


def test_repr_callout():
    p = NodePayload(node_id="cal1", label="Callout", raw_text="Be careful here.",
                    extras={"callout_kind": "warning"})
    result = build_raw_text_repr(p)
    assert "[warning]" in result
    assert "Be careful here." in result


def test_repr_tool():
    p = NodePayload(node_id="Bash", label="Tool", raw_text="Runs shell commands.",
                    extras={"name": "Bash"})
    result = build_raw_text_repr(p)
    assert "Bash: Runs shell commands." == result
```

- [ ] **Step 2: Write failing test for indexed_text format**

Create `contextual_pipeline/tests/test_indexed_text_format.py`:
```python
from src.routing import build_raw_text_repr
from src.schemas import NodePayload


def build_indexed_text(context: str, raw_text_repr: str) -> str:
    return context + "\n\n" + raw_text_repr


def test_indexed_text_exact_format_section():
    p = NodePayload(node_id="s1", label="Section", raw_text="The text content.")
    ctx = "This section explains tool permissions."
    repr_ = build_raw_text_repr(p)
    result = build_indexed_text(ctx, repr_)
    assert result == "This section explains tool permissions.\n\nThe text content."


def test_indexed_text_exact_format_codeblock():
    p = NodePayload(node_id="c1", label="CodeBlock", raw_text="npm install",
                    extras={"language": "bash"})
    ctx = "Install command for the project."
    repr_ = build_raw_text_repr(p)
    result = build_indexed_text(ctx, repr_)
    assert result == "Install command for the project.\n\n[bash]\nnpm install"


def test_indexed_text_separator_is_double_newline():
    p = NodePayload(node_id="s1", label="Section", raw_text="content")
    result = build_indexed_text("context", build_raw_text_repr(p))
    assert "\n\n" in result
    parts = result.split("\n\n", 1)
    assert parts[0] == "context"
    assert parts[1] == "content"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_templater.py tests/test_indexed_text_format.py -v
```
Expected: `ERROR` — templater functions not found in `src.routing`.

- [ ] **Step 4: Add templater functions and build_raw_text_repr to routing.py**

Append to `contextual_pipeline/src/routing.py`:
```python
def build_raw_text_repr(payload: NodePayload) -> str:
    if payload.label == "Section":
        return payload.raw_text

    if payload.label == "CodeBlock":
        lang = payload.extras.get("language", "code")
        return f"[{lang}]\n{payload.raw_text}"

    if payload.label == "TableRow":
        headers = payload.extras.get("headers", "").split("|")
        cells = payload.extras.get("cells", "").split("|")
        return f"{' | '.join(headers)}\n{' | '.join(cells)}"

    if payload.label == "Callout":
        kind = payload.extras.get("callout_kind", "note")
        return f"[{kind}] {payload.raw_text}"

    # Tool, Hook, SettingKey, PermissionMode, MessageType, Provider
    name = payload.extras.get("name", payload.node_id)
    return f"{name}: {payload.raw_text}"


def template_section(p: NodePayload) -> str:
    crumb = p.breadcrumb or ""
    if p.extras.get("synthesized") == "true":
        tail = crumb.split(" > ")[-1]
        return f"[changelog entry] Release notes for {tail}."
    tail = crumb.split(" > ")[-1]
    parent_crumb = crumb.rsplit(" > ", 1)[0] if " > " in crumb else crumb
    parent = (p.parent_text or "")[:120].strip()
    base = f"[{p.page_title} — {tail}] Subsection of \"{parent_crumb}\"."
    return f"{base} {parent}".strip()


def template_codeblock(p: NodePayload) -> str:
    lang = p.extras.get("language", "code")
    pp = (p.extras.get("preceding", "") or "")[:200]
    return f"[{lang}: example] In {p.breadcrumb}. {pp}".strip()


def templater_for(label: str):
    if label == "Section":
        return template_section
    if label == "CodeBlock":
        return template_codeblock
    raise ValueError(f"No template for label: {label}")
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_routing.py tests/test_templater.py tests/test_indexed_text_format.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/routing.py tests/test_templater.py tests/test_indexed_text_format.py
git commit -m "feat: build_raw_text_repr, templaters, indexed_text format tests"
```

---

## Task 7: Cypher files

**Files:** All `.cypher` files under `contextual_pipeline/src/cypher/`

- [ ] **Step 1: Create cypher/fetch_ids/ files**

Create `contextual_pipeline/src/cypher/fetch_ids/sections.cypher`:
```cypher
MATCH (n:Section)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.id AS node_id
```

Create `contextual_pipeline/src/cypher/fetch_ids/codeblocks.cypher`:
```cypher
MATCH (n:CodeBlock)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.id AS node_id
```

Create `contextual_pipeline/src/cypher/fetch_ids/tablerows.cypher`:
```cypher
MATCH (n:TableRow)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.id AS node_id
```

Create `contextual_pipeline/src/cypher/fetch_ids/callouts.cypher`:
```cypher
MATCH (n:Callout)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.id AS node_id
```

Create `contextual_pipeline/src/cypher/fetch_ids/entities.cypher` (handles Tool, Hook, SettingKey, PermissionMode, MessageType, Provider — caller picks right file via label→file map, but entities share this pattern; individual files below are needed):

Actually create one file per entity label:

`contextual_pipeline/src/cypher/fetch_ids/tools.cypher`:
```cypher
MATCH (n:Tool)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
```

`contextual_pipeline/src/cypher/fetch_ids/hooks.cypher`:
```cypher
MATCH (n:Hook)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
```

`contextual_pipeline/src/cypher/fetch_ids/settingkeys.cypher`:
```cypher
MATCH (n:SettingKey)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.key AS node_id
```

`contextual_pipeline/src/cypher/fetch_ids/permissionmodes.cypher`:
```cypher
MATCH (n:PermissionMode)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
```

`contextual_pipeline/src/cypher/fetch_ids/messagetypes.cypher`:
```cypher
MATCH (n:MessageType)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
```

`contextual_pipeline/src/cypher/fetch_ids/providers.cypher`:
```cypher
MATCH (n:Provider)
WHERE CASE $resume_mode
    WHEN 'missing_only' THEN n.context IS NULL
    WHEN 'by_source' THEN n.context_source IN $target_sources
    ELSE true
  END
RETURN n.name AS node_id
```

- [ ] **Step 2: Create cypher/fetch_nodes/ files**

Create `contextual_pipeline/src/cypher/fetch_nodes/sections.cypher`:
```cypher
MATCH (s:Section)
WHERE s.id IN $ids
OPTIONAL MATCH (parent:Section)-[:HAS_SUBSECTION]->(s)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(root:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH s, parent, collect(DISTINCT p)[0] AS p
RETURN
  s.id            AS node_id,
  s.text          AS raw_text,
  s.breadcrumb    AS breadcrumb,
  toString(coalesce(s.synthesized, false)) AS synthesized,
  parent.text     AS parent_text,
  p.title         AS page_title,
  p.description   AS page_description
```

Create `contextual_pipeline/src/cypher/fetch_nodes/codeblocks.cypher`:
```cypher
MATCH (c:CodeBlock)
WHERE c.id IN $ids
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH c, s, collect(DISTINCT p)[0] AS p
RETURN
  c.id                    AS node_id,
  c.text                  AS raw_text,
  c.language              AS language,
  c.preceding_paragraph   AS preceding,
  s.breadcrumb            AS breadcrumb,
  s.text                  AS parent_text,
  p.title                 AS page_title,
  p.description           AS page_description
```

Create `contextual_pipeline/src/cypher/fetch_nodes/tablerows.cypher`:
```cypher
MATCH (t:TableRow)
WHERE t.id IN $ids
OPTIONAL MATCH (tbl)-[:HAS_ROW]->(t)
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE]->(tbl)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH t, s, collect(DISTINCT p)[0] AS p
RETURN
  t.id          AS node_id,
  t.text        AS raw_text,
  t.headers     AS headers,
  t.cells       AS cells,
  s.breadcrumb  AS breadcrumb,
  s.text        AS parent_text,
  p.title       AS page_title,
  p.description AS page_description
```

Create `contextual_pipeline/src/cypher/fetch_nodes/callouts.cypher`:
```cypher
MATCH (c:Callout)
WHERE c.id IN $ids
OPTIONAL MATCH (s:Section)-[:CONTAINS_CALLOUT]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH c, s, collect(DISTINCT p)[0] AS p
RETURN
  c.id          AS node_id,
  c.text        AS raw_text,
  c.kind        AS callout_kind,
  s.breadcrumb  AS breadcrumb,
  p.title       AS page_title,
  p.description AS page_description
```

Create `contextual_pipeline/src/cypher/fetch_nodes/entities.cypher`:
```cypher
MATCH (n)
WHERE (n:Tool OR n:Hook OR n:PermissionMode OR n:MessageType OR n:Provider)
  AND n.name IN $ids
RETURN
  n.name        AS node_id,
  n.description AS raw_text,
  n.name        AS name,
  labels(n)[0]  AS label
UNION ALL
MATCH (n:SettingKey)
WHERE n.key IN $ids
RETURN
  n.key         AS node_id,
  n.description AS raw_text,
  n.key         AS name,
  'SettingKey'  AS label
```

- [ ] **Step 3: Create write_context.cypher**

Create `contextual_pipeline/src/cypher/write_context.cypher`:
```cypher
UNWIND $rows AS row
MATCH (n)
WHERE n.id = row.node_id
   OR n.url = row.node_id
   OR n.name = row.node_id
   OR n.key = row.node_id
SET
  n.context         = row.context,
  n.indexed_text    = row.indexed_text,
  n.embedding       = row.embedding,
  n.context_source  = row.context_source,
  n.context_version = row.context_version
RETURN count(n) AS updated
```

- [ ] **Step 4: Commit**

```bash
git add src/cypher/
git commit -m "feat: parameterized Cypher templates for all node types"
```

---

## Task 8: Neo4j client + Cypher safety test (TDD)

**Files:**
- Create: `contextual_pipeline/tests/test_cypher_safety.py`
- Create: `contextual_pipeline/src/neo4j_client.py`

- [ ] **Step 1: Write failing Cypher safety test**

Create `contextual_pipeline/tests/test_cypher_safety.py`:
```python
import subprocess
import pytest


PATTERNS = [
    r'f"MATCH',
    r"f'MATCH",
    r'f"CREATE',
    r"f'CREATE",
    r'f"MERGE',
    r"f'MERGE",
    r'f"SET',
    r"f'SET",
    r'" MATCH',
    r"' MATCH",
    r'" CREATE',
    r"' CREATE",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_fstring_cypher(pattern):
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", pattern, "src/"],
        capture_output=True, text=True,
    )
    assert result.stdout == "", (
        f"Found dangerous inline Cypher pattern '{pattern}':\n{result.stdout}"
    )
```

- [ ] **Step 2: Run test — verify it passes on current code (no violations yet)**

```bash
pytest tests/test_cypher_safety.py -v
```
Expected: all pass (no src/ Python files yet contain Cypher).

- [ ] **Step 3: Implement neo4j_client.py**

Create `contextual_pipeline/src/neo4j_client.py`:
```python
from pathlib import Path
from functools import lru_cache
from neo4j import GraphDatabase, AsyncGraphDatabase
from .config import get_settings

_CYPHER_DIR = Path(__file__).parent / "cypher"


def load_cypher(relative_path: str) -> str:
    return (_CYPHER_DIR / relative_path).read_text(encoding="utf-8")


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
```

- [ ] **Step 4: Re-run safety test to confirm neo4j_client.py has no violations**

```bash
pytest tests/test_cypher_safety.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/neo4j_client.py tests/test_cypher_safety.py
git commit -m "feat: neo4j_client with load_cypher helper; Cypher injection safety tests"
```

---

## Task 9: Prompt template + PydanticAI agent

**Files:**
- Create: `contextual_pipeline/src/prompts/context_generation.txt`
- Create: `contextual_pipeline/src/agents/context_agent.py`

- [ ] **Step 1: Create the prompt template**

Create `contextual_pipeline/src/prompts/context_generation.txt`:
```
You are generating a 1–3 sentence retrieval context for a node in the
Claude Code documentation knowledge graph. The context will be
prepended to the node's text and embedded for hybrid (vector + BM25)
search.

Node type: {node_type}
Node tags: {node_tags}
Page: {page_title} — {page_description}
Breadcrumb: {breadcrumb}
Parent section text: {parent_text}
Related hints: {related_hints}

Node content:
---
{raw_text}
---

Write context that:
1. Names what the node is (a tool definition? a step in a workflow? a
   schema row? a security warning?).
2. Says where in the documentation it lives, only when ambiguous.
3. Disambiguates from any near-identical content elsewhere — the
   "Related hints" field flags these.
4. Does NOT restate the node's content. Context is metadata, not
   summary.
5. Is dense: every word should help retrieval. No hedging phrases like
   "this section discusses".

Output 1–3 sentences. No preamble. No markdown.
```

- [ ] **Step 2: Create context_agent.py**

Create `contextual_pipeline/src/agents/context_agent.py`:
```python
from pathlib import Path
from pydantic_ai import Agent
from ..schemas import NodePayload, ContextResult
from ..config import get_settings

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "context_generation.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_settings = get_settings()

agent = Agent(
    f"anthropic:{_settings.llm_model}",
    result_type=ContextResult,
)


def _build_user_prompt(payload: NodePayload) -> str:
    return _PROMPT_TEMPLATE.format(
        node_type=payload.label,
        node_tags=", ".join(payload.tags) if payload.tags else "none",
        page_title=payload.page_title or "unknown",
        page_description=payload.page_description or "",
        breadcrumb=payload.breadcrumb or "unknown",
        parent_text=(payload.parent_text or "")[:300],
        related_hints=", ".join(payload.related_hints) if payload.related_hints else "none",
        raw_text=payload.raw_text[:2000],
    )


async def generate_context(payload: NodePayload) -> ContextResult:
    user_prompt = _build_user_prompt(payload)
    result = await agent.run(user_prompt)
    return result.data
```

- [ ] **Step 3: Create a manual smoke-test script**

Create `contextual_pipeline/scripts/test_agent_smoke.py`:
```python
"""
Run with: python scripts/test_agent_smoke.py
Tests the PydanticAI agent against 5 hand-picked NodePayload fixtures.
"""
import asyncio
from src.schemas import NodePayload
from src.agents.context_agent import generate_context

FIXTURES = [
    NodePayload(
        node_id="tool_bash",
        label="Tool",
        raw_text="Executes a given bash command and returns its output.",
        page_title="Tools Reference",
        breadcrumb="Tools Reference > Bash",
        extras={"name": "Bash"},
    ),
    NodePayload(
        node_id="s_permissions",
        label="Section",
        raw_text="Claude Code uses a permission system to control which operations are allowed. Each tool requires explicit permission before it can run.",
        page_title="Security",
        breadcrumb="Security > Permissions > Overview",
    ),
    NodePayload(
        node_id="cb_install",
        label="CodeBlock",
        raw_text="npm install -g @anthropic-ai/claude-code",
        page_title="Getting Started",
        breadcrumb="Getting Started > Installation",
        extras={"language": "bash", "preceding": "Install Claude Code globally:"},
    ),
    NodePayload(
        node_id="hook_prestop",
        label="Hook",
        raw_text="Fires before Claude stops responding. Use to run cleanup or save state.",
        page_title="Hooks",
        breadcrumb="Hooks > PreStop",
        extras={"name": "PreStop"},
    ),
    NodePayload(
        node_id="sk_model",
        label="SettingKey",
        raw_text="The Claude model to use for responses.",
        page_title="Settings",
        breadcrumb="Settings > model",
        extras={"name": "model"},
    ),
]


async def main():
    for fixture in FIXTURES:
        print(f"\n{'='*60}")
        print(f"Node: {fixture.node_id} ({fixture.label})")
        print(f"Raw:  {fixture.raw_text[:80]}...")
        result = await generate_context(fixture)
        print(f"Context: {result.context}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run the smoke test**

```bash
cd contextual_pipeline
python scripts/test_agent_smoke.py
```
Expected: 5 context strings printed, each 1-3 sentences, no markdown, no preamble.
If you get a `ValidationError` (context too short/long), the model output didn't fit the schema — check the prompt is being sent correctly.
If you get `AuthenticationError`, verify `ANTHROPIC_API_KEY` in .env is correct.

- [ ] **Step 5: Commit**

```bash
git add src/prompts/ src/agents/ scripts/
git commit -m "feat: PydanticAI agent with Claude Sonnet + prompt template; smoke test passes"
```

---

## Task 10: Activities — fetch

**Files:**
- Create: `contextual_pipeline/src/activities/fetch.py`

- [ ] **Step 1: Create fetch.py**

Create `contextual_pipeline/src/activities/fetch.py`:
```python
from datetime import timedelta
from typing import Literal
from temporalio import activity
from temporalio.exceptions import ApplicationError
from ..neo4j_client import load_cypher, run_query_async
from ..schemas import NodePayload

ResumeMode = Literal["all", "missing_only", "by_source"]

LABEL_TO_IDS_CYPHER: dict[str, str] = {
    "Section":       "fetch_ids/sections.cypher",
    "CodeBlock":     "fetch_ids/codeblocks.cypher",
    "TableRow":      "fetch_ids/tablerows.cypher",
    "Callout":       "fetch_ids/callouts.cypher",
    "Tool":          "fetch_ids/tools.cypher",
    "Hook":          "fetch_ids/hooks.cypher",
    "SettingKey":    "fetch_ids/settingkeys.cypher",
    "PermissionMode":"fetch_ids/permissionmodes.cypher",
    "MessageType":   "fetch_ids/messagetypes.cypher",
    "Provider":      "fetch_ids/providers.cypher",
}

LABEL_TO_NODES_CYPHER: dict[str, str] = {
    "Section":       "fetch_nodes/sections.cypher",
    "CodeBlock":     "fetch_nodes/codeblocks.cypher",
    "TableRow":      "fetch_nodes/tablerows.cypher",
    "Callout":       "fetch_nodes/callouts.cypher",
    "Tool":          "fetch_nodes/entities.cypher",
    "Hook":          "fetch_nodes/entities.cypher",
    "SettingKey":    "fetch_nodes/entities.cypher",
    "PermissionMode":"fetch_nodes/entities.cypher",
    "MessageType":   "fetch_nodes/entities.cypher",
    "Provider":      "fetch_nodes/entities.cypher",
}


def _row_to_payload(label: str, row: dict) -> NodePayload:
    extras: dict[str, str] = {}
    if label == "Section":
        extras = {"synthesized": str(row.get("synthesized", "false")).lower()}
    elif label == "CodeBlock":
        extras = {
            "language": row.get("language") or "code",
            "preceding": row.get("preceding") or "",
        }
    elif label == "TableRow":
        raw_headers = row.get("headers") or []
        raw_cells = row.get("cells") or []
        if isinstance(raw_headers, list):
            raw_headers = "|".join(str(h) for h in raw_headers)
        if isinstance(raw_cells, list):
            raw_cells = "|".join(str(c) for c in raw_cells)
        extras = {"headers": raw_headers, "cells": raw_cells}
    elif label == "Callout":
        extras = {"callout_kind": row.get("callout_kind") or "note"}
    else:
        extras = {"name": row.get("name") or row.get("node_id", "")}

    return NodePayload(
        node_id=str(row["node_id"]),
        label=label,
        raw_text=row.get("raw_text") or "",
        page_title=row.get("page_title"),
        page_description=row.get("page_description"),
        breadcrumb=row.get("breadcrumb"),
        parent_text=row.get("parent_text"),
        extras=extras,
    )


@activity.defn
async def fetch_node_ids(
    label: str,
    resume_mode: ResumeMode = "all",
    target_sources: list[str] | None = None,
) -> list[str]:
    cypher_path = LABEL_TO_IDS_CYPHER.get(label)
    if not cypher_path:
        raise ApplicationError(f"Unknown label: {label}", non_retryable=True)
    cypher = load_cypher(cypher_path)
    params = {
        "resume_mode": resume_mode,
        "target_sources": target_sources or [],
    }
    rows = await run_query_async(cypher, params)
    return [str(r["node_id"]) for r in rows]


@activity.defn
async def fetch_nodes_batch(label: str, ids: list[str]) -> list[NodePayload]:
    if not ids:
        return []
    cypher_path = LABEL_TO_NODES_CYPHER.get(label)
    if not cypher_path:
        raise ApplicationError(f"Unknown label: {label}", non_retryable=True)
    cypher = load_cypher(cypher_path)

    try:
        rows = await run_query_async(cypher, {"ids": ids})
    except Exception as exc:
        raise ApplicationError(f"Neo4j fetch failed for {label}: {exc}") from exc

    payloads = []
    for row in rows:
        row_label = row.get("label") or label
        try:
            payloads.append(_row_to_payload(row_label, row))
        except Exception as exc:
            activity.logger.warning(f"Skipping malformed row {row.get('node_id')}: {exc}")
    return payloads
```

- [ ] **Step 2: Commit**

```bash
git add src/activities/fetch.py
git commit -m "feat: fetch_node_ids and fetch_nodes_batch activities"
```

---

## Task 11: Activities — generate_context

**Files:**
- Create: `contextual_pipeline/src/activities/generate_context.py`

- [ ] **Step 1: Create generate_context.py**

Create `contextual_pipeline/src/activities/generate_context.py`:
```python
from datetime import timedelta
from temporalio import activity
from temporalio.exceptions import ApplicationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError
from ..schemas import NodePayload, ContextResult
from ..agents.context_agent import generate_context as _agent_generate


class _RetryableError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_RetryableError),
    reraise=True,
)
async def _call_with_retry(payload: NodePayload) -> ContextResult:
    try:
        return await _agent_generate(payload)
    except ValidationError as exc:
        raise _RetryableError(f"Schema validation failed, retrying: {exc}") from exc
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("connection", "timeout", "502", "503", "429")):
            raise _RetryableError(f"Transient error: {exc}") from exc
        raise ApplicationError(
            f"Non-retryable LLM error for {payload.node_id}: {exc}",
            non_retryable=True,
        ) from exc


@activity.defn
async def generate_context(payload: NodePayload) -> ContextResult:
    try:
        return await _call_with_retry(payload)
    except _RetryableError as exc:
        raise ApplicationError(
            f"LLM failed after 3 attempts for {payload.node_id}: {exc}",
            non_retryable=True,
        ) from exc
```

- [ ] **Step 2: Commit**

```bash
git add src/activities/generate_context.py
git commit -m "feat: generate_context activity with tenacity retry (3 attempts, exp backoff)"
```

---

## Task 12: Activities — embed

**Files:**
- Create: `contextual_pipeline/src/activities/embed.py`

- [ ] **Step 1: Create embed.py**

Create `contextual_pipeline/src/activities/embed.py`:
```python
from temporalio import activity
from temporalio.exceptions import ApplicationError
from openai import AsyncOpenAI
from ..config import get_settings


def _get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(base_url=s.embedding_base_url, api_key="ollama")


@activity.defn
async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = _get_client()
    batch_size = s.embedding_batch_size
    results: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        activity.heartbeat(f"embedding batch {i // batch_size + 1}")
        try:
            response = await client.embeddings.create(
                model=s.embedding_model,
                input=chunk,
            )
        except Exception as exc:
            raise ApplicationError(
                f"Embedding API error at chunk {i}: {exc}"
            ) from exc

        for item in response.data:
            results.append(item.embedding)

    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/activities/embed.py
git commit -m "feat: embed_batch activity via Ollama OpenAI-compat endpoint"
```

---

## Task 13: Activities — write

**Files:**
- Create: `contextual_pipeline/src/activities/write.py`

- [ ] **Step 1: Create write.py**

Create `contextual_pipeline/src/activities/write.py`:
```python
from temporalio import activity
from temporalio.exceptions import ApplicationError
from ..neo4j_client import load_cypher, run_query_async
from ..config import get_settings

_WRITE_CYPHER = load_cypher("write_context.cypher")


@activity.defn
async def write_results(rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        result = await run_query_async(_WRITE_CYPHER, {"rows": rows})
        updated = result[0]["updated"] if result else 0
        activity.logger.info(f"Wrote {updated} nodes to Neo4j")
        return updated
    except Exception as exc:
        raise ApplicationError(f"Neo4j write failed: {exc}") from exc
```

- [ ] **Step 2: Commit**

```bash
git add src/activities/write.py
git commit -m "feat: write_results activity using parameterized write_context.cypher"
```

---

## Task 14: PerTypeWorkflow (child workflow)

**Files:**
- Create: `contextual_pipeline/src/workflows/per_type.py`

- [ ] **Step 1: Create per_type.py**

Create `contextual_pipeline/src/workflows/per_type.py`:
```python
from datetime import timedelta
from dataclasses import dataclass, field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..schemas import NodePayload, ContextResult
    from ..routing import decide_route, build_raw_text_repr, templater_for
    from ..config import get_settings
    from ..activities.fetch import fetch_nodes_batch
    from ..activities.generate_context import generate_context
    from ..activities.embed import embed_batch
    from ..activities.write import write_results

_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
_SHORT_TO = timedelta(minutes=2)
_EMBED_TO = timedelta(minutes=5)
_WRITE_TO = timedelta(minutes=2)
_LLM_TO   = timedelta(minutes=3)


@dataclass
class PerTypeResult:
    label: str
    processed: int = 0
    skipped: int = 0
    templated: int = 0
    llm_called: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)


@workflow.defn
class PerTypeWorkflow:
    @workflow.run
    async def run(self, label: str, ids: list[str]) -> PerTypeResult:
        stats = PerTypeResult(label=label)
        settings = get_settings()

        payloads: list[NodePayload] = await workflow.execute_activity(
            fetch_nodes_batch,
            args=[label, ids],
            start_to_close_timeout=_SHORT_TO,
            retry_policy=_RETRY,
        )

        rows_to_write: list[dict] = []
        to_embed_indices: list[int] = []

        for payload in payloads:
            route = decide_route(payload)
            raw_repr = build_raw_text_repr(payload)

            if route == "skip":
                rows_to_write.append({
                    "node_id": payload.node_id,
                    "context": "",
                    "indexed_text": "",
                    "embedding": None,
                    "context_source": "skipped",
                    "context_version": settings.context_version,
                })
                stats.skipped += 1
                continue

            if route == "template":
                ctx = templater_for(payload.label)(payload)
                rows_to_write.append({
                    "node_id": payload.node_id,
                    "context": ctx,
                    "indexed_text": ctx + "\n\n" + raw_repr,
                    "embedding": None,
                    "context_source": "template",
                    "context_version": settings.context_version,
                })
                to_embed_indices.append(len(rows_to_write) - 1)
                stats.templated += 1
                continue

            # llm
            try:
                ctx_result: ContextResult = await workflow.execute_activity(
                    generate_context,
                    args=[payload],
                    start_to_close_timeout=_LLM_TO,
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                ctx = ctx_result.context
                rows_to_write.append({
                    "node_id": payload.node_id,
                    "context": ctx,
                    "indexed_text": ctx + "\n\n" + raw_repr,
                    "embedding": None,
                    "context_source": "llm",
                    "context_version": settings.context_version,
                })
                to_embed_indices.append(len(rows_to_write) - 1)
                stats.llm_called += 1
            except Exception as exc:
                workflow.logger.error(f"LLM failed for {payload.node_id}: {exc}")
                stats.failed += 1
                stats.failed_ids.append(payload.node_id)

        # Batch embed all non-skipped rows
        if to_embed_indices:
            texts = [rows_to_write[i]["indexed_text"] for i in to_embed_indices]
            embeddings: list[list[float]] = await workflow.execute_activity(
                embed_batch,
                args=[texts],
                start_to_close_timeout=_EMBED_TO,
                retry_policy=_RETRY,
            )
            for idx, emb in zip(to_embed_indices, embeddings):
                rows_to_write[idx]["embedding"] = emb

        # Write all rows (including skipped — idempotent)
        rows_for_db = [r for r in rows_to_write if r["embedding"] is not None or r["context_source"] == "skipped"]
        await workflow.execute_activity(
            write_results,
            args=[rows_for_db],
            start_to_close_timeout=_WRITE_TO,
            retry_policy=_RETRY,
        )

        stats.processed = len(payloads)
        return stats
```

- [ ] **Step 2: Commit**

```bash
git add src/workflows/per_type.py
git commit -m "feat: PerTypeWorkflow - routes, templates, LLM, embeds, writes per batch"
```

---

## Task 15: Parent workflow

**Files:**
- Create: `contextual_pipeline/src/workflows/parent.py`

- [ ] **Step 1: Create parent.py**

Create `contextual_pipeline/src/workflows/parent.py`:
```python
from datetime import timedelta
from dataclasses import dataclass, field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..activities.fetch import fetch_node_ids, ResumeMode
    from .per_type import PerTypeWorkflow, PerTypeResult

LABELS = [
    "CodeBlock", "TableRow", "Section", "Callout",
    "Tool", "Hook", "SettingKey", "PermissionMode", "MessageType", "Provider",
]
BATCH_SIZE = 50
CONTINUE_AS_NEW_AFTER = 500


@dataclass
class PipelineResult:
    by_label: dict[str, dict] = field(default_factory=dict)
    total_child_workflows: int = 0


@workflow.defn
class ContextualizationWorkflow:
    @workflow.run
    async def run(
        self,
        resume_mode: ResumeMode = "all",
        target_sources: list[str] | None = None,
        pilot_limit: int | None = None,
        labels: list[str] | None = None,
    ) -> PipelineResult:
        result = PipelineResult()
        child_count = 0
        active_labels = labels or LABELS

        for label in active_labels:
            all_ids: list[str] = await workflow.execute_activity(
                fetch_node_ids,
                args=[label, resume_mode, target_sources],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if pilot_limit:
                all_ids = all_ids[:pilot_limit]

            batches = [all_ids[i:i+BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
            label_stats: dict = {"processed": 0, "skipped": 0, "templated": 0,
                                  "llm_called": 0, "failed": 0, "failed_ids": []}

            for batch in batches:
                child_result: PerTypeResult = await workflow.execute_child_workflow(
                    PerTypeWorkflow.run,
                    args=[label, batch],
                    id=f"per-type-{label}-{workflow.now().timestamp():.0f}",
                )
                label_stats["processed"] += child_result.processed
                label_stats["skipped"] += child_result.skipped
                label_stats["templated"] += child_result.templated
                label_stats["llm_called"] += child_result.llm_called
                label_stats["failed"] += child_result.failed
                label_stats["failed_ids"].extend(child_result.failed_ids)
                child_count += 1

                if child_count >= CONTINUE_AS_NEW_AFTER:
                    workflow.continue_as_new(
                        resume_mode=resume_mode,
                        target_sources=target_sources,
                        pilot_limit=pilot_limit,
                        labels=active_labels[active_labels.index(label):],
                    )

            result.by_label[label] = label_stats

        result.total_child_workflows = child_count
        return result
```

- [ ] **Step 2: Commit**

```bash
git add src/workflows/parent.py
git commit -m "feat: ContextualizationWorkflow parent with continue_as_new and pilot_limit"
```

---

## Task 16: Worker and starter

**Files:**
- Create: `contextual_pipeline/src/worker.py`
- Create: `contextual_pipeline/src/starter.py`

- [ ] **Step 1: Create worker.py**

Create `contextual_pipeline/src/worker.py`:
```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from .config import get_settings
from .activities.fetch import fetch_node_ids, fetch_nodes_batch
from .activities.generate_context import generate_context
from .activities.embed import embed_batch
from .activities.write import write_results
from .workflows.per_type import PerTypeWorkflow
from .workflows.parent import ContextualizationWorkflow


async def main():
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    worker = Worker(
        client,
        task_queue="contextual-pipeline",
        workflows=[ContextualizationWorkflow, PerTypeWorkflow],
        activities=[
            fetch_node_ids,
            fetch_nodes_batch,
            generate_context,
            embed_batch,
            write_results,
        ],
    )
    print("Worker starting on task queue: contextual-pipeline")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Create starter.py**

Create `contextual_pipeline/src/starter.py`:
```python
"""
Usage:
  python -m src.starter                          # full run, all labels
  python -m src.starter --pilot 50               # 50 nodes per label
  python -m src.starter --resume missing_only    # only nodes with no context
  python -m src.starter --resume by_source --sources llm  # re-run LLM nodes
  python -m src.starter --labels CodeBlock Tool  # specific labels only
"""
import asyncio
import argparse
from temporalio.client import Client
from .config import get_settings
from .workflows.parent import ContextualizationWorkflow, PipelineResult


async def main(args: argparse.Namespace):
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)

    handle = await client.start_workflow(
        ContextualizationWorkflow.run,
        args={
            "resume_mode": args.resume,
            "target_sources": args.sources or None,
            "pilot_limit": args.pilot,
            "labels": args.labels or None,
        },
        id=f"contextual-pipeline-{asyncio.get_event_loop().time():.0f}",
        task_queue="contextual-pipeline",
    )
    print(f"Started workflow: {handle.id}")
    result: PipelineResult = await handle.result()
    print("\n=== Pipeline Result ===")
    for label, stats in result.by_label.items():
        print(f"\n{label}:")
        for k, v in stats.items():
            if k != "failed_ids":
                print(f"  {k}: {v}")
        if stats["failed_ids"]:
            print(f"  failed_ids: {stats['failed_ids']}")
    print(f"\nTotal child workflows: {result.total_child_workflows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="all",
                        choices=["all", "missing_only", "by_source"])
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--labels", nargs="+")
    asyncio.run(main(parser.parse_args()))
```

- [ ] **Step 3: Commit**

```bash
git add src/worker.py src/starter.py
git commit -m "feat: worker registers all activities+workflows; starter CLI with pilot/resume flags"
```

---

## Task 17: Pilot run — 50 CodeBlocks

**Goal:** Run the full pipeline on 50 CodeBlocks, then verify with 4 Neo4j queries before scaling.

- [ ] **Step 1: Start the Temporal worker in a terminal**

```bash
cd contextual_pipeline
python -m src.worker
```
Keep this running. Expected: `Worker starting on task queue: contextual-pipeline`

- [ ] **Step 2: Start the pilot workflow in a second terminal**

```bash
cd contextual_pipeline
python -m src.starter --pilot 50 --labels CodeBlock
```
Expected: workflow ID printed, then result summary when complete.

- [ ] **Step 3: Verify — indexed_text populated**

In Neo4j Browser:
```cypher
MATCH (c:CodeBlock) WHERE c.indexed_text IS NOT NULL
RETURN count(c) AS count,
       avg(size(c.indexed_text)) AS avg_indexed_text_len,
       avg(size(c.context)) AS avg_context_len
```
Expected: `count ≥ 1`, both averages > 0.

- [ ] **Step 4: Verify — embedding dimension**

```cypher
MATCH (c:CodeBlock) WHERE c.embedding IS NOT NULL
RETURN size(c.embedding) AS dim, count(*) AS count
LIMIT 1
```
Expected: `dim = 1024`.

- [ ] **Step 5: Verify — fulltext index picks up indexed_text**

```cypher
CALL db.index.fulltext.queryNodes('codeblock_fulltext', 'permission')
YIELD node, score
RETURN node.id, score
LIMIT 5
```
Expected: results returned with scores > 0.
(If fulltext index name differs, run `SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT'` first.)

- [ ] **Step 6: Verify — vector search smoke test**

```cypher
MATCH (c:CodeBlock) WHERE c.embedding IS NOT NULL
WITH c LIMIT 1
CALL db.index.vector.queryNodes('section_embedding', 5, c.embedding)
YIELD node, score
RETURN node.id, score
```
Expected: 5 nodes returned with similarity scores.
(Replace `section_embedding` with actual vector index name if different.)

- [ ] **Step 7: Report results to user and await approval before full run**

Print the output of all 4 queries. Ask: "All 4 verification checks passed — proceed with full corpus run?"

---

## Task 18: Full corpus run (gated on user approval)

- [ ] **Step 1: Run full pipeline on all labels**

```bash
cd contextual_pipeline
python -m src.starter
```
This runs `resume_mode=all` on all 10 labels with no pilot limit.

- [ ] **Step 2: Monitor via Temporal UI**

Open `http://localhost:8080` (Temporal Web UI). Watch child workflow counts and failure rates.

- [ ] **Step 3: Re-run failed nodes if any**

```bash
python -m src.starter --resume missing_only
```

- [ ] **Step 4: Final verification**

Re-run all 4 verification queries from Task 17, Steps 3-6, for each label. Confirm all nodes have `context_source IN ['llm', 'template', 'skipped']` and all non-skipped nodes have a non-null `embedding`.

```cypher
MATCH (n)
WHERE (n:Section OR n:CodeBlock OR n:TableRow OR n:Callout
       OR n:Tool OR n:Hook OR n:SettingKey OR n:PermissionMode
       OR n:MessageType OR n:Provider)
  AND n.context IS NULL
RETURN labels(n)[0] AS label, count(n) AS missing_context
```
Expected: zero rows (all processed).

---

## Self-Review Checklist

**Spec coverage:**
- [x] Property contract (context, indexed_text, embedding, context_source, context_version) — Task 14, 15
- [x] Skip/route rules — Task 5 (decide_route), tested with 30+ cases
- [x] Template rules for Section + CodeBlock — Task 6
- [x] LLM rules for all other types — Task 5 + 11
- [x] TRIVIAL_LABELS — Task 5
- [x] Parameterized Cypher only — Task 7, enforced by Task 8 safety test
- [x] Batch sizes (embed=32, LLM=single via activity) — Task 12, config
- [x] Idempotent writes — Task 13 (write_context.cypher uses SET not CREATE)
- [x] Resume modes (all/missing_only/by_source) — Task 7 (cypher WHERE clause), Task 15 (parent param)
- [x] continue_as_new after 500 child workflows — Task 15
- [x] Pilot-first gate — Task 17 (pilot 50 CodeBlocks before full run)
- [x] Failed nodes collected, not silently dropped — PerTypeResult.failed_ids
- [x] .env typo fix (Antropic→Anthropic) — Task 1
- [x] All 5 test files — Tasks 3, 5, 6, 8

**No placeholders found.**

**Type consistency:** `NodePayload`, `ContextResult`, `PerTypeResult`, `PipelineResult` — names consistent across all tasks.
