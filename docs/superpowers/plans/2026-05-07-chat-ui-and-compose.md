# Chat UI and Docker Compose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Chainlit web chat interface for the GraphRAG Q&A agent and package the full stack as a `docker compose` distribution that runs from a fresh clone, seeded by a published Neo4j dump.

**Architecture:** Ten compose services (eight long-running, two one-shot seeders). Chainlit UI calls the existing stateless `POST /ask`. Neo4j is seeded on first start by downloading and verifying a release-asset dump; Ollama pulls the embedding model on first start. Only `qa-api` and `qa-worker` see API-key secrets.

**Tech Stack:** Chainlit, httpx, Flask (existing), Temporal (existing), Neo4j 5 Community, Ollama, Docker Compose v2.

**Spec:** [`docs/superpowers/specs/2026-05-07-chat-ui-and-compose-design.md`](../specs/2026-05-07-chat-ui-and-compose-design.md)

**Repo-relative paths.** All paths in this plan are relative to the repo root (`d:\Ecompany GraphRAG\Q&A Agent\`).

---

## Task 1: Repo-root config files

**Files:**
- Create: `.env.example`
- Create: `.dockerignore`
- Modify: `.gitignore` (verify `.env` already excluded; add if not)

- [ ] **Step 1: Create root `.env.example`**

```env
# secrets — required for the QA agent to answer questions
OPENAI_API_KEY=
COHERE_API_KEY=

# Neo4j dump asset — defaults point at the published release
NEO4J_DUMP_URL=https://github.com/MilosJovicic/GraphRAG-workflow/releases/download/neo4j-dump-2026-05-07/neo4j-2026-05-07T13-19-28.dump
NEO4J_DUMP_SHA256=2c47fd8146ad4ee1d94baa65e2453340dc10cabf87470627b5707233fa38715c

# overridable defaults
NEO4J_PASSWORD=changeme-local-only
OPENAI_MODEL=gpt-5.4-mini
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B:Q8_0
COHERE_RERANK_MODEL=rerank-v3.5
```

- [ ] **Step 2: Create root `.dockerignore`**

```
**/.venv
**/__pycache__
**/.pytest_cache
**/.mypy_cache
**/.ruff_cache
**/*.egg-info
**/.env
**/.env.*
!**/.env.example
.git
.gitignore
docs
ETL/parsed
ETL/resolved
ETL/entities
contextual-embeddings-agentic-workflow
ragas_evals
tests
**/tests
**/test_*.py
*.dump
node_modules
```

- [ ] **Step 3: Verify `.gitignore` covers `.env`**

Run: `grep -E "^\.env$|^\*\.env$|^\.env\.\*$" .gitignore`
Expected: at least one match. If none, append `.env` and `.env.*` to `.gitignore`.

- [ ] **Step 4: Commit**

```bash
git add .env.example .dockerignore .gitignore
git commit -m "chore: add root .env.example and .dockerignore for compose"
```

---

## Task 2: Q&A agent shared Dockerfile

**Files:**
- Create: `graphrag-agent-workflow/Dockerfile`
- Create: `graphrag-agent-workflow/.dockerignore`

- [ ] **Step 1: Create `graphrag-agent-workflow/Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by container healthchecks; build-essential is required for any
# pure-Python deps that fall back to source builds (e.g. neo4j driver wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install -e .

# Default command is overridden by docker-compose for api vs. worker.
CMD ["python", "-m", "qa_agent.worker"]
```

- [ ] **Step 2: Create `graphrag-agent-workflow/.dockerignore`**

```
.venv
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
*.egg-info
.env
.env.*
!.env.example
tests
ragas_evals
docs
```

- [ ] **Step 3: Verify the image builds**

Run from repo root:
```bash
docker build -t graphrag-qa-agent:dev graphrag-agent-workflow/
```
Expected: build succeeds. If a missing system lib comes up (e.g. for `lxml` or `cryptography`), add it to the `apt-get install` line and rebuild.

- [ ] **Step 4: Commit**

```bash
git add graphrag-agent-workflow/Dockerfile graphrag-agent-workflow/.dockerignore
git commit -m "build: add shared Dockerfile for qa-api and qa-worker"
```

---

## Task 3: Chainlit project skeleton

**Files:**
- Create: `chat/pyproject.toml`
- Create: `chat/.dockerignore`
- Create: `chat/tests/__init__.py` (empty file, makes `tests` a package)

- [ ] **Step 1: Create `chat/pyproject.toml`**

```toml
[project]
name = "graphrag-chat"
version = "0.1.0"
description = "Chainlit chat UI for the GraphRAG Q&A agent."
requires-python = ">=3.11"
dependencies = [
    "chainlit>=1.3,<2",
    "httpx>=0.27,<1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["app"]

[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `chat/.dockerignore`**

```
.venv
__pycache__
.pytest_cache
*.egg-info
tests
```

- [ ] **Step 3: Create empty `chat/tests/__init__.py`**

(File is empty.)

- [ ] **Step 4: Verify deps install**

```bash
cd chat
python -m venv .venv
. .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```
Expected: install succeeds, `chainlit`, `httpx`, `pytest`, `respx` all on path.

- [ ] **Step 5: Commit**

```bash
git add chat/pyproject.toml chat/.dockerignore chat/tests/__init__.py
git commit -m "chore: scaffold chat/ project with chainlit and httpx deps"
```

---

## Task 4: `render_qa_response` helper (TDD)

**Files:**
- Test: `chat/tests/test_app.py`
- Create: `chat/app.py`

- [ ] **Step 1: Write the failing tests**

Create `chat/tests/test_app.py`:

```python
"""Tests for the Chainlit app pure helpers."""

from __future__ import annotations

import chainlit as cl

from app import render_qa_response


def test_render_qa_response_with_citations_uses_title():
    payload = {
        "answer": "Set the permissionMode in settings.json [1].",
        "citations": [
            {
                "id": 1,
                "node_id": "n1",
                "node_label": "Section",
                "url": "https://example.com/x#a",
                "title": "Permissions",
                "snippet": "Use permissionMode...",
                "score": 0.9,
            }
        ],
        "no_results": False,
    }

    content, elements = render_qa_response(payload)

    assert content == payload["answer"]
    assert len(elements) == 1
    assert isinstance(elements[0], cl.Text)
    assert elements[0].name == "[1] Permissions"
    assert elements[0].content == "Use permissionMode..."
    assert elements[0].display == "side"


def test_render_qa_response_falls_back_to_url_when_title_missing():
    payload = {
        "answer": "answer [1]",
        "citations": [
            {
                "id": 1,
                "node_id": "n1",
                "node_label": "Chunk",
                "url": "https://example.com/y",
                "snippet": "...",
                "score": 0.5,
            }
        ],
        "no_results": False,
    }

    _, elements = render_qa_response(payload)

    assert elements[0].name == "[1] https://example.com/y"


def test_render_qa_response_falls_back_to_node_id_when_url_missing():
    payload = {
        "answer": "answer [1]",
        "citations": [
            {
                "id": 1,
                "node_id": "section-abc",
                "node_label": "Section",
                "snippet": "...",
                "score": 0.5,
            }
        ],
        "no_results": False,
    }

    _, elements = render_qa_response(payload)

    assert elements[0].name == "[1] section-abc"


def test_render_qa_response_no_results_returns_empty_elements():
    payload = {
        "answer": "I couldn't find anything relevant in the knowledge base.",
        "citations": [],
        "no_results": True,
    }

    content, elements = render_qa_response(payload)

    assert content == payload["answer"]
    assert elements == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd chat
pytest tests/test_app.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Create `chat/app.py` with only what these tests need**

```python
"""Chainlit chat UI for the GraphRAG Q&A agent.

Logic lives in pure helpers so it can be unit-tested without a Chainlit
runtime context.
"""

from __future__ import annotations

import logging
import os

import chainlit as cl

QA_API_URL = os.environ.get("QA_API_URL", "http://qa-api:5000")
QA_API_TIMEOUT = float(os.environ.get("QA_API_TIMEOUT", "120"))

log = logging.getLogger(__name__)


def render_qa_response(payload: dict) -> tuple[str, list[cl.Text]]:
    """Convert a QA API response into a Chainlit message body and side elements."""
    answer = payload["answer"]
    if payload.get("no_results"):
        return answer, []

    elements: list[cl.Text] = []
    for citation in payload.get("citations", []):
        label = (
            citation.get("title")
            or citation.get("url")
            or citation.get("node_id", "?")
        )
        elements.append(
            cl.Text(
                name=f"[{citation['id']}] {label}",
                content=citation["snippet"],
                display="side",
            )
        )
    return answer, elements
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd chat
pytest tests/test_app.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add chat/app.py chat/tests/test_app.py
git commit -m "feat(chat): add render_qa_response helper with tests"
```

---

## Task 5: `call_qa_api` helper (TDD)

**Files:**
- Modify: `chat/tests/test_app.py` (add import and append tests)
- Modify: `chat/app.py` (add `call_qa_api` and `httpx` import)

- [ ] **Step 1: Update the import line and append the failing tests in `chat/tests/test_app.py`**

Change the existing line:
```python
from app import render_qa_response
```
to:
```python
import json

import httpx
import pytest
import respx

from app import call_qa_api, render_qa_response
```

Append these tests to the bottom of `chat/tests/test_app.py`:

```python
@pytest.mark.asyncio
async def test_call_qa_api_returns_payload_on_success():
    payload = {
        "answer": "ok",
        "citations": [],
        "no_results": False,
        "fallback_used": False,
        "plan": {"sub_queries": [], "expansion_patterns": [], "notes": ""},
        "latency_ms": {},
    }

    async with respx.mock(base_url="http://qa") as router:
        router.post("/ask").respond(200, json=payload)
        async with httpx.AsyncClient() as client:
            result = await call_qa_api(client, "http://qa", "How do I X?")

    assert result == payload


@pytest.mark.asyncio
async def test_call_qa_api_raises_on_5xx():
    async with respx.mock(base_url="http://qa") as router:
        router.post("/ask").respond(500, json={"error": "boom"})
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await call_qa_api(client, "http://qa", "How do I X?")


@pytest.mark.asyncio
async def test_call_qa_api_sends_question_in_json_body():
    async with respx.mock(base_url="http://qa") as router:
        route = router.post("/ask").respond(
            200,
            json={"answer": "x", "citations": [], "no_results": False},
        )
        async with httpx.AsyncClient() as client:
            await call_qa_api(client, "http://qa", "the question")

        assert route.called
        sent = route.calls.last.request
        assert sent.headers["content-type"].startswith("application/json")
        assert json.loads(sent.content) == {"question": "the question"}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd chat
pytest tests/test_app.py -v
```
Expected: FAIL with `ImportError: cannot import name 'call_qa_api' from 'app'`.

- [ ] **Step 3: Add `call_qa_api` to `chat/app.py`**

In `chat/app.py`, add `import httpx` to the existing imports:

```python
import httpx
```

Then insert this function above `render_qa_response`:

```python
async def call_qa_api(client: httpx.AsyncClient, url: str, question: str) -> dict:
    """POST one question to the Flask QA API and return the parsed JSON body.

    Raises ``httpx.HTTPStatusError`` on non-2xx and other ``httpx`` exceptions
    on transport failures.
    """
    resp = await client.post(
        f"{url}/ask",
        json={"question": question},
        timeout=QA_API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd chat
pytest tests/test_app.py -v
```
Expected: 7 passed (4 from Task 4 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add chat/app.py chat/tests/test_app.py
git commit -m "feat(chat): add call_qa_api helper with tests"
```

---

## Task 6: Chainlit handlers (welcome + on_message)

**Files:**
- Modify: `chat/app.py` (append handler functions)

The decorated handlers can't be unit-tested without Chainlit runtime context — they are exercised only by the smoke test in Task 12. Keep them thin.

- [ ] **Step 1: Append handler functions to `chat/app.py`**

Append at the bottom of `chat/app.py`:

```python
@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content="Ask me about Claude Code documentation.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    async with httpx.AsyncClient() as client:
        try:
            payload = await call_qa_api(client, QA_API_URL, message.content)
        except Exception:
            log.exception("qa-api call failed")
            await cl.Message(
                content=(
                    "The agent ran into a problem. Try again, or check "
                    "`docker compose logs qa-api`."
                )
            ).send()
            return

    content, elements = render_qa_response(payload)
    await cl.Message(content=content, elements=elements).send()
```

- [ ] **Step 2: Verify the file imports cleanly**

```bash
cd chat
python -c "import app; print(app.QA_API_URL)"
```
Expected: prints `http://qa-api:5000` (the default).

- [ ] **Step 3: Run the test suite again to confirm nothing broke**

```bash
cd chat
pytest -v
```
Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add chat/app.py
git commit -m "feat(chat): wire on_chat_start and on_message handlers"
```

---

## Task 7: Chainlit Dockerfile

**Files:**
- Create: `chat/Dockerfile`

- [ ] **Step 1: Create `chat/Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app.py ./

RUN pip install .

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]
```

- [ ] **Step 2: Verify the image builds**

```bash
docker build -t graphrag-chat:dev chat/
```
Expected: build succeeds.

- [ ] **Step 3: Smoke-run the container in isolation (optional, no qa-api)**

```bash
docker run --rm -p 8000:8000 -e QA_API_URL=http://localhost:9999 graphrag-chat:dev
```
Open `http://localhost:8000`. Expected: Chainlit UI loads, welcome message shows. Sending a message returns the friendly error (because `QA_API_URL` does not resolve). Stop the container with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add chat/Dockerfile
git commit -m "build(chat): add Chainlit Dockerfile"
```

---

## Task 8: Neo4j seed container

**Files:**
- Create: `infra/neo4j-seed/Dockerfile`
- Create: `infra/neo4j-seed/seed.sh`

- [ ] **Step 1: Create `infra/neo4j-seed/seed.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
MARKER="${DATA_DIR}/.seeded"
DUMP_PATH="/tmp/graph.dump"

if [ -f "$MARKER" ]; then
  echo "[neo4j-seed] Volume already seeded at ${MARKER}, skipping."
  exit 0
fi

: "${NEO4J_DUMP_URL:?NEO4J_DUMP_URL is required}"
: "${NEO4J_DUMP_SHA256:?NEO4J_DUMP_SHA256 is required}"

echo "[neo4j-seed] Downloading dump from ${NEO4J_DUMP_URL}"
curl -fL --retry 3 --retry-delay 5 "${NEO4J_DUMP_URL}" -o "${DUMP_PATH}"

echo "[neo4j-seed] Verifying SHA-256"
echo "${NEO4J_DUMP_SHA256}  ${DUMP_PATH}" | sha256sum -c -

echo "[neo4j-seed] Loading dump into Neo4j volume"
neo4j-admin database load --from-path=/tmp --overwrite-destination=true neo4j

touch "${MARKER}"
echo "[neo4j-seed] Done."
```

- [ ] **Step 2: Create `infra/neo4j-seed/Dockerfile`**

```dockerfile
FROM neo4j:5-community

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

COPY seed.sh /usr/local/bin/seed.sh
RUN chmod +x /usr/local/bin/seed.sh

# Run as the neo4j user so files written to /data have the right ownership for
# the neo4j service that mounts the same volume.
USER neo4j

ENTRYPOINT ["/usr/local/bin/seed.sh"]
```

- [ ] **Step 3: Verify the image builds**

```bash
docker build -t graphrag-neo4j-seed:dev infra/neo4j-seed/
```
Expected: build succeeds.

- [ ] **Step 4: Sanity-check the script's marker-skip path**

```bash
docker run --rm \
  -e NEO4J_DUMP_URL=http://invalid \
  -e NEO4J_DUMP_SHA256=0 \
  -v $(pwd)/infra-test-vol:/data \
  graphrag-neo4j-seed:dev
```
First run: fails on download (expected — the URL is invalid).
Second run setup: `mkdir -p infra-test-vol && touch infra-test-vol/.seeded`, re-run the same command.
Expected: prints `Volume already seeded ... skipping.` and exits 0.
Cleanup: `rm -rf infra-test-vol`.

- [ ] **Step 5: Commit**

```bash
git add infra/neo4j-seed/seed.sh infra/neo4j-seed/Dockerfile
git commit -m "feat(infra): add idempotent Neo4j dump seed container"
```

---

## Task 9: Ollama seed container

**Files:**
- Create: `infra/ollama-seed/Dockerfile`
- Create: `infra/ollama-seed/seed.sh`

- [ ] **Step 1: Create `infra/ollama-seed/seed.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
: "${EMBEDDING_MODEL:?EMBEDDING_MODEL is required}"

echo "[ollama-seed] Waiting for Ollama at ${OLLAMA_URL}"
for _ in $(seq 1 60); do
  if curl -fs "${OLLAMA_URL}/api/tags" > /dev/null; then
    break
  fi
  sleep 2
done

if ! curl -fs "${OLLAMA_URL}/api/tags" > /dev/null; then
  echo "[ollama-seed] Ollama never became reachable" >&2
  exit 1
fi

echo "[ollama-seed] Pulling model: ${EMBEDDING_MODEL}"
# Ollama's /api/pull streams progress; -N keeps stdout flowing, -f fails on
# HTTP errors. The pull is idempotent: Ollama short-circuits if the model is
# already present.
curl -fSN "${OLLAMA_URL}/api/pull" -d "{\"name\": \"${EMBEDDING_MODEL}\"}"

echo "[ollama-seed] Done."
```

- [ ] **Step 2: Create `infra/ollama-seed/Dockerfile`**

```dockerfile
FROM alpine:3.20

RUN apk add --no-cache curl bash ca-certificates

COPY seed.sh /usr/local/bin/seed.sh
RUN chmod +x /usr/local/bin/seed.sh

ENTRYPOINT ["/usr/local/bin/seed.sh"]
```

- [ ] **Step 3: Verify the image builds**

```bash
docker build -t graphrag-ollama-seed:dev infra/ollama-seed/
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add infra/ollama-seed/seed.sh infra/ollama-seed/Dockerfile
git commit -m "feat(infra): add Ollama model-pull seed container"
```

---

## Task 10: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `docker-compose.yml` at the repo root**

```yaml
services:
  temporal-postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: temporal
      POSTGRES_PASSWORD: temporal
      POSTGRES_DB: temporal
    volumes:
      - temporal-postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U temporal"]
      interval: 5s
      timeout: 5s
      retries: 10

  temporal:
    image: temporalio/auto-setup:1.25.2
    environment:
      DB: postgres12
      DB_PORT: "5432"
      POSTGRES_USER: temporal
      POSTGRES_PWD: temporal
      POSTGRES_SEEDS: temporal-postgres
    depends_on:
      temporal-postgres:
        condition: service_healthy
    ports:
      - "7233:7233"
    healthcheck:
      test: ["CMD-SHELL", "tctl --address temporal:7233 cluster health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 30s

  temporal-ui:
    image: temporalio/ui:2.30.0
    environment:
      TEMPORAL_ADDRESS: temporal:7233
      TEMPORAL_CORS_ORIGINS: http://localhost:8233
    ports:
      - "8233:8080"
    depends_on:
      temporal:
        condition: service_healthy

  neo4j-seed:
    build: ./infra/neo4j-seed
    environment:
      NEO4J_DUMP_URL: ${NEO4J_DUMP_URL}
      NEO4J_DUMP_SHA256: ${NEO4J_DUMP_SHA256}
    volumes:
      - neo4j-data:/data
    restart: "no"

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_server_memory_pagecache_size: 512M
      NEO4J_server_memory_heap_initial__size: 512M
      NEO4J_server_memory_heap_max__size: 1G
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j-data:/data
    depends_on:
      neo4j-seed:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p $${NEO4J_PASSWORD} 'RETURN 1' || exit 1"]
      interval: 10s
      timeout: 10s
      retries: 30
      start_period: 30s

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    healthcheck:
      test: ["CMD-SHELL", "curl -fs http://localhost:11434/api/tags || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 30

  ollama-seed:
    build: ./infra/ollama-seed
    environment:
      OLLAMA_URL: http://ollama:11434
      EMBEDDING_MODEL: ${EMBEDDING_MODEL}
    depends_on:
      ollama:
        condition: service_healthy
    restart: "no"

  qa-api:
    build:
      context: ./graphrag-agent-workflow
    env_file: .env
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      TEMPORAL_HOST: temporal:7233
      TEMPORAL_NAMESPACE: default
      OLLAMA_BASE_URL: http://ollama:11434/v1
    command: ["python", "-m", "qa_agent.api"]
    depends_on:
      neo4j:
        condition: service_healthy
      temporal:
        condition: service_healthy
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -fs http://localhost:5000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 20s

  qa-worker:
    build:
      context: ./graphrag-agent-workflow
    env_file: .env
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      TEMPORAL_HOST: temporal:7233
      TEMPORAL_NAMESPACE: default
      OLLAMA_BASE_URL: http://ollama:11434/v1
    command: ["python", "-m", "qa_agent.worker"]
    depends_on:
      neo4j:
        condition: service_healthy
      temporal:
        condition: service_healthy
      ollama:
        condition: service_healthy

  chainlit:
    build:
      context: ./chat
    environment:
      QA_API_URL: http://qa-api:5000
    ports:
      - "8000:8000"
    depends_on:
      qa-api:
        condition: service_healthy

volumes:
  temporal-postgres-data:
  neo4j-data:
  ollama-data:
```

- [ ] **Step 2: Validate the compose file (cheap pre-smoke check)**

```bash
docker compose config > /dev/null
```
Expected: exit 0, no output. If it errors out about a missing variable, ensure `.env` exists locally (`cp .env.example .env`) before re-running.

- [ ] **Step 3: Verify all build contexts resolve**

```bash
docker compose build --pull
```
Expected: all four locally-built images (qa-api, qa-worker, chainlit, neo4j-seed, ollama-seed — `qa-api` and `qa-worker` share one image so it builds once) build successfully. This may take several minutes the first time.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add docker-compose.yml with full chat + agent stack"
```

---

## Task 11: README updates and legacy notice

**Files:**
- Modify: `README.md` (append a new "Run via Docker Compose" section)
- Modify: `graphrag-agent-workflow/.env.example` (add legacy header)

- [ ] **Step 1: Append a "Run via Docker Compose" section to root `README.md`**

Append this section at the end of `README.md`:

```markdown

## Run via Docker Compose

A full-stack `docker compose` distribution is included so you can clone and run
the system end-to-end without a host Python or Neo4j install.

### Prerequisites

- Docker Desktop or Docker Engine 24+ with Compose v2.
- ~6 GB free disk for images, Neo4j data, and the Ollama embedding model.
- An OpenAI API key (used by the planner and answerer).
- Optionally, a Cohere API key for reranking; if omitted the workflow falls
  back to RRF top-8.

### One-time setup

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY (and optionally COHERE_API_KEY).
docker compose config > /dev/null   # cheap pre-flight: validates YAML + env interpolation
```

### Start the stack

```bash
docker compose up
```

On the first run, two one-shot containers seed the system:

- `neo4j-seed` downloads the published Neo4j dump, verifies its SHA-256, and
  restores it into the `neo4j-data` volume. The dump is ~298 MB; the URL and
  checksum are pinned in `.env.example`.
- `ollama-seed` waits for the Ollama service and pulls the embedding model
  configured in `EMBEDDING_MODEL`.

Both seed steps are idempotent and skip on subsequent `docker compose up` runs.

### Use the chat UI

Open [`http://localhost:8000`](http://localhost:8000) in a browser. Ask a
question about the Claude Code documentation. The answer renders in the chat
window; supporting citations appear as expandable elements in the side panel.

Other exposed endpoints:

- Neo4j Browser: [`http://localhost:7474`](http://localhost:7474) (user
  `neo4j`, password from `NEO4J_PASSWORD`)
- Temporal UI: [`http://localhost:8233`](http://localhost:8233)

### Smoke test

After bring-up, validate the stack:

1. `docker compose ps` — every service is `running` or `exited` (`neo4j-seed`
   and `ollama-seed` exit cleanly after success).
2. Open `http://localhost:8000`, ask: *"How do I configure the Bash tool's
   permission mode?"* — expect an answer plus at least one citation in the
   side panel within ~30 seconds.
3. `docker compose down && docker compose up` — second startup is fast because
   the seed containers detect their marker / model and skip.

### Stop the stack

```bash
docker compose down              # stop containers, keep data volumes
docker compose down -v           # stop and wipe volumes (forces re-seed)
```

### Note on configuration scope

`OPENAI_API_KEY` and `COHERE_API_KEY` are mounted only into `qa-api` and
`qa-worker` via `env_file: .env`. Chainlit, Neo4j, Ollama, and the seed
containers never see them.
```

- [ ] **Step 2: Add a legacy header to `graphrag-agent-workflow/.env.example`**

Modify the top of `graphrag-agent-workflow/.env.example`. Insert these lines at the very top:

```env
# LEGACY: this file documents the host-Python-only workflow (uvicorn directly,
# Neo4j installed locally, Ollama on localhost). For the docker-compose flow,
# use the canonical .env.example at the repo root.
#
```

- [ ] **Step 3: Commit**

```bash
git add README.md graphrag-agent-workflow/.env.example
git commit -m "docs: add docker-compose quickstart and mark legacy .env.example"
```

---

## Task 12: End-to-end smoke test (manual)

This task is a manual checklist. No new files, no commits unless step 5 surfaces a fix.

- [ ] **Step 1: Pre-flight**

```bash
docker compose config > /dev/null
test -f .env && grep -q "^OPENAI_API_KEY=." .env && echo "ok" || echo "set OPENAI_API_KEY in .env"
```
Expected: `ok`. If not, fill in the key.

- [ ] **Step 2: First-run bring-up**

```bash
docker compose up
```
Watch the logs. Expected sequence (rough):

1. `temporal-postgres` healthy.
2. `neo4j-seed` downloads the dump, verifies SHA, runs `neo4j-admin load`, exits 0.
3. `neo4j` becomes healthy.
4. `ollama` becomes healthy; `ollama-seed` pulls the model and exits 0.
5. `temporal` becomes healthy.
6. `qa-api`, `qa-worker` start; `qa-api` becomes healthy.
7. `chainlit` starts.

In another terminal: `docker compose ps`. All long-running services should be `Up (healthy)`; `neo4j-seed` and `ollama-seed` should be `Exited (0)`.

- [ ] **Step 3: Functional check**

Open `http://localhost:8000`. Ask: *"How do I configure the Bash tool's permission mode?"*
Expected:
- Answer text appears within ~30 seconds.
- At least one citation shows in the side panel with a title or URL and a snippet.
- Answer body contains `[1]`-style citation markers that match the side-panel IDs.

- [ ] **Step 4: Restart skip-seed check**

```bash
docker compose down
docker compose up
```
Expected: `neo4j-seed` logs `Volume already seeded ... skipping.` and exits within seconds. `ollama-seed` either skips fast or no-ops because the model is already present. Total restart should take <60 seconds.

- [ ] **Step 5: Stop**

```bash
docker compose down
```

- [ ] **Step 6: Final commit only if fixes were needed**

If steps 1–5 surfaced a small fix (typo in env var, healthcheck command, etc.), commit it now. Otherwise, no commit.

```bash
git status
# (only commit if there are changes)
```

---

## Verification summary

After Task 12 completes successfully:

- A recipient can `git clone`, `cp .env.example .env`, fill in `OPENAI_API_KEY`, and run `docker compose up` to get a working system.
- All API-key secrets are scoped to `qa-api` and `qa-worker` only.
- Bring-up is one-shot and idempotent across restarts.
- The Chainlit UI surfaces answer + citations and degrades gracefully on backend errors.
- Unit tests for the chat helpers pass (`pytest` from `chat/`).
- Existing QA agent tests are unaffected (`pytest` from `graphrag-agent-workflow/` continues to behave as before).
