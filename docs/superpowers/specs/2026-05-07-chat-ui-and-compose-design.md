# Chat UI and Docker Compose Design

**Date:** 2026-05-07
**Status:** Approved (awaiting implementation plan)
**Scope:** Add a browser chat interface for the GraphRAG Q&A agent and package the
full stack as a `docker compose` distribution that someone can clone from GitHub
and run end-to-end.

## Goals

- Provide a browser-based chat UI an end user can interact with.
- Package the runtime so a recipient can `git clone`, set one or two API keys, run
  `docker compose up`, and have a working Q&A system.
- Avoid forcing the recipient to run the ETL or contextual-embeddings pipelines —
  ship a pre-populated Neo4j graph as a release asset.

## Non-Goals

- Multi-turn chat with conversation memory. Each user turn calls the existing
  stateless `POST /ask` endpoint with only the new question.
- Authentication. The compose stack is local-only.
- Streaming responses. The backend does not stream; the UI does not pretend to.
- Plan or latency surfaces in the UI. Only the answer and citations render.
- Containerizing the ETL or contextual-embeddings pipelines. Recipients consume
  the dump; they do not regenerate it.

## Architecture

Ten services in a single `docker-compose.yml` at the repo root (eight long-running
plus two one-shots).

```
┌────────────┐        ┌────────────┐        ┌──────────────┐
│  Chainlit  │ HTTP → │   qa-api   │  ───→  │   temporal   │
│  (web UI)  │ :8000  │  (Flask)   │        │ (auto-setup) │
└────────────┘        └────────────┘        └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌────────────┐
                                            │ qa-worker  │
                                            │ (Temporal) │
                                            └──┬──┬──┬───┘
                                  Neo4j ◄──────┘  │  │
                                  Ollama ◄────────┘  │
                                  OpenAI/Cohere ◄────┘
```

| Service | Image / Build | Purpose | Host port |
| --- | --- | --- | --- |
| `chainlit` | build `chat/` | Browser chat UI | `8000` |
| `qa-api` | build `graphrag-agent-workflow/` | Existing Flask `POST /ask` | none (internal) |
| `qa-worker` | build `graphrag-agent-workflow/` | Temporal worker for QA workflows | none |
| `temporal` | `temporalio/auto-setup` | Temporal frontend | `7233` (optional) |
| `temporal-postgres` | `postgres:16` | Temporal persistence | none |
| `temporal-ui` | `temporalio/ui` | Workflow inspector | `8233` |
| `neo4j` | `neo4j:5-community` | Graph + indexes | `7474`, `7687` |
| `neo4j-seed` | build `infra/neo4j-seed/` | One-shot dump restore | none |
| `ollama` | `ollama/ollama` | Embedding server | `11434` |
| `ollama-seed` | build `infra/ollama-seed/` | One-shot model pull | none |

`qa-api` and `qa-worker` share one Dockerfile (`graphrag-agent-workflow/Dockerfile`) and
differ only in `command:` (`python -m qa_agent.api` vs `python -m qa_agent.worker`).

## Data Flow

### Chat turn (every user message)

1. Chainlit `@cl.on_message` fires.
2. Helper `call_qa_api(client, QA_API_URL, question)` does one `POST /ask` with
   `{"question": text}`. No history, no session — strictly stateless.
3. Flask validates with `QARequest`, starts `QAWorkflow` on Temporal, awaits the
   result.
4. Worker runs the existing pipeline (plan → retrieve → expand → rerank → answer)
   against Neo4j, Ollama, OpenAI, and (optionally) Cohere.
5. The `QAResponse` returns up the chain.
6. Helper `render_qa_response(payload)` builds `(content, elements)`:
   - On success: `content` is the answer; each citation becomes a
     `cl.Text(name="[N] <title|url|node_id>", content=snippet, display="side")`.
   - On `no_results=True`: `content` is the fallback copy, `elements` is empty.
7. Handler sends `cl.Message(content, elements)`. On `httpx` exceptions or non-2xx,
   sends a friendly "agent had a problem" message and logs the detail.

### First-time bring-up

1. `docker compose up` starts all services with health-gated `depends_on`.
2. `neo4j-seed` runs first:
   - If `/data/.seeded` exists → exit 0.
   - Else → `curl -fL --retry 3 --retry-delay 5 "$NEO4J_DUMP_URL" -o /tmp/graph.dump`,
     verify `$NEO4J_DUMP_SHA256`, run `neo4j-admin database load
     --from-path=/tmp --overwrite-destination=true neo4j`, `touch /data/.seeded`.
3. `neo4j` starts only after `neo4j-seed` completes successfully
   (`condition: service_completed_successfully`).
4. In parallel: `ollama` boots, `ollama-seed` waits for `/api/tags` to return 200,
   runs `ollama pull "$EMBEDDING_MODEL"` (idempotent — Ollama itself skips if
   present), exits.
5. `temporal-postgres` → `temporal` → `qa-worker` chain comes up health-gated.
6. `qa-api` starts after Temporal is healthy.
7. `chainlit` starts last; user opens `http://localhost:8000`.

Subsequent `docker compose up` runs skip both seed steps via marker files / Ollama's
own dedup, so restart is fast.

## Components

### Chainlit app (`chat/`)

```
chat/
  pyproject.toml          # chainlit, httpx (runtime); pytest, pytest-asyncio, respx (dev)
  app.py
  Dockerfile
  tests/test_app.py
  .dockerignore
```

`chat/` is a standalone Python project: it owns its own `pyproject.toml` and
dependencies and is independently buildable and testable. The recipient does not
need the QA agent's Python environment to run or test the UI.

`app.py` exposes two pure helpers and one Chainlit handler:

- `async def call_qa_api(client, url, question) -> dict` — `POST /ask` with a
  120-second timeout, raises on non-2xx.
- `def render_qa_response(payload) -> (str, list[cl.Text])` — pure data
  transformation, no Chainlit context required.
- `@cl.on_message async def on_message(message)` — the thin shell that combines
  the two and sends a `cl.Message`. Catches all exceptions from the API call and
  sends a fixed error message.

`@cl.on_chat_start` sends a one-line welcome and nothing else. No session state.

`Dockerfile` is `python:3.11-slim`, `pip install`s from `chat/pyproject.toml`,
`EXPOSE 8000`, runs `chainlit run app.py --host 0.0.0.0 --port 8000 --headless`.

### Q&A agent image (`graphrag-agent-workflow/Dockerfile`)

New file. Builds from `python:3.11-slim`, `pip install -e .` against the existing
`pyproject.toml`. Two compose services consume this image with different
`command:` directives. Image does not include `.env`, `tests/`, or
`ragas_evals/` (excluded via `.dockerignore`).

### Seed containers (`infra/`)

```
infra/
  neo4j-seed/
    Dockerfile                # FROM neo4j:5-community + curl
    seed.sh                   # idempotent download + verify + load
  ollama-seed/
    seed.sh                   # waits on /api/tags then pulls EMBEDDING_MODEL
```

`neo4j-seed/seed.sh` is the load-bearing script:

```bash
#!/usr/bin/env bash
set -euo pipefail
if [ -f /data/.seeded ]; then
  echo "Neo4j volume already seeded, skipping."
  exit 0
fi
curl -fL --retry 3 --retry-delay 5 "$NEO4J_DUMP_URL" -o /tmp/graph.dump
echo "$NEO4J_DUMP_SHA256  /tmp/graph.dump" | sha256sum -c -
neo4j-admin database load --from-path=/tmp --overwrite-destination=true neo4j
touch /data/.seeded
```

The script aborts on any failure; `neo4j` won't start because of the
`service_completed_successfully` dependency, and the recipient sees the cause in
`docker compose logs neo4j-seed`.

`ollama-seed/seed.sh` polls `http://ollama:11434/api/tags` until 200, then runs
`ollama pull "$EMBEDDING_MODEL"` (Ollama dedups internally if the model is
already present).

## Configuration & Secrets

The canonical `.env.example` lives at the **repo root**, alongside
`docker-compose.yml`. The pre-existing `graphrag-agent-workflow/.env.example`
remains in place for the host-Python-only workflow but a top-of-file note
declares it legacy for compose users.

```env
# secrets — recipient must provide
OPENAI_API_KEY=
COHERE_API_KEY=                  # optional; empty triggers RRF top-8 fallback

# dump asset — defaults baked in, override only if mirroring elsewhere
NEO4J_DUMP_URL=https://github.com/MilosJovicic/GraphRAG-workflow/releases/download/neo4j-dump-2026-05-07/neo4j-2026-05-07T13-19-28.dump
NEO4J_DUMP_SHA256=2c47fd8146ad4ee1d94baa65e2453340dc10cabf87470627b5707233fa38715c

# overridable
NEO4J_PASSWORD=changeme-local-only
OPENAI_MODEL=gpt-5.4-mini
EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B:Q8_0
COHERE_RERANK_MODEL=rerank-v3.5
```

### Compose env scoping

Only `qa-api` and `qa-worker` attach the root `.env` via `env_file:`. Every
other service receives only the variables it needs via explicit `environment:`
keys. This keeps `OPENAI_API_KEY` and `COHERE_API_KEY` invisible to Chainlit,
Neo4j, Ollama, Temporal, and the seed containers.

```yaml
qa-api:
  env_file: .env
  environment:
    NEO4J_URI: bolt://neo4j:7687
    NEO4J_USER: neo4j
    NEO4J_PASSWORD: ${NEO4J_PASSWORD}
    TEMPORAL_HOST: temporal:7233
    OLLAMA_BASE_URL: http://ollama:11434/v1   # /v1 — AsyncOpenAI-compatible

qa-worker:
  env_file: .env
  environment: { ... same as qa-api ... }

chainlit:
  environment:
    QA_API_URL: http://qa-api:5000

neo4j:
  environment:
    NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}

neo4j-seed:
  environment:
    NEO4J_DUMP_URL: ${NEO4J_DUMP_URL}
    NEO4J_DUMP_SHA256: ${NEO4J_DUMP_SHA256}

ollama-seed:
  environment:
    EMBEDDING_MODEL: ${EMBEDDING_MODEL}
```

`OLLAMA_BASE_URL` ends in `/v1` because `qa_agent.embeddings` uses
`AsyncOpenAI(base_url=...)`, which expects an OpenAI-compatible root.

`pydantic-settings` already reads from `os.environ` when no `.env` file is
present, so no Python code changes are required.

## Error Handling

| Failure | Detection | Behavior |
| --- | --- | --- |
| Dump download HTTP error | `curl -f --retry 3` | `neo4j-seed` exits non-zero; `neo4j` does not start; visible in `docker compose logs neo4j-seed`. |
| Dump checksum mismatch | `sha256sum -c` | Same as above; protects against corrupted download or tampered URL. |
| Ollama model pull fails | Non-zero exit | `ollama-seed` fails; Ollama itself stays up; first embedding call from `qa-worker` returns the underlying error. Retry with `docker compose up ollama-seed`. |
| Chainlit → `qa-api` 5xx or timeout | `httpx` exception in `call_qa_api` | UI shows fixed message: "The agent ran into a problem. Try again, or check `docker compose logs qa-api`." Detail is logged server-side. No client-side retry. |
| Cohere key missing | Existing agent code | RRF top-8 fallback (current behavior, unchanged). |

Healthchecks are defined for `neo4j`, `temporal`, `ollama`, `qa-api`, and
`chainlit`. `depends_on` uses `condition: service_healthy` (or
`service_completed_successfully` for one-shots) so startup is orderly and the
recipient does not see "Neo4j not ready" thrash.

## Testing

### Unit (automated)

`chat/tests/test_app.py` — three tests covering the pure helpers:

1. `call_qa_api` happy path: mock `httpx.AsyncClient.post` (via `respx` or a
   stub) to return a `QAResponse`-shaped JSON, assert the dict is returned.
2. `render_qa_response` with citations: pass a payload with two citations,
   assert two `cl.Text` elements with the right `name`, `content`, and
   `display="side"` values.
3. `render_qa_response` with `no_results=True`: assert empty elements list and
   the answer string passed through unchanged.

The decorated `@cl.on_message` handler is **not** unit-tested — it requires the
Chainlit runtime context. It is exercised only via the smoke test below.

### Smoke (manual, documented in root `README.md`)

```bash
docker compose config > /dev/null    # 0. validate YAML + env interpolation
cp .env.example .env                  # 1. fill in OPENAI_API_KEY
docker compose up                     # 2. wait for healthchecks green
                                      # 3. open http://localhost:8000
                                      # 4. ask a known-good question
                                      # 5. verify answer + ≥1 side-panel citation
docker compose down && docker compose up
                                      # 6. verify seed steps skip on second up
```

### Out of scope

- Re-testing the existing agent pipeline (covered by
  `graphrag-agent-workflow/tests/`).
- Compose YAML syntax beyond `docker compose config` (compose validates on `up`).
- Shell-script edge cases beyond the marker-file path.
- CI runs that download the real dump asset (couples tests to release
  availability and bandwidth).

## File Layout

```
docker-compose.yml                          # NEW — all services
.env.example                                # NEW — root canonical
.dockerignore                               # NEW — excludes .venv, __pycache__, etc.
README.md                                   # UPDATED — top-level compose quickstart

chat/                                       # NEW — Chainlit app, standalone Python project
  pyproject.toml
  app.py
  Dockerfile
  tests/test_app.py
  .dockerignore

graphrag-agent-workflow/
  Dockerfile                                # NEW — shared image for qa-api + qa-worker
  .env.example                              # UNCHANGED — declared legacy in top-of-file note

infra/                                      # NEW — compose helper assets
  neo4j-seed/
    Dockerfile
    seed.sh
  ollama-seed/
    seed.sh
```

`ETL/` and `contextual-embeddings-agentic-workflow/` are not containerized.
Recipients consume the published Neo4j dump and never run those pipelines.

## Open Questions / Risks

- **Dump size (298 MB):** fits a GitHub release asset (2 GB limit) and the seed
  container's `curl` handles it in one shot. First-time recipients on slow
  connections may wait a few minutes; the seed container logs progress.
- **Neo4j edition compatibility:** the dump was produced by a specific Neo4j
  version. The compose pins `neo4j:5-community` to the same major version. If
  the dump source version is later than the compose image, the README will
  document how to bump the image tag.
- **Embedding model availability:** if Ollama upstream removes
  `dengcao/Qwen3-Embedding-0.6B:Q8_0`, `ollama-seed` fails. Mitigation: the
  model is pinned in `.env.example` and overridable, but the dump's embeddings
  are 1024-dim and any replacement must produce the same dimension.
