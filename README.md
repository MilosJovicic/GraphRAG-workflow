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
  "citations": [{"id": 1, "node_id": "...", "url": "..."}],
  "plan": {},
  "latency_ms": {},
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
