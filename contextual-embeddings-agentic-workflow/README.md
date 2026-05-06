# Contextual Embeddings Agentic Workflow

This segment contains the agentic contextual enrichment pipeline for the GraphRAG Workflow project.

The imported implementation lives in [`contextual_pipeline/`](contextual_pipeline/). It uses Temporal workflows and activities to populate Neo4j graph nodes with contextual text, indexed text, embeddings, and provenance metadata.

Start here:

- [`contextual_pipeline/README.md`](contextual_pipeline/README.md) for setup, worker, pilot run, full run, and verification commands.
- [`contextual_pipeline/pyproject.toml`](contextual_pipeline/pyproject.toml) for Python dependencies.
- [`contextual_pipeline/tests/`](contextual_pipeline/tests/) for the contract and safety test suite.

Local-only files such as `.env`, `CLAUDE.md`, virtual environments, caches, and run logs are intentionally excluded from this public repository.
