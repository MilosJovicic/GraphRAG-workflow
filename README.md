# GraphRAG Workflow

GraphRAG Workflow is a monorepo for a three-segment retrieval and agent workflow:

1. `ETL/`
2. `contextual-embeddings-agentic-workflow/`
3. `graphrag-agent-workflow/`

## Current Status

The first import contains the ETL segment. It includes the pipeline code, checks, tests, schema, source documents, and generated ETL artifacts currently needed by the workflow.

The contextual embeddings agentic workflow and GraphRAG agent workflow are reserved as top-level segments and will be added in later imports.

## Repository Hygiene

This repository intentionally excludes local-only and sensitive files such as `.env`, `.env.*`, `CLAUDE.md`, virtual environments, caches, editor settings, and generated Python bytecode.
