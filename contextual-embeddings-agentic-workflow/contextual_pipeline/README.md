# Contextual Pipeline

Temporal pipeline for populating Neo4j documentation graph nodes with:

- `context`
- `indexed_text`
- `embedding`
- `context_source`
- `context_version`

The pipeline is restartable. Writes are idempotent `SET` operations keyed by node label and primary key, so re-running the same batch overwrites the same final properties.

## Environment

Run commands from `contextual_pipeline/` with the project virtualenv:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Configuration is loaded from the parent `.env` file.

Required variables:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `ANTHROPIC_API_KEY`
- `LLM_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `TEMPORAL_HOST`
- `TEMPORAL_NAMESPACE`
- `CONTEXT_VERSION`

## Worker

Start the worker:

```powershell
cd "d:\Ecompany GraphRAG\Contextual enrichement\contextual_pipeline"
..\.venv\Scripts\python.exe -m src.worker
```

After code changes, stop and restart any existing worker before starting another workflow.

## Pilot Run

Run a 50-node CodeBlock pilot:

```powershell
cd "d:\Ecompany GraphRAG\Contextual enrichement\contextual_pipeline"
..\.venv\Scripts\python.exe -m src.starter --pilot 50 --labels CodeBlock
```

Verification queries:

```cypher
MATCH (c:CodeBlock) WHERE c.indexed_text IS NOT NULL
RETURN count(c) AS count,
       avg(size(c.indexed_text)) AS avg_indexed_text_len,
       avg(size(c.context)) AS avg_context_len
```

```cypher
MATCH (c:CodeBlock) WHERE c.embedding IS NOT NULL
RETURN size(c.embedding) AS dim, count(*) AS count
LIMIT 1
```

```cypher
CALL db.index.fulltext.queryNodes('codeblock_fulltext', 'permission')
YIELD node, score
RETURN node.id, score
LIMIT 5
```

```cypher
MATCH (c:CodeBlock) WHERE c.embedding IS NOT NULL
WITH c LIMIT 1
CALL db.index.vector.queryNodes('codeblock_embedding', 5, c.embedding)
YIELD node, score
RETURN node.id, score
```

## Full Run

Run all configured labels:

```powershell
cd "d:\Ecompany GraphRAG\Contextual enrichement\contextual_pipeline"
..\.venv\Scripts\python.exe -m src.starter
```

Resume only missing contexts:

```powershell
..\.venv\Scripts\python.exe -m src.starter --resume missing_only
```

Re-process only LLM-generated context:

```powershell
..\.venv\Scripts\python.exe -m src.starter --resume by_source --sources llm
```

## Index Coverage

Fulltext (`<label>_fulltext` on `indexed_text`) and vector (`<label>_embedding` on `embedding`, 1024 dims, COSINE, HNSW) indexes exist for every label this pipeline processes:

- `Section`, `CodeBlock`, `TableRow`, `Callout`
- `Tool`, `Hook`, `SettingKey`, `PermissionMode`, `MessageType`, `Provider`

`Chunk` indexes also exist but `Chunk` nodes are not processed by this pipeline.

## Tests

Run all tests:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Compile check:

```powershell
..\.venv\Scripts\python.exe -m compileall -q src tests scripts
```
