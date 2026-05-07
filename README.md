# GraphRAG Workflow

GraphRAG Workflow is a monorepo for an end-to-end retrieval and agent workflow.
It is organized into three top-level segments: document ETL, contextual
embedding preparation, and a GraphRAG Q&A agent.

## Repository Map

```mermaid
flowchart LR
    repo["GraphRAG-workflow"]
    etl["ETL<br/>Parse, enrich, resolve, and export source documents"]
    contextual["contextual-embeddings-agentic-workflow<br/>Prepare contextual embeddings workflow assets"]
    agent["graphrag-agent-workflow<br/>Hybrid GraphRAG Q&A agent with eval harness"]

    repo --> etl
    repo --> contextual
    repo --> agent
```

## Workflow

```mermaid
flowchart TD
    docs["Source documents"] --> etl["ETL pipeline"]
    etl --> graph["Graph-ready artifacts"]
    graph --> contextual["Contextual embedding workflow"]
    contextual --> indexes["Vector and full-text retrieval indexes"]
    indexes --> agent["GraphRAG Q&A agent"]
    graph --> agent
    agent --> evals["RAGAS evaluation harness"]
```

## Segments

| Segment | Purpose |
| --- | --- |
| [`ETL/`](ETL/) | Pipeline code, checks, source documents, parsed outputs, resolved entities, and graph-ready artifacts. |
| [`contextual-embeddings-agentic-workflow/`](contextual-embeddings-agentic-workflow/) | Contextual embedding workflow assets used between document processing and retrieval. |
| [`graphrag-agent-workflow/`](graphrag-agent-workflow/) | GraphRAG Q&A agent with Temporal workflow wiring, hybrid retrieval, entity lookup, graph expansion, reranking, tests, and RAGAS evaluation support. |

## Agent Retrieval Path

```mermaid
sequenceDiagram
    participant User
    participant API as Q&A API
    participant Temporal
    participant Retrieval
    participant Graph as Neo4j Graph
    participant LLM as Answerer

    User->>API: Ask a documentation question
    API->>Temporal: Start QA workflow
    Temporal->>Retrieval: Plan, search, expand, rerank
    Retrieval->>Graph: BM25, vector, entity, and graph traversal queries
    Graph-->>Retrieval: Candidate evidence
    Retrieval-->>Temporal: Ranked evidence
    Temporal->>LLM: Generate grounded answer with citations
    LLM-->>API: Answer and citation IDs
    API-->>User: Response
```
