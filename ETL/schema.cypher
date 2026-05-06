// Claude Code Docs -> Neo4j GraphRAG schema
// Authoritative. If this file and CLAUDE.md disagree, this file wins.
// Apply constraints and lookup indexes before loading data, then apply vector
// and full-text indexes after embeddings/indexed_text are written. This file is
// safe to re-run because all schema operations use IF NOT EXISTS.
//
// Run order:
//   1. Constraints  (idempotent; create before any MERGE)
//   2. Bulk load    (Stage 6 in CLAUDE.md)
//   3. Indexes      (vector + full-text; create AFTER bulk load)

// ---------------------------------------------------------------------------
// Uniqueness constraints
// ---------------------------------------------------------------------------

CREATE CONSTRAINT page_url IF NOT EXISTS
  FOR (p:Page) REQUIRE p.url IS UNIQUE;

CREATE CONSTRAINT section_id IF NOT EXISTS
  FOR (s:Section) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
  FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT codeblock_id IF NOT EXISTS
  FOR (b:CodeBlock) REQUIRE b.id IS UNIQUE;

CREATE CONSTRAINT tablerow_id IF NOT EXISTS
  FOR (r:TableRow) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT callout_id IF NOT EXISTS
  FOR (c:Callout) REQUIRE c.id IS UNIQUE;

// Closed-vocabulary entity catalogs (Stage 3)
CREATE CONSTRAINT tool_name IF NOT EXISTS
  FOR (t:Tool) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT setting_key IF NOT EXISTS
  FOR (s:SettingKey) REQUIRE s.key IS UNIQUE;

CREATE CONSTRAINT provider_name IF NOT EXISTS
  FOR (p:Provider) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT messagetype_name IF NOT EXISTS
  FOR (m:MessageType) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT hook_name IF NOT EXISTS
  FOR (h:Hook) REQUIRE h.name IS UNIQUE;

CREATE CONSTRAINT permissionmode_name IF NOT EXISTS
  FOR (m:PermissionMode) REQUIRE m.name IS UNIQUE;

// ---------------------------------------------------------------------------
// Existence guarantees on properties used by retrieval
// (only on labels we know carry these props)
// ---------------------------------------------------------------------------

CREATE CONSTRAINT page_path_exists IF NOT EXISTS
  FOR (p:Page) REQUIRE p.path IS NOT NULL;

CREATE CONSTRAINT page_content_hash_exists IF NOT EXISTS
  FOR (p:Page) REQUIRE p.content_hash IS NOT NULL;

CREATE CONSTRAINT section_anchor_exists IF NOT EXISTS
  FOR (s:Section) REQUIRE s.anchor IS NOT NULL;

CREATE CONSTRAINT section_level_exists IF NOT EXISTS
  FOR (s:Section) REQUIRE s.level IS NOT NULL;

// ---------------------------------------------------------------------------
// Lookup indexes (cheap; create up front)
// ---------------------------------------------------------------------------

CREATE INDEX page_path IF NOT EXISTS
  FOR (p:Page) ON (p.path);

CREATE INDEX page_volatile IF NOT EXISTS
  FOR (p:Page) ON (p.volatile);

CREATE INDEX section_page IF NOT EXISTS
  FOR (s:Section) ON (s.page_url);

CREATE INDEX chunk_section IF NOT EXISTS
  FOR (c:Chunk) ON (c.section_id);

CREATE INDEX codeblock_lang IF NOT EXISTS
  FOR (b:CodeBlock) ON (b.language);

// ---------------------------------------------------------------------------
// Vector indexes (Stage 5; create AFTER embeddings are written)
// Embedding dimension: 1024. Update the literal below if the embedding model
// changes.
// Similarity: cosine
// ---------------------------------------------------------------------------

CREATE VECTOR INDEX section_embedding IF NOT EXISTS
  FOR (s:Section) ON s.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
  FOR (c:Chunk) ON c.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX codeblock_embedding IF NOT EXISTS
  FOR (b:CodeBlock) ON b.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX tablerow_embedding IF NOT EXISTS
  FOR (r:TableRow) ON r.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX callout_embedding IF NOT EXISTS
  FOR (c:Callout) ON c.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

// ---------------------------------------------------------------------------
// Full-text indexes (Stage 5; same indexed_text as the vector index)
// ---------------------------------------------------------------------------

CREATE FULLTEXT INDEX section_fulltext IF NOT EXISTS
  FOR (s:Section) ON EACH [s.indexed_text];

CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
  FOR (c:Chunk) ON EACH [c.indexed_text];

CREATE FULLTEXT INDEX codeblock_fulltext IF NOT EXISTS
  FOR (b:CodeBlock) ON EACH [b.indexed_text];

CREATE FULLTEXT INDEX tablerow_fulltext IF NOT EXISTS
  FOR (r:TableRow) ON EACH [r.indexed_text];

CREATE FULLTEXT INDEX callout_fulltext IF NOT EXISTS
  FOR (c:Callout) ON EACH [c.indexed_text];

// ---------------------------------------------------------------------------
// Relationship types (declarative — Neo4j infers from MERGE, but listed here
// as the canonical set so any new edge type is a deliberate schema change)
// ---------------------------------------------------------------------------
//
//   (Page)-[:HAS_SECTION]->(Section)
//   (Section)-[:HAS_SUBSECTION]->(Section)
//   (Section)-[:HAS_CHUNK]->(Chunk)
//   (Section)-[:CONTAINS_CODE]->(CodeBlock)
//   (Section)-[:CONTAINS_TABLE_ROW]->(TableRow)
//   (Section)-[:HAS_CALLOUT]->(Callout)
//   (Section)-[:LINKS_TO]->(Section)        // intra-corpus, with anchor
//   (Section)-[:LINKS_TO_PAGE]->(Page)      // intra-corpus, page-level
//   (Section)-[:NAVIGATES_TO]->(Section)    // from navigation tables
//   (CodeBlock)-[:EQUIVALENT_TO]->(CodeBlock)  // <CodeGroup> siblings
//   (Section)-[:MENTIONS]->(Tool|SettingKey|Provider|MessageType|Hook|PermissionMode)
//   (Section|TableRow)-[:DEFINES]->(Tool|SettingKey|Provider|MessageType|Hook|PermissionMode)
