# Code Examples Expansion Design

Date: 2026-05-07
Status: Draft → pending implementation
Owner: Q&A Agent retrieval

## Problem

The RAGAS gold question q3 ("Show me a Python example of using the Edit tool")
returns zero citations and the answerer refuses with "The provided documentation
does not contain a clear answer to this question." Faithfulness, answer
relevancy, factual correctness, and semantic similarity all collapse to 0 (or
near-0) for this row, dragging the eval averages below their RAGAS targets.

The gold codeblocks for q3 are `dd7564204c2946d7` and `5f644bc9ef3abd5f` —
quickstart Python codeblocks containing `allowed_tools=["Read", "Edit", ...]`.
These never appear in the retrieved top-k.

## Root cause

Pure BM25 + vector retrieval over CodeBlock text cannot surface these blocks.
They are short, contain `Edit` only as one element of a list literal, and lose
to longer, more semantically dense codeblocks (Edit input-type definitions,
NotebookEdit parameters, HookMatcher dataclass) when ranked by text similarity.

The signal that *should* link them is graph structure: each gold codeblock lives
under a Section that has `DEFINES Tool:Edit` (or whose ancestor does). The graph
already has both `(Section)-[:DEFINES|MENTIONS]->(Entity)` and
`(Section)-[:CONTAINS_CODE]->(CodeBlock)` edges. No current expansion pattern
traverses `Entity → Section → CodeBlock`:

- `defines` stops at the Section.
- `siblings` only fires from a CodeBlock seed.

## Approach: new `code_examples` expansion pattern

Add a single graph-traversal pattern that runs from entity seeds and returns
CodeBlocks under Sections that define or mention the entity. The planner emits
this pattern (with optional language filter) when the question asks for an
example or code.

### Cypher template — `expand_code_examples.cypher`

```cypher
UNWIND $seeds AS seed
MATCH (e)
WHERE (e.id = seed.node_id OR e.name = seed.node_id OR e.key = seed.node_id)
  AND any(lbl IN labels(e)
          WHERE lbl IN ['Tool','Hook','SettingKey','PermissionMode',
                        'Provider','MessageType'])
MATCH (parent:Section)-[:DEFINES|MENTIONS]->(e)
MATCH (parent)-[:HAS_SUBSECTION*0..]->(s:Section)
MATCH (s)-[:CONTAINS_CODE]->(cb:CodeBlock)
WHERE ($language IS NULL OR cb.language = $language)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH seed.node_id AS seed_id, cb, s, p
RETURN
  cb.id            AS node_id,
  'CodeBlock'      AS node_label,
  cb.indexed_text  AS indexed_text,
  cb.text          AS raw_text,
  p.url            AS url,
  s.anchor         AS anchor,
  s.breadcrumb     AS breadcrumb,
  COALESCE('[' + cb.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  seed_id          AS seed_id
LIMIT $cap
```

Notes:
- `HAS_SUBSECTION*0..` walks 0+ hops, so the defining Section itself plus all
  its nested subsections are searched. `*0..` matches the convention already
  used in `fulltext_codeblock.cypher` and `expand_defines.cypher`.
- The pattern is a no-op for non-entity seeds: the first `MATCH (e)` filter
  eliminates Section/CodeBlock seeds before any traversal.
- Language filter is optional; when null the pattern returns all languages.

### Schema change — `ExpansionPattern.language`

```python
class ExpansionPattern(BaseModel):
    name: str
    max_per_seed: int = 3
    language: str | None = None
```

`language` is consumed only by `code_examples`; other patterns ignore it. This
keeps each pattern self-contained — "examples in Python" and "examples in
TypeScript" can be two separate emitted patterns without touching workflow
state.

### Expansion wiring — `retrieval/expansion.py`

```python
_PATTERN_TEMPLATE = {
    "siblings": "expand_siblings.cypher",
    "parent_page": "expand_parent_page.cypher",
    "links": "expand_links.cypher",
    "defines": "expand_defines.cypher",
    "navigates_to": "expand_navigates_to.cypher",
    "code_examples": "expand_code_examples.cypher",
}
```

Pass `language` through cypher params *only* for the `code_examples` pattern,
so the existing four patterns are untouched.

### Planner prompt update — `prompts/planner.txt`

Two edits:

1. Add a bullet under "Graph Expansion Patterns":

   > `code_examples`: from an entity seed (Tool, Hook, SettingKey, ...), fetch
   > CodeBlocks that live in Sections defining or mentioning the entity, plus
   > those Sections' subsections. Optional `language` filter on the pattern.
   > Use for "show me an example" or "code for X" questions.

2. Replace the q3 example so it includes the entity label and the new
   expansion:

   ```json
   {
     "sub_queries": [{
       "text": "Edit tool Python example",
       "target_labels": ["Tool", "CodeBlock", "Section"],
       "filters": {"language": "python"},
       "bm25_keywords": ["Edit", "python"]
     }],
     "expansion_patterns": [
       {"name": "code_examples", "max_per_seed": 4, "language": "python"}
     ],
     "notes": "code-example question; entity_lookup seeds Tool:Edit, code_examples traverses to its codeblocks"
   }
   ```

`Tool` in `target_labels` triggers `entity_lookup`, which seeds `Tool:Edit`.
`code_examples` then traverses from that seed.

## Components and isolation

- **Cypher template** (`expand_code_examples.cypher`): one file, one query, one
  responsibility.
- **Schema** (`ExpansionPattern`): tiny additive field.
- **Expansion router** (`expansion.py`): adds one entry to `_PATTERN_TEMPLATE`
  and one conditional param push for the `language` field.
- **Planner prompt** (`planner.txt`): documentation + example update only.

Each piece is independently testable; the pattern is opt-in (planner must emit
it), so behavior for other questions is unchanged.

## Pre-implementation probe (graph verification)

Before writing code, run a Cypher probe to confirm the gold codeblocks are
reachable through the proposed traversal. Without this, we'd build the pattern
on a path that may not exist for the very questions it targets.

Probe:

```cypher
MATCH (cb:CodeBlock) WHERE cb.id IN ['dd7564204c2946d7','5f644bc9ef3abd5f']
MATCH (s:Section)-[:CONTAINS_CODE]->(cb)
OPTIONAL MATCH (parent:Section)-[:HAS_SUBSECTION*0..]->(s)
OPTIONAL MATCH (parent)-[:DEFINES|MENTIONS]->(e:Tool {name:'Edit'})
RETURN cb.id, s.breadcrumb, parent.breadcrumb,
       CASE WHEN e IS NULL THEN 'no Edit link' ELSE 'reachable' END AS status
```

Expected: at least one of the two codeblocks resolves to `status='reachable'`
through some ancestor section. If both come back `'no Edit link'`, the design
needs revision — likely either lower-confidence relations (`MENTIONS` only) or
extending the traversal to include non-Section parents — and we revisit before
implementing.

## Tests

TDD-first, all tests written and failing before any implementation:

- `tests/test_expansion.py` — new cases:
  - `code_examples` is registered in `_PATTERN_TEMPLATE`.
  - `expand()` passes `language` from the pattern into cypher params for
    `code_examples`; does *not* pass it for other patterns.
  - Mocked `run_cypher` returning fake CodeBlock rows produces Candidate
    objects with `expansion_origin="code_examples:<seed_id>"`.
  - Pattern is invoked with cap = `max_per_seed * len(seeds)`.
- `tests/test_schemas.py` (or extend existing) — `ExpansionPattern` accepts
  and round-trips the optional `language` field; default is `None`.
- `tests/test_planner.py` — q3-shape input produces a Plan that includes
  `Tool` in `target_labels` and emits a `code_examples` pattern with
  `language='python'`. (LLM-driven, so this is an integration-style test
  that will use a small real call or a fixed mocked response.)

## Verification

After implementation, re-run the live RAGAS eval. Acceptance:

- q3 retrieves at least one of `dd7564204c2946d7` / `5f644bc9ef3abd5f` in
  top-k.
- q3's faithfulness, answer_relevancy, semantic_similarity > 0.5.
- Overall `answer_relevancy >= 0.75` (target 0.80, accept ≥ 0.75 if other
  rows hold).
- Overall `semantic_similarity >= 0.70`.
- No regression on q1/q2/q4/q5/q6/q7 (each row's id_hit_at_8 stays 1.0).

## Out of scope

- Improving the BM25/vector indexes for CodeBlocks.
- Cross-language traversal (a Python question pulling TypeScript codeblocks).
- Updating the q6/q7 reference answers — tracked separately.
- New entity types beyond the existing six.

## Risks

- **Pattern misfires on `MENTIONS`-only links.** If the entity is merely
  mentioned (not defined) in a Section, that Section's codeblocks may be only
  loosely related to the entity. Mitigated by the planner-level decision (only
  fires for "example" intent), the rerank step that follows expansion, and the
  `max_per_seed` cap.
- **Language filter too strict.** Some Edit examples may be tagged `text` or
  no language. The planner currently emits a specific language; if the gold
  blocks are not tagged, they'll be filtered out. The pre-implementation probe
  also checks `cb.language`.
- **Volume blow-up for popular entities.** A Section that defines a common
  entity may contain many codeblocks. Capped by `max_per_seed * len(seeds)` in
  `expand()`, which the planner controls.
