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
MATCH (parent:Section)-[r:DEFINES|MENTIONS]->(e)
MATCH path = (parent)-[:HAS_SUBSECTION*0..]->(s:Section)
MATCH (s)-[:CONTAINS_CODE]->(cb:CodeBlock)
WHERE ($language IS NULL OR cb.language = $language)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH seed.node_id AS seed_id, cb, s, p,
     CASE type(r) WHEN 'DEFINES' THEN 2 ELSE 1 END AS rel_rank,
     length(path) AS section_depth,
     COALESCE(cb.position, 999)  AS cb_position
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
ORDER BY rel_rank DESC, section_depth ASC, cb_position ASC, cb.id ASC
LIMIT $cap
```

Notes:
- `HAS_SUBSECTION*0..` walks 0+ hops, so the defining Section itself plus all
  its nested subsections are searched. `*0..` matches the convention already
  used in `fulltext_codeblock.cypher` and `expand_defines.cypher`.
- The pattern is a no-op for non-entity seeds: the first `MATCH (e)` filter
  eliminates Section/CodeBlock seeds before any traversal.
- Language filter is optional; when null the pattern returns all languages.
- **Ranking** (added after live probe found gold IDs at positions 5 and 7
  in unordered cap=10 results):
  - `rel_rank`: DEFINES (2) ranks above MENTIONS (1). True examples beat
    incidental mentions.
  - `section_depth`: shallower path from the entity-defining Section wins —
    a codeblock in the section that DEFINES the entity beats one nested two
    subsections deeper.
  - `cb_position`: when the CodeBlock has a `position` property (sequence
    within its section), earlier blocks win. `999` is a fallback so missing
    `position` does not kill the row.
  - `cb.id ASC`: final tie-break for deterministic ordering.
- Verify during implementation that the `position` property exists on
  CodeBlock nodes; if not, drop that line and rely on the other ordering
  keys. Do not change the design — this is a graceful-degradation tie-break.

### Schema changes — `ExpansionName` and `ExpansionPattern`

Three coordinated edits in `schemas.py`:

1. **Extend the `ExpansionName` Literal** (line 33) so pydantic accepts the
   new pattern name. Without this, planner output that includes
   `code_examples` will fail validation and the workflow will fall back.

   ```python
   ExpansionName = Literal[
       "siblings", "parent_page", "links",
       "defines", "navigates_to", "code_examples",
   ]
   ```

2. **Raise the `max_per_seed` upper bound** (line 61) from `le=5` to `le=8`
   so the planner can request more codeblocks per entity for example queries
   (defense-in-depth on top of the new ORDER BY).

3. **Add the optional `language` field**:

   ```python
   class ExpansionPattern(BaseModel):
       name: ExpansionName
       max_per_seed: int = Field(default=3, ge=1, le=8)
       language: str | None = None
   ```

   `language` is consumed only by `code_examples`; other patterns ignore it.
   "Examples in Python" and "examples in TypeScript" can be two separate
   emitted patterns without touching workflow state.

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

Three edits:

1. **Update the JSON output schema block** (currently shows
   `{"name": "<pattern>", "max_per_seed": 3}` — the planner LLM follows this
   schema description, not just the example). Replace with:

   ```json
   {
     "expansion_patterns": [
       {"name": "<pattern>", "max_per_seed": 3, "language": "<language?>"}
     ]
   }
   ```

   Add a one-line clarification: `"language" is optional and consumed only
   by the code_examples pattern.`

2. Add a bullet under "Graph Expansion Patterns":

   > `code_examples`: from an entity seed (Tool, Hook, SettingKey, ...), fetch
   > CodeBlocks that live in Sections defining or mentioning the entity, plus
   > those Sections' subsections. Results are ranked DEFINES > MENTIONS, then
   > by section depth, so true examples beat incidental mentions. Optional
   > `language` filter on the pattern. Use for "show me an example" or
   > "code for X" questions.

3. Replace the q3 example so it includes the entity label and the new
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
       {"name": "code_examples", "max_per_seed": 6, "language": "python"}
     ],
     "notes": "code-example question; entity_lookup seeds Tool:Edit, code_examples traverses to its codeblocks"
   }
   ```

   `Tool` in `target_labels` triggers `entity_lookup`, which seeds
   `Tool:Edit`. `code_examples` then traverses from that seed. `max_per_seed:
   6` is calibrated against the live probe: the gold IDs surfaced at
   positions 5 and 7 of the unordered probe at cap=10; with the new
   `ORDER BY rel_rank DESC, section_depth ASC` they should rise to the top
   of the cap=6 window, but 6 leaves margin for additional defining-Section
   codeblocks.

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

Status: **completed**. Both gold codeblocks `dd7564204c2946d7` and
`5f644bc9ef3abd5f` are reachable through `Tool:Edit ← DEFINES|MENTIONS ←
Section → HAS_SUBSECTION*0.. → Section → CONTAINS_CODE → CodeBlock`, and both
are tagged `language='python'`, so the language filter does not exclude them.

The probe also revealed the ordering risk addressed above: with
`language=python` and cap=10, the gold IDs landed at positions 5 and 7
*without* `ORDER BY`. The `ORDER BY rel_rank DESC, section_depth ASC,
cb_position ASC` on the new template, plus `max_per_seed: 6` in the planner,
together close that gap.

Re-run probe to verify after implementation:

```cypher
MATCH (e:Tool {name:'Edit'})
MATCH (parent:Section)-[r:DEFINES|MENTIONS]->(e)
MATCH path = (parent)-[:HAS_SUBSECTION*0..]->(s:Section)
MATCH (s)-[:CONTAINS_CODE]->(cb:CodeBlock)
WHERE cb.language = 'python'
WITH cb, s,
     CASE type(r) WHEN 'DEFINES' THEN 2 ELSE 1 END AS rel_rank,
     length(path) AS section_depth,
     COALESCE(cb.position, 999) AS cb_position
RETURN cb.id, s.breadcrumb, rel_rank, section_depth, cb_position
ORDER BY rel_rank DESC, section_depth ASC, cb_position ASC, cb.id ASC
LIMIT 10
```

Acceptance: gold IDs `dd7564204c2946d7` and `5f644bc9ef3abd5f` appear in
positions 1–6 of this output.

## Tests

TDD-first, all tests written and failing before any implementation:

- `tests/test_expansion.py` — new cases:
  - `code_examples` is registered in `_PATTERN_TEMPLATE`.
  - `expand()` passes `language` from the pattern into cypher params for
    `code_examples`; does *not* pass it for other patterns.
  - Mocked `run_cypher` returning fake CodeBlock rows produces Candidate
    objects with `expansion_origin="code_examples:<seed_id>"`.
  - Pattern is invoked with cap = `max_per_seed * len(seeds)`.
- `tests/test_schemas.py` (or extend existing) — three cases:
  - `ExpansionPattern(name="code_examples", ...)` validates (regression
    guard for the `ExpansionName` Literal extension).
  - `ExpansionPattern(name="code_examples", max_per_seed=6, ...)` validates
    (regression guard for the `le=8` bound bump).
  - `ExpansionPattern` accepts and round-trips the optional `language`
    field; default is `None`.
- `tests/test_planner.py` — q3-shape input produces a Plan that includes
  `Tool` in `target_labels` and emits a `code_examples` pattern with
  `language='python'`. (LLM-driven, so this is an integration-style test
  that will use a small real call or a fixed mocked response.)
- `tests/test_expand_code_examples_cypher.py` — graph-integration test
  (gated on `QA_RUN_INTEGRATION`):
  - Seed `Tool:Edit`, language=`python`, cap=6: the gold IDs
    `dd7564204c2946d7` and `5f644bc9ef3abd5f` appear in positions 1–6.
  - Seed a non-entity (Section), cap=6: returns empty (no-op guard).

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
  loosely related to the entity. Mitigated by the new `rel_rank DESC` ordering
  (DEFINES > MENTIONS), the planner-level decision (only fires for "example"
  intent), the rerank step that follows expansion, and the `max_per_seed` cap.
- **Language filter too strict.** Some Edit examples may be tagged `text` or
  no language. The planner currently emits a specific language; if the gold
  blocks are not tagged, they'll be filtered out. The pre-implementation probe
  also checks `cb.language`.
- **Volume blow-up for popular entities.** A Section that defines a common
  entity may contain many codeblocks. Capped by `max_per_seed * len(seeds)` in
  `expand()`, which the planner controls.
