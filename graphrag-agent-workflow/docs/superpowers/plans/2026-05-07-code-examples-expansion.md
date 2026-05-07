# Code Examples Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `code_examples` graph-expansion pattern that fetches CodeBlocks under Sections defining or mentioning an entity, so q3-style "show me an example" questions retrieve concrete code rather than refusing.

**Architecture:** New Cypher template traverses `Entity ← DEFINES|MENTIONS ← Section → HAS_SUBSECTION*0.. → Section → CONTAINS_CODE → CodeBlock`, ranked DEFINES > MENTIONS > section depth > position. The planner emits the pattern with an optional language filter when the question asks for an example; the entity is seeded by the existing `entity_lookup`. Schema and planner prompt are updated together so the new field validates and the LLM emits it.

**Tech Stack:** Python 3.11, pydantic, neo4j-python-driver, pydantic-ai, pytest, pytest-asyncio.

**Spec:** [docs/superpowers/specs/2026-05-07-code-examples-expansion-design.md](../specs/2026-05-07-code-examples-expansion-design.md)

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/qa_agent/schemas.py` | Modify | Extend `ExpansionName` Literal, raise `max_per_seed` upper bound to 8, add `language: str \| None` to `ExpansionPattern`. |
| `src/qa_agent/cypher/expand_code_examples.cypher` | Create | Graph traversal from entity seed → defining/mentioning Section → subsections → CodeBlock, with language filter and ranking. |
| `src/qa_agent/retrieval/expansion.py` | Modify | Register `code_examples` in `_PATTERN_TEMPLATE`; pass `language` cypher param. |
| `src/qa_agent/prompts/planner.txt` | Modify | Update JSON output schema to expose optional `language`; add bullet documenting `code_examples`; replace q3 example. |
| `tests/test_schemas.py` | Modify | Add validation tests for the Literal extension, `le=8` bound, and `language` round-trip. |
| `tests/test_expansion.py` | Modify | Add unit tests for `code_examples` registration, `language` cypher-param passthrough, and `expansion_origin` formatting. |
| `tests/test_planner_prompt.py` | Create | Static-content tests on `planner.txt`: schema block shows `language`, bullet exists, q3 example uses `code_examples` and `Tool`. |
| `tests/test_expand_code_examples_cypher.py` | Create | Graph-integration test (gated on `QA_RUN_INTEGRATION`): seeds `Tool:Edit`, verifies gold IDs in top 6. |

---

## Task 1: Extend `ExpansionName` Literal to include `code_examples`

**Files:**
- Modify: `src/qa_agent/schemas.py:33`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas.py` (after `test_expansion_pattern_unknown_name_rejected`):

```python
def test_expansion_pattern_accepts_code_examples_name():
    pattern = ExpansionPattern(name="code_examples", max_per_seed=4)
    assert pattern.name == "code_examples"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py::test_expansion_pattern_accepts_code_examples_name -v`
Expected: FAIL with `ValidationError` — input should be one of `'siblings', 'parent_page', 'links', 'defines', 'navigates_to'`.

- [ ] **Step 3: Extend the Literal**

Replace line 33 of `src/qa_agent/schemas.py`:

```python
ExpansionName = Literal[
    "siblings",
    "parent_page",
    "links",
    "defines",
    "navigates_to",
    "code_examples",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py::test_expansion_pattern_accepts_code_examples_name -v`
Expected: PASS.

Also run the existing schema suite to confirm no regression:

Run: `pytest tests/test_schemas.py -v`
Expected: all PASS, including `test_expansion_pattern_unknown_name_rejected` (still rejects `"bogus"`).

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): allow code_examples ExpansionName"
```

---

## Task 2: Raise `max_per_seed` upper bound from 5 to 8

**Files:**
- Modify: `src/qa_agent/schemas.py:61`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schemas.py`:

```python
def test_expansion_pattern_max_per_seed_accepts_8():
    pattern = ExpansionPattern(name="code_examples", max_per_seed=8)
    assert pattern.max_per_seed == 8


def test_expansion_pattern_max_per_seed_rejects_9():
    with pytest.raises(ValidationError):
        ExpansionPattern(name="code_examples", max_per_seed=9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py::test_expansion_pattern_max_per_seed_accepts_8 tests/test_schemas.py::test_expansion_pattern_max_per_seed_rejects_9 -v`
Expected: `accepts_8` FAILs (`ValidationError: input less than or equal to 5`), `rejects_9` PASSes (already rejected by `le=5`).

- [ ] **Step 3: Bump the bound**

Edit `src/qa_agent/schemas.py` line 61. Change:

```python
class ExpansionPattern(BaseModel):
    name: ExpansionName
    max_per_seed: int = Field(default=3, ge=1, le=5)
```

To:

```python
class ExpansionPattern(BaseModel):
    name: ExpansionName
    max_per_seed: int = Field(default=3, ge=1, le=8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: all PASS, including the existing `test_expansion_pattern_max_per_seed_bounds` (which checks `max_per_seed=10` is still rejected — it is, because 10 > 8).

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): raise ExpansionPattern.max_per_seed upper bound to 8"
```

---

## Task 3: Add optional `language` field to `ExpansionPattern`

**Files:**
- Modify: `src/qa_agent/schemas.py:59-61`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schemas.py`:

```python
def test_expansion_pattern_language_default_is_none():
    pattern = ExpansionPattern(name="code_examples")
    assert pattern.language is None


def test_expansion_pattern_language_round_trip():
    pattern = ExpansionPattern(
        name="code_examples", max_per_seed=6, language="python"
    )
    assert pattern.language == "python"
    dumped = pattern.model_dump()
    rehydrated = ExpansionPattern.model_validate(dumped)
    assert rehydrated.language == "python"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py::test_expansion_pattern_language_default_is_none tests/test_schemas.py::test_expansion_pattern_language_round_trip -v`
Expected: both FAIL with `AttributeError: 'ExpansionPattern' object has no attribute 'language'`.

- [ ] **Step 3: Add the field**

Edit `src/qa_agent/schemas.py`. Replace the class:

```python
class ExpansionPattern(BaseModel):
    name: ExpansionName
    max_per_seed: int = Field(default=3, ge=1, le=8)
    language: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add optional language field to ExpansionPattern"
```

---

## Task 4: Write the Cypher template `expand_code_examples.cypher`

**Files:**
- Create: `src/qa_agent/cypher/expand_code_examples.cypher`

- [ ] **Step 1: Verify the template directory and create the file**

Run: `ls "src/qa_agent/cypher/" | head -5`
Expected: shows existing `expand_*.cypher` files.

Create `src/qa_agent/cypher/expand_code_examples.cypher` with exactly this content:

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
     COALESCE(cb.position, 999) AS cb_position
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

- [ ] **Step 2: Static safety check**

Run: `pytest tests/test_cypher_safety.py -v`
Expected: PASS. (This file enumerates `src/qa_agent/cypher/` and applies repo-wide cypher safety checks.)

If the new file fails any check, fix the template. If `test_cypher_safety.py` does not auto-discover new files, skip this step and rely on Task 9's integration test instead.

- [ ] **Step 3: Commit**

```bash
git add src/qa_agent/cypher/expand_code_examples.cypher
git commit -m "feat(cypher): add expand_code_examples template"
```

---

## Task 5: Register `code_examples` in expansion router

**Files:**
- Modify: `src/qa_agent/retrieval/expansion.py:8-14`
- Modify: `tests/test_expansion.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expansion.py`:

```python
@pytest.mark.asyncio
async def test_expand_dispatches_code_examples_to_correct_template():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [ExpansionPattern(name="code_examples", max_per_seed=4)]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    assert args[0] == "expand_code_examples.cypher"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_expansion.py::test_expand_dispatches_code_examples_to_correct_template -v`
Expected: FAIL — the pattern is silently dropped (`run_cypher` is never called) because `code_examples` is not in `_PATTERN_TEMPLATE`. The assertion on `mocked_run.call_args` raises `AttributeError: 'NoneType' object has no attribute '__getitem__'`.

- [ ] **Step 3: Add the registration**

Edit `src/qa_agent/retrieval/expansion.py` lines 8–14. Replace:

```python
_PATTERN_TEMPLATE = {
    "siblings": "expand_siblings.cypher",
    "parent_page": "expand_parent_page.cypher",
    "links": "expand_links.cypher",
    "defines": "expand_defines.cypher",
    "navigates_to": "expand_navigates_to.cypher",
}
```

With:

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_expansion.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/expansion.py tests/test_expansion.py
git commit -m "feat(expansion): register code_examples pattern"
```

---

## Task 6: Pass `language` cypher param through `expand()`

**Files:**
- Modify: `src/qa_agent/retrieval/expansion.py:53-64`
- Modify: `tests/test_expansion.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expansion.py`:

```python
@pytest.mark.asyncio
async def test_expand_passes_language_param_for_code_examples():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [
        ExpansionPattern(name="code_examples", max_per_seed=6, language="python")
    ]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    assert args[0] == "expand_code_examples.cypher"
    params = args[1]
    assert params["language"] == "python"
    assert params["cap"] == 6


@pytest.mark.asyncio
async def test_expand_passes_null_language_when_unset():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [ExpansionPattern(name="code_examples", max_per_seed=4)]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await expand(seeds, patterns, total_cap=20)

    args, _ = mocked_run.call_args
    params = args[1]
    assert params["language"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_expansion.py::test_expand_passes_language_param_for_code_examples tests/test_expansion.py::test_expand_passes_null_language_when_unset -v`
Expected: both FAIL with `KeyError: 'language'` — current `expand()` only passes `seeds` and `cap`.

- [ ] **Step 3: Pass `language` in cypher params**

Edit `src/qa_agent/retrieval/expansion.py`. Replace the loop in `expand()` (the body currently at lines 53–73) with:

```python
    for pattern in patterns:
        template = _PATTERN_TEMPLATE.get(pattern.name)
        if not template:
            continue

        rows = await run_cypher(
            template,
            {
                "seeds": seed_payload,
                "cap": pattern.max_per_seed * len(seeds),
                "language": pattern.language,
            },
        )
        for row in rows:
            candidate = _row_to_candidate(row, pattern.name)
            key = (candidate.node_id, candidate.node_label)
            if key in seen:
                continue
            seen.add(key)
            expansions.append(candidate)
            if len(expansions) >= total_cap:
                break

        if len(expansions) >= total_cap:
            break
```

The other expansion templates do not reference `$language`; passing the param is harmless (Cypher tolerates unused parameters; this matches `bm25.py`'s convention of always passing the full filter dict).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_expansion.py -v`
Expected: all PASS — the new tests AND the original three (which now also receive a `language` key, but they did not assert on its absence).

- [ ] **Step 5: Commit**

```bash
git add src/qa_agent/retrieval/expansion.py tests/test_expansion.py
git commit -m "feat(expansion): pass language cypher param through expand()"
```

---

## Task 7: Update planner prompt — output schema + bullet + q3 example

**Files:**
- Modify: `src/qa_agent/prompts/planner.txt`
- Create: `tests/test_planner_prompt.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_planner_prompt.py`:

```python
from pathlib import Path

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "qa_agent"
    / "prompts"
    / "planner.txt"
)


def _read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_prompt_output_schema_block_documents_optional_language():
    prompt = _read_prompt()
    assert '"language": "<language?>"' in prompt
    assert (
        '"language" is optional and consumed only by the code_examples pattern.'
        in prompt
    )


def test_prompt_documents_code_examples_pattern():
    prompt = _read_prompt()
    assert "- code_examples:" in prompt or "`code_examples`" in prompt
    assert "DEFINES" in prompt
    assert "MENTIONS" in prompt


def test_prompt_q3_example_uses_tool_label_and_code_examples_pattern():
    prompt = _read_prompt()
    q3_marker = "Show me a Python example of using the Edit tool."
    assert q3_marker in prompt
    after_q3 = prompt.split(q3_marker, 1)[1]
    next_user = after_q3.find("\nUser:")
    block = after_q3 if next_user == -1 else after_q3[:next_user]
    assert '"Tool"' in block
    assert '"code_examples"' in block
    assert '"language": "python"' in block
    assert '"max_per_seed": 6' in block
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planner_prompt.py -v`
Expected: all three FAIL — the strings do not appear yet.

- [ ] **Step 3: Update the prompt — output schema block**

Edit `src/qa_agent/prompts/planner.txt`. Replace the schema block (around lines 38–53) with:

```
You must produce JSON matching this shape exactly:

{
  "sub_queries": [
    {
      "text": "<cleaned/expanded search query>",
      "target_labels": ["<NodeLabel>", "..."],
      "filters": {"language": "...", "page_path_prefix": "..."},
      "bm25_keywords": ["<keyword>", "..."]
    }
  ],
  "expansion_patterns": [
    {"name": "<pattern>", "max_per_seed": 3, "language": "<language?>"}
  ],
  "notes": "<one-line rationale>"
}

"language" is optional and consumed only by the code_examples pattern.
```

- [ ] **Step 4: Update the prompt — graph expansion patterns bullet**

In the same file, find the "Graph Expansion Patterns" section. Append (after the `navigates_to` bullet) this new bullet:

```
- code_examples: from an entity seed (Tool, Hook, SettingKey, PermissionMode, Provider, MessageType), fetch CodeBlocks that live in Sections defining or mentioning the entity, plus those Sections' subsections. Results are ranked DEFINES > MENTIONS, then by section depth, so true examples beat incidental mentions. Optional language filter on the pattern. Use for "show me an example" or "code for X" questions.
```

- [ ] **Step 5: Update the prompt — replace q3 example**

Find the q3 example block in the same file (starts with `User: Show me a Python example of using the Edit tool.`). Replace the entire JSON block that follows it with:

```
User: Show me a Python example of using the Edit tool.

{
  "sub_queries": [
    {
      "text": "Edit tool Python example",
      "target_labels": ["Tool", "CodeBlock", "Section"],
      "filters": {"language": "python"},
      "bm25_keywords": ["Edit", "python"]
    }
  ],
  "expansion_patterns": [
    {"name": "code_examples", "max_per_seed": 6, "language": "python"}
  ],
  "notes": "code-example question; entity_lookup seeds Tool:Edit, code_examples traverses to its codeblocks"
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_planner_prompt.py -v`
Expected: all PASS.

Also run the existing planner tests to confirm no regression:

Run: `pytest tests/test_planner.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qa_agent/prompts/planner.txt tests/test_planner_prompt.py
git commit -m "feat(planner): document and emit code_examples for example queries"
```

---

## Task 8: Add `expansion_origin` formatting test for `code_examples`

**Files:**
- Modify: `tests/test_expansion.py`

This is a separate guard for the `expansion_origin="code_examples:<seed_id>"` invariant — the rerank step uses this to identify lookup vs expansion candidates (see [src/qa_agent/retrieval/rerank.py:32-37](../../../src/qa_agent/retrieval/rerank.py#L32-L37)).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expansion.py`:

```python
@pytest.mark.asyncio
async def test_expand_code_examples_marks_origin_with_seed_id():
    seeds = [_seed("Tool:Edit", label="Tool")]
    patterns = [
        ExpansionPattern(name="code_examples", max_per_seed=4, language="python")
    ]
    rows = [
        {
            "node_id": "dd7564204c2946d7",
            "node_label": "CodeBlock",
            "indexed_text": "[python] allowed_tools=['Edit']",
            "raw_text": "allowed_tools=['Edit']",
            "url": None,
            "anchor": None,
            "breadcrumb": "Quickstart > Step 3",
            "title": "[python] Quickstart > Step 3",
            "seed_id": "Tool:Edit",
        },
    ]

    with patch(
        "qa_agent.retrieval.expansion.run_cypher",
        new=AsyncMock(return_value=rows),
    ):
        out = await expand(seeds, patterns, total_cap=20)

    cb = next(c for c in out if c.node_id == "dd7564204c2946d7")
    assert cb.expansion_origin == "code_examples:Tool:Edit"
    assert cb.node_label == "CodeBlock"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_expansion.py::test_expand_code_examples_marks_origin_with_seed_id -v`

This test will already pass because Tasks 5–6 wired the dispatch and `_row_to_candidate` already prefixes `f"{pattern}:{seed_id}"` (`expansion.py:27`). If it fails, do not write new code — debug whichever earlier task is at fault.

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_expansion.py
git commit -m "test(expansion): guard expansion_origin format for code_examples"
```

---

## Task 9: Graph-integration test — gold IDs land in top 6

**Files:**
- Create: `tests/test_expand_code_examples_cypher.py`

This test runs against a live Neo4j and is gated on `QA_RUN_INTEGRATION=1`, matching the convention in `tests/test_retrieval_smoke.py`. It is the acceptance test for Risk #2 (language filter too strict) and the live-probe finding from the spec (gold IDs at positions 5/7 unordered → must move into top 6 ordered).

- [ ] **Step 1: Confirm the integration-gate convention**

Run: `grep -l "QA_RUN_INTEGRATION" tests/`
Expected: at least `tests/test_retrieval_smoke.py` listed.

Run: `head -30 tests/test_retrieval_smoke.py`
Expected: shows the gating idiom (typically `pytestmark = pytest.mark.skipif(...)` or a per-test marker on `os.environ.get("QA_RUN_INTEGRATION") != "1"`). Reuse that exact pattern.

- [ ] **Step 2: Write the failing test**

Create `tests/test_expand_code_examples_cypher.py`:

```python
import os

import pytest

from qa_agent.neo4j_client import run_cypher

pytestmark = pytest.mark.skipif(
    os.environ.get("QA_RUN_INTEGRATION") != "1",
    reason="requires live Neo4j with the Claude Code docs graph",
)

GOLD_IDS = {"dd7564204c2946d7", "5f644bc9ef3abd5f"}


@pytest.mark.asyncio
async def test_code_examples_returns_gold_edit_codeblocks_in_top_6():
    rows = await run_cypher(
        "expand_code_examples.cypher",
        {
            "seeds": [{"node_id": "Edit", "node_label": "Tool"}],
            "cap": 6,
            "language": "python",
        },
    )
    ids = [row["node_id"] for row in rows]
    hits = GOLD_IDS.intersection(ids)
    assert hits, (
        f"Neither gold codeblock found in top 6 results. Got: {ids}"
    )


@pytest.mark.asyncio
async def test_code_examples_is_noop_for_non_entity_seed():
    rows = await run_cypher(
        "expand_code_examples.cypher",
        {
            "seeds": [
                {
                    "node_id": "https://code.claude.com/docs/en/quickstart.md",
                    "node_label": "Section",
                }
            ],
            "cap": 6,
            "language": "python",
        },
    )
    assert rows == []
```

- [ ] **Step 3: Run test to verify it passes**

This task adds no production code — Tasks 4–7 already shipped the template, dispatch, and language wiring. Run the test against the live graph:

```bash
QA_RUN_INTEGRATION=1 pytest tests/test_expand_code_examples_cypher.py -v
```

Expected: both PASS.

If `test_code_examples_returns_gold_edit_codeblocks_in_top_6` fails:
- First, drop the `cb_position` line from the Cypher template (graceful degradation noted in spec) and re-run.
- If still failing, raise `cap` to 10 in the test temporarily and inspect `rows` — the issue is either the seed not matching `Tool:Edit` (check `e.name = seed.node_id`) or the gold codeblocks not having `cb.language='python'`. Do not relax the acceptance criteria; root-cause the path.

If `test_code_examples_is_noop_for_non_entity_seed` fails: the entity-label guard in the Cypher (`any(lbl IN labels(e) WHERE lbl IN [...])`) is matching a Section. Inspect the seed-id resolution and tighten.

- [ ] **Step 4: Commit**

```bash
git add tests/test_expand_code_examples_cypher.py
git commit -m "test(integration): code_examples surfaces gold Edit codeblocks"
```

---

## Task 10: Run full unit suite

**Files:** none modified.

- [ ] **Step 1: Run unit tests**

```bash
pytest tests/ -q --ignore=tests/test_expand_code_examples_cypher.py --ignore=tests/test_retrieval_smoke.py
```

Expected: all PASS.

- [ ] **Step 2: Run cypher safety**

```bash
pytest tests/test_cypher_safety.py -v
```

Expected: PASS, including the new template.

- [ ] **Step 3: Workflow replay**

```bash
pytest tests/test_workflow_replay.py -v
```

Expected: PASS — the QAWorkflow's behavior must remain deterministic. Schema additions are backward-compatible (new field defaults to `None`), so existing replay histories still validate.

If any test fails: do **not** mark the plan complete. Debug systematically (use the systematic-debugging skill) before proceeding.

- [ ] **Step 4: No commit needed (verification only)**

---

## Task 11: Restart Temporal/worker and run live RAGAS eval

**Files:** none modified.

- [ ] **Step 1: Confirm services**

```bash
python -c "import socket; s=socket.create_connection(('localhost',7233),timeout=2); s.close(); print('Temporal OK')"
```

Expected: `Temporal OK`. If not, restart Temporal:

```bash
"C:/Users/User/.temporalio/bin/temporal.exe" server start-dev --headless --log-level error --db-filename "$(pwd)/.tmp_temporal.db" &
```

Then start the worker:

```bash
python -m qa_agent.worker > /tmp/qa_worker.log 2>&1 &
```

Wait until: `python -c "import socket; s=socket.create_connection(('localhost',7233),timeout=2)"` succeeds and the worker process is alive.

- [ ] **Step 2: Run the RAGAS eval**

```bash
QA_RUN_INTEGRATION=1 python -u ragas_evals/run_ragas_eval.py --enforce-id-thresholds
```

Expected exit 0 and a fresh result file under `ragas_evals/results/`.

- [ ] **Step 3: Verify acceptance criteria**

Open the new `ragas_eval_<timestamp>.md`. Confirm:

- q3's row in the JSONL has at least one of `dd7564204c2946d7` / `5f644bc9ef3abd5f` in its retrieved IDs **and** in its `cited_ids`.
- q3's `faithfulness`, `answer_relevancy`, `semantic_similarity` are each `> 0.5`.
- Overall averages: `answer_relevancy >= 0.75`, `semantic_similarity >= 0.70`, `faithfulness >= 0.85`, `context_precision >= 0.70`, `context_recall >= 0.70`.
- No regression: every other row's `id_hit_at_8 == 1.0`.

If any criterion fails: capture the failing q3 row from the JSONL, re-read [docs/superpowers/specs/2026-05-07-code-examples-expansion-design.md](../specs/2026-05-07-code-examples-expansion-design.md) "Risks" section, and use systematic-debugging before proposing changes.

- [ ] **Step 4: No commit needed (verification only)**

---

## Task 12: Update the eval plan with the new pattern (housekeeping)

**Files:**
- Modify: `ragas_evals/EVALUATION_PLAN.md`

The eval plan documents the metrics and thresholds; it should mention `code_examples` so the next reader knows why q3 scores moved.

- [ ] **Step 1: Read the current eval plan**

Run: `head -120 ragas_evals/EVALUATION_PLAN.md`
Expected: shows the metrics and thresholds sections.

- [ ] **Step 2: Add a one-paragraph note**

Append (or insert below the "Metrics" section) a new section:

```markdown
## Retrieval patterns relevant to this eval

- `code_examples` (added 2026-05-07): graph-traversal expansion that surfaces
  CodeBlocks for entity seeds. The planner emits this when the question asks
  for an example. Q3 (Edit Python example) depends on this pattern; without
  it, retrieval falls back to BM25/vector over CodeBlocks and the gold
  quickstart codeblocks do not surface.
```

- [ ] **Step 3: Commit**

```bash
git add ragas_evals/EVALUATION_PLAN.md
git commit -m "docs(eval): note code_examples expansion in plan"
```

---

## Self-Review

**Spec coverage:**

- ✅ Cypher template (spec §"Cypher template") → Task 4.
- ✅ ORDER BY rel_rank/section_depth/cb_position (spec §"Cypher template" notes) → Task 4.
- ✅ Schema: `ExpansionName` Literal extension (spec §"Schema changes" 1) → Task 1.
- ✅ Schema: `max_per_seed` `le=8` bump (spec §"Schema changes" 2) → Task 2.
- ✅ Schema: `language` field (spec §"Schema changes" 3) → Task 3.
- ✅ Expansion router registration (spec §"Expansion wiring") → Task 5.
- ✅ `language` cypher-param passthrough (spec §"Expansion wiring") → Task 6.
- ✅ Planner prompt: output schema block update (spec §"Planner prompt update" 1) → Task 7 step 3.
- ✅ Planner prompt: code_examples bullet (spec §"Planner prompt update" 2) → Task 7 step 4.
- ✅ Planner prompt: q3 example replacement (spec §"Planner prompt update" 3) → Task 7 step 5.
- ✅ Tests: schemas validation (spec §"Tests") → Tasks 1–3.
- ✅ Tests: expansion router (spec §"Tests") → Tasks 5, 6, 8.
- ✅ Tests: planner prompt content (spec §"Tests") → Task 7.
- ✅ Tests: graph integration (spec §"Tests") → Task 9.
- ✅ Acceptance criteria (spec §"Verification") → Task 11.
- ✅ Probe re-run (spec §"Pre-implementation probe") → Task 9 (shape matches the spec's verify probe).

**Placeholder scan:** No "TBD", no "TODO", no "implement later", no "similar to Task N" without code. Each step has either a concrete file path + content, or a concrete command + expected output. ✅

**Type consistency:** `ExpansionName` Literal name, `ExpansionPattern.language: str | None`, cypher template name `expand_code_examples.cypher`, registry key `code_examples`, planner prompt key `code_examples` — consistent throughout. Test imports use the names defined in earlier tasks. ✅

---

**Plan ends.**
