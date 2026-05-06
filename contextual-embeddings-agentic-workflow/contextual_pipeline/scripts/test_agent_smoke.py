"""
Run from contextual_pipeline/ with:
  python scripts/test_agent_smoke.py
Tests the PydanticAI agent against 5 hand-picked NodePayload fixtures.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import NodePayload
from src.agents.context_agent import generate_context

FIXTURES = [
    NodePayload(
        node_id="tool_bash",
        label="Tool",
        raw_text="Executes a given bash command and returns its output.",
        page_title="Tools Reference",
        breadcrumb="Tools Reference > Bash",
        extras={"name": "Bash"},
    ),
    NodePayload(
        node_id="s_permissions",
        label="Section",
        raw_text="Claude Code uses a permission system to control which operations are allowed. Each tool requires explicit permission before it can run.",
        page_title="Security",
        breadcrumb="Security > Permissions > Overview",
    ),
    NodePayload(
        node_id="cb_install",
        label="CodeBlock",
        raw_text="npm install -g @anthropic-ai/claude-code",
        page_title="Getting Started",
        breadcrumb="Getting Started > Installation",
        extras={"language": "bash", "preceding": "Install Claude Code globally:"},
    ),
    NodePayload(
        node_id="hook_prestop",
        label="Hook",
        raw_text="Fires before Claude stops responding. Use to run cleanup or save state.",
        page_title="Hooks",
        breadcrumb="Hooks > PreStop",
        extras={"name": "PreStop"},
    ),
    NodePayload(
        node_id="sk_model",
        label="SettingKey",
        raw_text="The Claude model to use for responses.",
        page_title="Settings",
        breadcrumb="Settings > model",
        extras={"name": "model"},
    ),
]


async def main():
    for fixture in FIXTURES:
        print(f"\n{'='*60}")
        print(f"Node: {fixture.node_id} ({fixture.label})")
        print(f"Raw:  {fixture.raw_text[:80]}")
        result = await generate_context(fixture)
        print(f"Context ({len(result.context)} chars): {result.context}")
    print("\nAll 5 fixtures processed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
