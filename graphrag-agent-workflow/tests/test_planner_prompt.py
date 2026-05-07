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
