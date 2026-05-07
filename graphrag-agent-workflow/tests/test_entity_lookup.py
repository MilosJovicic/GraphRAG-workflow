from unittest.mock import AsyncMock, patch

import pytest

from qa_agent.retrieval.entity import ENTITY_LABELS, entity_lookup
from qa_agent.schemas import SubQuery


@pytest.mark.asyncio
async def test_entity_lookup_calls_correct_template_for_tool():
    sq = SubQuery(
        text="configure Bash tool permission mode",
        target_labels=["Tool"],
        bm25_keywords=["Bash", "permission"],
    )
    fake_records = [
        {
            "node_id": "Tool:Bash",
            "node_label": "Tool",
            "indexed_text": "Bash tool",
            "raw_text": "Run shell commands",
            "url": "https://code.claude.com/docs/en/permissions.md",
            "anchor": "bash",
            "breadcrumb": "Permissions > Bash",
            "title": "Bash",
            "bm25_score": 3.0,
        }
    ]

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=fake_records),
    ) as mocked_run:
        out = await entity_lookup(sq, label="Tool", limit=10)

    assert len(out) == 1
    assert out[0].node_id == "Tool:Bash"
    assert out[0].node_label == "Tool"
    assert out[0].bm25_score == 3.0

    args, _ = mocked_run.call_args
    assert args[0] == "lookup_tool.cypher"
    params = args[1]
    assert params["limit"] == 10
    lower_terms = [t.lower() for t in params["terms"]]
    assert "bash" in lower_terms
    assert "permission" in lower_terms


@pytest.mark.asyncio
async def test_entity_lookup_calls_settingkey_template():
    sq = SubQuery(
        text="permissions setting",
        target_labels=["SettingKey"],
        bm25_keywords=["permissions"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await entity_lookup(sq, label="SettingKey", limit=5)

    args, _ = mocked_run.call_args
    assert args[0] == "lookup_settingkey.cypher"


@pytest.mark.asyncio
async def test_entity_lookup_dispatches_template_for_each_entity_label():
    sq = SubQuery(text="bash", target_labels=["Tool"], bm25_keywords=["bash"])
    expected = {
        "Tool": "lookup_tool.cypher",
        "Hook": "lookup_hook.cypher",
        "SettingKey": "lookup_settingkey.cypher",
        "PermissionMode": "lookup_permissionmode.cypher",
        "Provider": "lookup_provider.cypher",
        "MessageType": "lookup_messagetype.cypher",
    }
    assert set(expected) == ENTITY_LABELS

    for label, template in expected.items():
        with patch(
            "qa_agent.retrieval.entity.run_cypher",
            new=AsyncMock(return_value=[]),
        ) as mocked_run:
            await entity_lookup(sq, label=label, limit=1)
        args, _ = mocked_run.call_args
        assert args[0] == template, f"label {label} should call {template}"


@pytest.mark.asyncio
async def test_entity_lookup_unknown_label_raises():
    sq = SubQuery(text="bash", target_labels=["Tool"], bm25_keywords=["bash"])

    with pytest.raises(ValueError):
        await entity_lookup(sq, label="Section", limit=10)


@pytest.mark.asyncio
async def test_entity_lookup_combines_text_tokens_and_keywords():
    sq = SubQuery(
        text="set model session",
        target_labels=["SettingKey"],
        bm25_keywords=["model", "availableModels", "settings.json"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await entity_lookup(sq, label="SettingKey", limit=5)

    args, _ = mocked_run.call_args
    terms = args[1]["terms"]
    lower = {t.lower() for t in terms}
    assert {"set", "model", "session"} <= lower
    assert {"availablemodels", "settings.json"} <= lower


@pytest.mark.asyncio
async def test_entity_lookup_splits_multi_word_keywords():
    sq = SubQuery(
        text="hooks",
        target_labels=["Hook"],
        bm25_keywords=["hook events", "PostToolUse"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await entity_lookup(sq, label="Hook", limit=5)

    args, _ = mocked_run.call_args
    terms = {t.lower() for t in args[1]["terms"]}
    assert "hook" in terms
    assert "events" in terms
    assert "posttooluse" in terms


@pytest.mark.asyncio
async def test_entity_lookup_dedupes_terms_case_insensitively():
    sq = SubQuery(
        text="Bash bash BASH",
        target_labels=["Tool"],
        bm25_keywords=["Bash"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await entity_lookup(sq, label="Tool", limit=5)

    args, _ = mocked_run.call_args
    terms = args[1]["terms"]
    lower = [t.lower() for t in terms]
    assert lower.count("bash") == 1


@pytest.mark.asyncio
async def test_entity_lookup_drops_short_tokens():
    sq = SubQuery(
        text="a b in on the Bash",
        target_labels=["Tool"],
        bm25_keywords=["a"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        await entity_lookup(sq, label="Tool", limit=5)

    args, _ = mocked_run.call_args
    terms = args[1]["terms"]
    assert all(len(t) >= 2 for t in terms)
    assert "Bash" in terms


@pytest.mark.asyncio
async def test_entity_lookup_returns_empty_when_no_usable_terms():
    sq = SubQuery(
        text="a a a",
        target_labels=["Tool"],
        bm25_keywords=["a"],
    )

    with patch(
        "qa_agent.retrieval.entity.run_cypher",
        new=AsyncMock(return_value=[]),
    ) as mocked_run:
        out = await entity_lookup(sq, label="Tool", limit=5)

    assert out == []
    mocked_run.assert_not_called()
