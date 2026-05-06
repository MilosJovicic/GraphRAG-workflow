from datetime import timedelta
from typing import Literal
from temporalio import activity
from temporalio.exceptions import ApplicationError
from ..neo4j_client import load_cypher, run_query_async
from ..schemas import NodePayload

ResumeMode = Literal["all", "missing_only", "by_source"]

LABEL_TO_IDS_CYPHER: dict[str, str] = {
    "Section":        "fetch_ids/sections.cypher",
    "CodeBlock":      "fetch_ids/codeblocks.cypher",
    "TableRow":       "fetch_ids/tablerows.cypher",
    "Callout":        "fetch_ids/callouts.cypher",
    "Tool":           "fetch_ids/tools.cypher",
    "Hook":           "fetch_ids/hooks.cypher",
    "SettingKey":     "fetch_ids/settingkeys.cypher",
    "PermissionMode": "fetch_ids/permissionmodes.cypher",
    "MessageType":    "fetch_ids/messagetypes.cypher",
    "Provider":       "fetch_ids/providers.cypher",
}

LABEL_TO_NODES_CYPHER: dict[str, str] = {
    "Section":        "fetch_nodes/sections.cypher",
    "CodeBlock":      "fetch_nodes/codeblocks.cypher",
    "TableRow":       "fetch_nodes/tablerows.cypher",
    "Callout":        "fetch_nodes/callouts.cypher",
    "Tool":           "fetch_nodes/entities.cypher",
    "Hook":           "fetch_nodes/entities.cypher",
    "SettingKey":     "fetch_nodes/entities.cypher",
    "PermissionMode": "fetch_nodes/entities.cypher",
    "MessageType":    "fetch_nodes/entities.cypher",
    "Provider":       "fetch_nodes/entities.cypher",
}


def _row_to_payload(label: str, row: dict) -> NodePayload:
    extras: dict[str, str] = {}

    if label == "Section":
        extras = {"synthesized": str(row.get("synthesized", "false")).lower()}

    elif label == "CodeBlock":
        extras = {
            "language": row.get("language") or "code",
            "preceding": row.get("preceding") or "",
        }

    elif label == "TableRow":
        raw_headers = row.get("headers") or []
        raw_cells = row.get("cells") or []
        if isinstance(raw_headers, list):
            raw_headers = "|".join(str(h) for h in raw_headers)
        if isinstance(raw_cells, list):
            raw_cells = "|".join(str(c) for c in raw_cells)
        extras = {"headers": raw_headers, "cells": raw_cells}

    elif label == "Callout":
        extras = {"callout_kind": row.get("callout_kind") or "note"}

    else:
        extras = {"name": row.get("name") or row.get("node_id", "")}
        if row.get("defined_in"):
            extras["defined_in"] = str(row["defined_in"])
        if row.get("definition_text"):
            extras["definition_text"] = str(row["definition_text"])

    related = row.get("related_hints") or []
    if not isinstance(related, list):
        related = []
    related = [str(x) for x in related if x is not None]

    return NodePayload(
        node_id=str(row["node_id"]),
        label=label,
        raw_text=row.get("raw_text") or "",
        page_title=row.get("page_title"),
        page_description=row.get("page_description"),
        breadcrumb=row.get("breadcrumb"),
        parent_text=row.get("parent_text"),
        related_hints=related,
        extras=extras,
    )


@activity.defn
async def fetch_node_ids(
    label: str,
    resume_mode: ResumeMode = "all",
    target_sources: list[str] | None = None,
) -> list[str]:
    cypher_path = LABEL_TO_IDS_CYPHER.get(label)
    if not cypher_path:
        raise ApplicationError(f"Unknown label: {label}", non_retryable=True)
    cypher = load_cypher(cypher_path)
    params = {
        "resume_mode": resume_mode,
        "target_sources": target_sources or [],
    }
    rows = await run_query_async(cypher, params)
    return [str(r["node_id"]) for r in rows]


@activity.defn
async def fetch_nodes_batch(label: str, ids: list[str]) -> list[NodePayload]:
    if not ids:
        return []
    cypher_path = LABEL_TO_NODES_CYPHER.get(label)
    if not cypher_path:
        raise ApplicationError(f"Unknown label: {label}", non_retryable=True)
    cypher = load_cypher(cypher_path)

    try:
        rows = await run_query_async(cypher, {"label": label, "ids": ids})
    except Exception as exc:
        raise ApplicationError(f"Neo4j fetch failed for {label}: {exc}") from exc

    payloads = []
    for row in rows:
        row_label = row.get("label") or label
        try:
            payloads.append(_row_to_payload(row_label, row))
        except Exception as exc:
            raise ApplicationError(
                f"Malformed Neo4j row {row.get('node_id')}: {exc}",
                non_retryable=True,
            ) from exc
    return payloads
