from datetime import timedelta
from dataclasses import dataclass, field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..schemas import NodePayload, ContextResult
    from ..routing import decide_route, build_raw_text_repr, templater_for
    from ..config import get_settings
    from ..activities.fetch import fetch_nodes_batch
    from ..activities.generate_context import generate_context
    from ..activities.embed import embed_batch
    from ..activities.write import write_results

_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
_SHORT_TO = timedelta(minutes=2)
_EMBED_TO = timedelta(minutes=10)
_WRITE_TO = timedelta(minutes=2)
_LLM_TO = timedelta(minutes=3)


@dataclass
class PerTypeResult:
    label: str
    processed: int = 0
    skipped: int = 0
    templated: int = 0
    llm_called: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)


@workflow.defn
class PerTypeWorkflow:
    @workflow.run
    async def run(self, label: str, ids: list[str]) -> PerTypeResult:
        stats = PerTypeResult(label=label)
        settings = get_settings()

        payloads: list[NodePayload] = await workflow.execute_activity(
            fetch_nodes_batch,
            args=[label, ids],
            start_to_close_timeout=_SHORT_TO,
            retry_policy=_RETRY,
        )

        rows_to_write: list[dict] = []
        to_embed_indices: list[int] = []

        for payload in payloads:
            route = decide_route(payload)
            raw_repr = build_raw_text_repr(payload)

            if route == "skip":
                rows_to_write.append({
                    "label": payload.label,
                    "node_id": payload.node_id,
                    "context": "",
                    "indexed_text": "",
                    "embedding": None,
                    "context_source": "skipped",
                    "context_version": settings.context_version,
                })
                stats.skipped += 1
                continue

            if route == "template":
                ctx = templater_for(payload.label)(payload)
                rows_to_write.append({
                    "label": payload.label,
                    "node_id": payload.node_id,
                    "context": ctx,
                    "indexed_text": ctx + "\n\n" + raw_repr,
                    "embedding": None,
                    "context_source": "template",
                    "context_version": settings.context_version,
                })
                to_embed_indices.append(len(rows_to_write) - 1)
                stats.templated += 1
                continue

            # llm
            try:
                ctx_result: ContextResult = await workflow.execute_activity(
                    generate_context,
                    args=[payload],
                    start_to_close_timeout=_LLM_TO,
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                ctx = ctx_result.context
                rows_to_write.append({
                    "label": payload.label,
                    "node_id": payload.node_id,
                    "context": ctx,
                    "indexed_text": ctx + "\n\n" + raw_repr,
                    "embedding": None,
                    "context_source": "llm",
                    "context_version": settings.context_version,
                })
                to_embed_indices.append(len(rows_to_write) - 1)
                stats.llm_called += 1
            except Exception as exc:
                workflow.logger.error(f"LLM failed for {payload.node_id}: {exc}")
                stats.failed += 1
                stats.failed_ids.append(payload.node_id)

        # Batch embed all non-skipped rows
        if to_embed_indices:
            texts = [rows_to_write[i]["indexed_text"] for i in to_embed_indices]
            embeddings: list[list[float]] = await workflow.execute_activity(
                embed_batch,
                args=[texts],
                start_to_close_timeout=_EMBED_TO,
                retry_policy=_RETRY,
            )
            if len(embeddings) != len(to_embed_indices):
                raise RuntimeError(
                    "Embedding count mismatch: "
                    f"got {len(embeddings)} for {len(to_embed_indices)} texts"
                )
            for idx, emb in zip(to_embed_indices, embeddings):
                rows_to_write[idx]["embedding"] = emb

        # Write all successful rows (skipped rows get NULL embedding).
        if rows_to_write:
            await workflow.execute_activity(
                write_results,
                args=[rows_to_write],
                start_to_close_timeout=_WRITE_TO,
                retry_policy=_RETRY,
            )

        stats.processed = len(payloads)
        return stats
