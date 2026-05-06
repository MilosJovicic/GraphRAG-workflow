from datetime import timedelta
from dataclasses import dataclass, field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..activities.fetch import fetch_node_ids, ResumeMode
    from .per_type import PerTypeWorkflow, PerTypeResult

LABELS = [
    "CodeBlock", "TableRow", "Section", "Callout",
    "Tool", "Hook", "SettingKey", "PermissionMode", "MessageType", "Provider",
]
BATCH_SIZE = 50
CONTINUE_AS_NEW_AFTER = 500


@dataclass
class PipelineResult:
    by_label: dict[str, dict] = field(default_factory=dict)
    total_child_workflows: int = 0


@workflow.defn
class ContextualizationWorkflow:
    @workflow.run
    async def run(
        self,
        resume_mode: ResumeMode = "all",
        target_sources: list[str] | None = None,
        pilot_limit: int | None = None,
        labels: list[str] | None = None,
    ) -> PipelineResult:
        result = PipelineResult()
        child_count = 0
        active_labels = labels or LABELS

        for label in active_labels:
            all_ids: list[str] = await workflow.execute_activity(
                fetch_node_ids,
                args=[label, resume_mode, target_sources],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if pilot_limit:
                all_ids = all_ids[:pilot_limit]

            batches = [
                all_ids[i : i + BATCH_SIZE]
                for i in range(0, len(all_ids), BATCH_SIZE)
            ]
            label_stats: dict = {
                "processed": 0, "skipped": 0, "templated": 0,
                "llm_called": 0, "failed": 0, "failed_ids": [],
            }

            for batch in batches:
                child_result: PerTypeResult = await workflow.execute_child_workflow(
                    PerTypeWorkflow.run,
                    args=[label, batch],
                    id=f"per-type-{label}-{workflow.now().timestamp():.3f}",
                )
                label_stats["processed"] += child_result.processed
                label_stats["skipped"] += child_result.skipped
                label_stats["templated"] += child_result.templated
                label_stats["llm_called"] += child_result.llm_called
                label_stats["failed"] += child_result.failed
                label_stats["failed_ids"].extend(child_result.failed_ids)
                child_count += 1

                if child_count >= CONTINUE_AS_NEW_AFTER:
                    # Preserve partial results by continuing from current label onwards
                    remaining_labels = active_labels[active_labels.index(label):]
                    workflow.continue_as_new(
                        resume_mode=resume_mode,
                        target_sources=target_sources,
                        pilot_limit=pilot_limit,
                        labels=remaining_labels,
                    )

            result.by_label[label] = label_stats

        result.total_child_workflows = child_count
        return result
