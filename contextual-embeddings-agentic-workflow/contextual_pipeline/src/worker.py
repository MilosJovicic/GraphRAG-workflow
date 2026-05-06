import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
from .config import get_settings
from .activities.fetch import fetch_node_ids, fetch_nodes_batch
from .activities.generate_context import generate_context
from .activities.embed import embed_batch
from .activities.write import write_results
from .workflows.per_type import PerTypeWorkflow
from .workflows.parent import ContextualizationWorkflow

# beartype (used by pydantic-ai) conflicts with Temporal's sandbox import machinery
_SANDBOX_RUNNER = SandboxedWorkflowRunner(
    restrictions=SandboxRestrictions.default.with_passthrough_modules("beartype")
)


async def main():
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)
    worker = Worker(
        client,
        task_queue="contextual-pipeline",
        workflows=[ContextualizationWorkflow, PerTypeWorkflow],
        activities=[
            fetch_node_ids,
            fetch_nodes_batch,
            generate_context,
            embed_batch,
            write_results,
        ],
        workflow_runner=_SANDBOX_RUNNER,
    )
    print(f"Worker starting — task queue: contextual-pipeline  namespace: {s.temporal_namespace}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
