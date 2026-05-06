"""
Usage (run from contextual_pipeline/):
  python -m src.starter                                    # full run, all labels
  python -m src.starter --pilot 50                         # 50 nodes per label (pilot)
  python -m src.starter --resume missing_only              # only nodes with no context
  python -m src.starter --resume by_source --sources llm   # re-run LLM nodes only
  python -m src.starter --labels CodeBlock Tool            # specific labels only
"""
import asyncio
import argparse
import time
from temporalio.client import Client
from .config import get_settings
from .workflows.parent import ContextualizationWorkflow, PipelineResult


async def main(args: argparse.Namespace):
    s = get_settings()
    client = await Client.connect(s.temporal_host, namespace=s.temporal_namespace)

    workflow_id = f"contextual-pipeline-{int(time.time())}"
    handle = await client.start_workflow(
        ContextualizationWorkflow.run,
        args=[args.resume, args.sources or None, args.pilot, args.labels or None],
        id=workflow_id,
        task_queue="contextual-pipeline",
    )
    print(f"Started workflow: {handle.id}")
    print("Waiting for completion...")

    result: PipelineResult = await handle.result()

    print("\n=== Pipeline Result ===")
    for label, stats in result.by_label.items():
        print(f"\n{label}:")
        for k, v in stats.items():
            if k != "failed_ids":
                print(f"  {k}: {v}")
        if stats.get("failed_ids"):
            print(f"  failed_ids: {stats['failed_ids']}")
    print(f"\nTotal child workflows: {result.total_child_workflows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the contextual pipeline")
    parser.add_argument("--resume", default="all",
                        choices=["all", "missing_only", "by_source"],
                        help="Resume mode")
    parser.add_argument("--sources", nargs="+",
                        help="context_source values to re-process (for by_source mode)")
    parser.add_argument("--pilot", type=int, default=None,
                        help="Process at most N nodes per label (pilot mode)")
    parser.add_argument("--labels", nargs="+",
                        help="Only process these labels")
    asyncio.run(main(parser.parse_args()))
