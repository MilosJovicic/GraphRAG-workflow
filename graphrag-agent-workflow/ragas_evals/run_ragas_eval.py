"""Run RAGAS and deterministic retrieval evals for the Q&A agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

from qa_agent.config import get_settings
from qa_agent.schemas import QAResponse

if TYPE_CHECKING:  # pragma: no cover - import only for static type-check
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = Path(__file__).with_name("gold_questions.yaml")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

RAGAS_METRIC_NAMES = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "factual_correctness",
    "semantic_similarity",
]


def _normalize_expected_groups(raw: list[Any]) -> list[list[str]]:
    """Each gold slot is a group of acceptable IDs (any-of). A plain string
    becomes a single-item group; a list becomes a multi-alternative group.
    """
    out: list[list[str]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append([entry])
        elif isinstance(entry, list):
            members = [str(x) for x in entry if str(x).strip()]
            if not members:
                raise ValueError("expected_context_ids contains an empty group")
            out.append(members)
        else:
            raise ValueError(
                "expected_context_ids entries must be a string or list of strings, "
                f"got {type(entry).__name__}"
            )
    return out


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_context_ids: list[list[str]]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "GoldQuestion":
        required = {
            "id",
            "category",
            "question",
            "reference_answer",
            "expected_context_ids",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"gold sample missing fields: {', '.join(missing)}")

        expected_ids = data["expected_context_ids"]
        if not isinstance(expected_ids, list) or not expected_ids:
            raise ValueError(f"gold sample {data['id']!r} needs expected_context_ids")

        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            question=str(data["question"]),
            reference_answer=str(data["reference_answer"]),
            expected_context_ids=_normalize_expected_groups(expected_ids),
        )


def load_gold_questions(path: Path = DEFAULT_GOLD_PATH) -> list[GoldQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list of gold samples")

    samples = [GoldQuestion.from_mapping(item) for item in raw]
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("gold sample ids must be unique")
    return samples


def build_retrieved_contexts(response: QAResponse, max_contexts: int = 8) -> list[str]:
    contexts: list[str] = []
    candidates = response.retrieved or []

    for candidate in candidates[:max_contexts]:
        text = (candidate.indexed_text or candidate.raw_text or "").strip()
        if not text:
            continue
        title = (candidate.title or candidate.breadcrumb or candidate.node_id).strip()
        contexts.append(f"[{candidate.node_label}] {title}\n{text}")

    if contexts:
        return contexts

    for citation in response.citations[:max_contexts]:
        text = citation.snippet.strip()
        if not text:
            continue
        title = (citation.title or citation.node_id).strip()
        contexts.append(f"[{citation.node_label}] {title}\n{text}")

    return contexts


def compute_id_metrics(
    response: QAResponse,
    expected_context_ids: list[Any],
    max_contexts: int = 8,
) -> dict[str, float | int]:
    groups = _normalize_expected_groups(list(expected_context_ids))
    retrieved_ids = {
        candidate.node_id
        for candidate in (response.retrieved or [])[:max_contexts]
    }
    cited_ids = {citation.node_id for citation in response.citations}

    retrieved_satisfied = sum(
        1 for group in groups if any(member in retrieved_ids for member in group)
    )
    cited_satisfied = sum(
        1 for group in groups if any(member in cited_ids for member in group)
    )

    total = len(groups)
    metrics: dict[str, float | int] = {
        "id_hit_at_8": 1.0 if retrieved_satisfied else 0.0,
        "id_recall_at_8": retrieved_satisfied / total if total else 0.0,
        "cited_expected_hit": 1.0 if cited_satisfied else 0.0,
        "citation_count": len(response.citations),
        "no_results": 1.0 if response.no_results else 0.0,
        "fallback_used": 1.0 if response.fallback_used else 0.0,
    }

    for key, value in response.latency_ms.items():
        metrics[f"latency_{key}_ms"] = value

    return metrics


def _metric_value(result: Any) -> float:
    value = getattr(result, "value", result)
    if value is None:
        return math.nan
    return float(value)


def ensure_openai_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running RAGAS evals")
    return api_key


def create_ragas_scorers(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
        SemanticSimilarity,
    )

    api_key = ensure_openai_api_key()
    llm_client = AsyncOpenAI(api_key=api_key, timeout=60.0)
    embedding_client = OpenAI(api_key=api_key, timeout=30.0)
    judge_llm = llm_factory(judge_model, provider="openai", client=llm_client)
    async_embeddings = OpenAIEmbeddings(client=llm_client, model=embedding_model)
    sync_embeddings = OpenAIEmbeddings(client=embedding_client, model=embedding_model)

    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=async_embeddings),
        "factual_correctness": FactualCorrectness(llm=judge_llm),
        "semantic_similarity": SemanticSimilarity(embeddings=sync_embeddings),
    }


async def score_with_ragas(
    sample: GoldQuestion,
    response: QAResponse,
    contexts: list[str],
    scorers: dict[str, Any],
    raise_exceptions: bool = False,
    metric_timeout_seconds: float = 90.0,
) -> tuple[dict[str, float], dict[str, str]]:
    safe_contexts = contexts or [""]
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    calls = {
        "faithfulness": {
            "user_input": sample.question,
            "response": response.answer,
            "retrieved_contexts": safe_contexts,
        },
        "context_precision": {
            "user_input": sample.question,
            "reference": sample.reference_answer,
            "retrieved_contexts": safe_contexts,
        },
        "context_recall": {
            "user_input": sample.question,
            "reference": sample.reference_answer,
            "retrieved_contexts": safe_contexts,
        },
        "answer_relevancy": {
            "user_input": sample.question,
            "response": response.answer,
        },
        "factual_correctness": {
            "response": response.answer,
            "reference": sample.reference_answer,
        },
        "semantic_similarity": {
            "response": response.answer,
            "reference": sample.reference_answer,
        },
    }

    for metric_name in RAGAS_METRIC_NAMES:
        print(f"    scoring {metric_name}", flush=True)
        try:
            result = await asyncio.wait_for(
                scorers[metric_name].ascore(**calls[metric_name]),
                timeout=metric_timeout_seconds,
            )
            scores[metric_name] = _metric_value(result)
            print(f"    {metric_name}={scores[metric_name]:.3f}", flush=True)
        except TimeoutError:
            if raise_exceptions:
                raise
            scores[metric_name] = math.nan
            errors[metric_name] = (
                f"timed out after {metric_timeout_seconds:.0f} seconds"
            )
            print(f"    {metric_name}=nan ({errors[metric_name]})", flush=True)
        except Exception as exc:
            if raise_exceptions:
                raise
            scores[metric_name] = math.nan
            errors[metric_name] = str(exc)
            print(f"    {metric_name}=nan ({type(exc).__name__})", flush=True)

    return scores, errors


async def run_question(client: Any, sample: GoldQuestion) -> QAResponse:
    from qa_agent.schemas import QARequest
    from qa_agent.workflows.qa import QAWorkflow

    settings = get_settings()
    return await client.execute_workflow(
        QAWorkflow.run,
        QARequest(question=sample.question, debug=True),
        id=f"qa-ragas-{sample.id}-{uuid.uuid4().hex[:8]}",
        task_queue=settings.qa_task_queue,
        execution_timeout=timedelta(minutes=5),
        result_type=QAResponse,
    )


def write_outputs(
    rows: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, Path, Path]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"ragas_eval_{timestamp}.jsonl"
    csv_path = output_dir / f"ragas_eval_{timestamp}.csv"
    summary_path = output_dir / f"ragas_eval_{timestamp}.md"

    jsonl_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in raw_records)
        + "\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    summary_path.write_text(build_summary(df, jsonl_path, csv_path), encoding="utf-8")
    return jsonl_path, csv_path, summary_path


def build_summary(df: "pd.DataFrame", jsonl_path: Path, csv_path: Path) -> str:
    import pandas as pd

    lines = [
        "# RAGAS Eval Summary",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Samples: {len(df)}",
        f"Raw JSONL: `{jsonl_path}`",
        f"Row CSV: `{csv_path}`",
        "",
        "## Averages",
        "",
    ]

    average_cols = [
        "id_hit_at_8",
        "id_recall_at_8",
        "cited_expected_hit",
        *RAGAS_METRIC_NAMES,
        "citation_count",
        "fallback_used",
        "no_results",
    ]
    for col in average_cols:
        if col in df:
            value = pd.to_numeric(df[col], errors="coerce").mean()
            lines.append(f"- `{col}`: {value:.3f}")

    lines.extend(["", "## Category Breakdown", ""])
    category_cols = [col for col in average_cols if col in df and col != "category"]
    if "category" in df and category_cols:
        category_df = df.groupby("category")[category_cols].mean(numeric_only=True)
        lines.append(category_df.to_markdown(floatfmt=".3f"))
    else:
        lines.append("No category data available.")

    lines.extend(["", "## ID Hit Failures", ""])
    if "id_hit_at_8" in df:
        failures = df[pd.to_numeric(df["id_hit_at_8"], errors="coerce") < 1.0]
        if failures.empty:
            lines.append("No deterministic ID hit failures.")
        else:
            lines.append(
                failures[
                    ["id", "category", "question", "id_recall_at_8", "fallback_used"]
                ].to_markdown(index=False, floatfmt=".3f")
            )
    else:
        lines.append("No ID metrics were recorded.")

    return "\n".join(lines) + "\n"


async def run_eval(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    load_dotenv()
    if os.getenv("QA_RUN_INTEGRATION") != "1" and not args.allow_without_integration_env:
        raise SystemExit(
            "Set QA_RUN_INTEGRATION=1 before running live evals, or pass "
            "--allow-without-integration-env."
        )

    get_settings.cache_clear()
    settings = get_settings()
    samples = load_gold_questions(args.gold_path)
    if args.limit:
        samples = samples[: args.limit]

    scorers = None if args.skip_ragas else create_ragas_scorers(
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )

    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] Running workflow for {sample.id}", flush=True)
        response = await run_question(client, sample)
        contexts = build_retrieved_contexts(response)
        id_metrics = compute_id_metrics(response, sample.expected_context_ids)
        ragas_scores: dict[str, float] = {}
        ragas_errors: dict[str, str] = {}
        if scorers is not None:
            print(f"[{index}/{len(samples)}] Scoring RAGAS for {sample.id}", flush=True)
            ragas_scores, ragas_errors = await score_with_ragas(
                sample,
                response,
                contexts,
                scorers,
                raise_exceptions=args.raise_ragas_exceptions,
                metric_timeout_seconds=args.metric_timeout_seconds,
            )

        row = {
            "id": sample.id,
            "category": sample.category,
            "question": sample.question,
            **id_metrics,
            **ragas_scores,
            "ragas_errors_json": json.dumps(ragas_errors, sort_keys=True),
        }
        rows.append(row)
        raw_records.append(
            {
                "sample": asdict(sample),
                "response": response.model_dump(mode="json"),
                "retrieved_contexts": contexts,
                "scores": row,
            }
        )
        print(f"[{index}/{len(samples)}] Completed {sample.id}", flush=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    paths = write_outputs(rows, raw_records, args.output_dir, timestamp)

    if args.enforce_id_thresholds and rows:
        hit_rate = sum(float(row["id_hit_at_8"]) for row in rows) / len(rows)
        if hit_rate < args.min_id_hit_rate:
            raise SystemExit(
                f"id_hit_at_8 average {hit_rate:.3f} is below "
                f"{args.min_id_hit_rate:.3f}; see {paths[2]}"
            )

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--raise-ragas-exceptions", action="store_true")
    parser.add_argument("--metric-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--enforce-id-thresholds", action="store_true")
    parser.add_argument("--min-id-hit-rate", type=float, default=1.0)
    parser.add_argument("--allow-without-integration-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    paths = asyncio.run(run_eval(parse_args()))
    print("RAGAS eval outputs:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
